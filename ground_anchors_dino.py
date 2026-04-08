#!/usr/bin/env python

"""
Step 3a: Ground Qwen3 object anchors and relation links using GroundingDINO.

Compatible inputs:
  - single Qwen JSON file, e.g.:
      timpact/data/anchors_qwen3.json
  - shard directory, e.g.:
      timpact/data/qwen_anchor_shards/
  - glob pattern, e.g.:
      timpact/data/qwen_anchor_shards/anchors_qwen3_shard_*.json

Output:
  - timpact/data/grounding/<id>/boxes.json
  - optional: timpact/data/grounding/<id>/debug_boxes.jpg  (if --debug-draw)

Command examples:
    python timpact/scripts/ground_anchors_dino.py \
        --anchors-json timpact/data/anchors_qwen3.json

    python timpact/scripts/ground_anchors_dino.py \
        --anchors-json /home/Student/s4826850/timpact/data/qwen_anchor_shards

    python timpact/scripts/ground_anchors_dino.py \
        --anchors-json "/home/Student/s4826850/timpact/data/qwen_anchor_shards/anchors_qwen3_shard_*.json"

Ground Qwen3 anchors with GroundingDINO and write per-image box metadata.

The pipeline expands each textual anchor into one or more grounding queries,
runs DINO, applies geometry/score filtering, optionally verifies surviving
crops with Qwen3-VL and/or CLIP, and writes the selected detections to
`timpact/data/grounding/<id>/boxes.json`.

Relation entries are preserved from Qwen3, but their boxes are inherited from
the grounded subject/object anchors rather than re-localized independently.
"""

import argparse
import glob
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

# GroundingDINO
from groundingdino.util.inference import load_image, load_model, predict

# spaCy improves phrase reduction and plurality detection, but the script
# falls back to heuristics when it is unavailable.
try:
    import spacy
except ImportError:
    spacy = None


# =============================================================================
# PATHS
# =============================================================================
ROOT = Path(__file__).resolve().parents[1]

# Default anchor payload from the Qwen3 extraction step.
ANCHORS_JSON = ROOT / "data" / "anchors_qwen3.json"

# Output root; each sample writes to `data/grounding/<id>/`.
OUT_BASE = ROOT / "data" / "grounding"


# =============================================================================
# DEFAULTS / MODEL IDS
# =============================================================================
BOX_THRESHOLD_DEFAULT = 0.25
TEXT_THRESHOLD_DEFAULT = 0.35

QWEN_VERIFY_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
CLIP_VERIFY_MODEL_ID = "openai/clip-vit-large-patch14"


# =============================================================================
# GROUNDINGDINO HF REPO DETAILS
# =============================================================================
DINO_REPO = "ShilongLiu/GroundingDINO"
DINO_CFG_NAME = "GroundingDINO_SwinB.cfg.py"

DINO_WEIGHT_CANDIDATES = [
    "groundingdino_swinb_cogcoor.pth",
    "groundingdino_swinb.pth",
]


# =============================================================================
# CONSTANTS FOR ANCHOR FILTERING / INTERPRETATION
# =============================================================================
NON_GROUNDABLE_TOKENS = {
    "something off-camera", "off-camera", "off camera", "something", "nothing", "everything",
}

PEOPLE_CATEGORIES = {"person", "people"}

ENTITY_SCENE_SENSITIVE = {
    "person", "people", "vehicle", "sign", "text", "symbol", "object", "water", "vegetation"
}

LARGE_OBJECT_CATEGORIES = {"building", "structure"}

SCENE_LEVEL_CATEGORIES = {"environment", "background"}

GROUP_HINTS = {
    "group", "crowd", "team", "line", "row", "cluster", "collection",
    "participants", "protesters", "people", "soldiers"
}

STOPWORDS_FALLBACK = {
    "a", "an", "the", "of", "with", "through", "in", "on", "at", "by", "from", "into", "over",
    "under", "behind", "near", "around", "across", "between", "during", "while", "wearing",
    "holding", "standing", "walking", "running", "sitting", "lying", "covering", "surrounded",
    "partially", "foreground", "background", "scene", "ground",
}

SCENE_TOKENS = {"scene", "background", "foreground", "view", "landscape"}


# =============================================================================
# GENERAL HELPERS
# =============================================================================
def choose_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def normalize_anchor(text: str) -> str:
    return " ".join((text or "").lower().strip().split())


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def is_non_groundable(phrase: str) -> bool:
    norm = normalize_anchor(phrase)
    return not norm or norm in NON_GROUNDABLE_TOKENS


def choose_anchor_json_path(explicit_path: str = "") -> Path:
    if explicit_path:
        return Path(explicit_path)
    return ANCHORS_JSON


def resolve_image_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (ROOT / p).resolve()


def safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Could not parse JSON: {path} ({e})")
        return None


# =============================================================================
# CATEGORY / RELATION INTERPRETATION
# =============================================================================
def semantic_tags_to_category(tags: List[Any], span: str, nlp) -> str:
    norm_tags = {normalize_anchor(str(t)) for t in tags if str(t).strip()}
    span_norm = normalize_anchor(span)

    if "person" in norm_tags:
        return "people" if is_plural_phrase(span_norm, "people", nlp) else "person"
    if "vehicle" in norm_tags:
        return "vehicle"
    if "structure" in norm_tags:
        return "structure"
    if "plant" in norm_tags:
        return "vegetation"
    if "sign_text" in norm_tags:
        return "sign"
    if "furniture" in norm_tags:
        return "object"
    if "device" in norm_tags:
        return "object"
    if "accessory" in norm_tags:
        return "object"
    if "clothing" in norm_tags:
        return "object"
    if "food" in norm_tags:
        return "object"
    if "animal" in norm_tags:
        return "object"

    return "object"


def relation_type_from_qwen3(rel_type: str) -> str:
    rel = normalize_anchor(rel_type)

    if rel in {"next_to", "behind", "in_front_of", "on", "under", "above", "below", "inside", "outside", "near", "beside"}:
        return "spatial"
    if rel in {"holding", "carried_by", "carrying", "wearing", "using", "attached_to", "overlapping"}:
        return "action"
    return "other"


# =============================================================================
# INPUT LOADING
# =============================================================================
def collect_anchor_input_files(anchor_json_path: Path) -> List[Path]:
    """
    Accept:
      - a single json file
      - a directory containing shard json files
      - a glob pattern string
    """
    anchor_json_str = str(anchor_json_path)

    if any(ch in anchor_json_str for ch in ["*", "?", "["]):
        files = sorted(Path(p) for p in glob.glob(anchor_json_str))
        return [p for p in files if p.is_file() and p.suffix.lower() == ".json"]

    if anchor_json_path.is_dir():
        files = sorted(
            [
                p for p in anchor_json_path.glob("*.json")
                if p.is_file() and not p.name.endswith(".tmp")
            ]
        )
        return files

    if anchor_json_path.is_file():
        return [anchor_json_path]

    return []


def extract_images_dir_from_meta(payload: Dict[str, Any]) -> Path:
    meta = payload.get("meta", {}) if isinstance(payload.get("meta"), dict) else {}
    images_dir_meta = str(meta.get("images_dir", "")).strip()

    if images_dir_meta:
        return resolve_image_path(images_dir_meta)

    return ROOT / "data" / "pristine_images" / "images" / "images"


def normalize_record_for_dino(rec: Dict[str, Any], images_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Normalize both old and new Qwen output record structures into the format
    expected downstream by DINO, while keeping compatibility with existing code.
    """
    if not isinstance(rec, dict):
        return None

    rec_copy = dict(rec)

    csv_fields = rec.get("csv_fields", {}) if isinstance(rec.get("csv_fields"), dict) else {}
    img_name = str(
        csv_fields.get("images", "") or csv_fields.get("image", "") or csv_fields.get("img", "")
    ).strip()

    resolved_image_path = str(images_dir / img_name) if img_name else ""

    rec_copy["resolved_image_path"] = resolved_image_path
    rec_copy["csv_fields"] = csv_fields
    rec_copy["status"] = str(rec.get("status", "")).strip().lower()
    rec_copy["outputs"] = rec.get("outputs", {}) if isinstance(rec.get("outputs"), dict) else {}
    rec_copy["id"] = str(rec.get("id", "")).strip()

    return rec_copy


def load_qwen3_records_from_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict) or "records" not in payload:
        raise ValueError("Expected Qwen anchors JSON to contain a top-level 'records' list.")

    images_dir = extract_images_dir_from_meta(payload)

    records: List[Dict[str, Any]] = []
    for rec in payload.get("records", []):
        norm_rec = normalize_record_for_dino(rec, images_dir)
        if norm_rec is not None:
            records.append(norm_rec)

    return records


def load_anchor_records(anchor_json_path: Path) -> List[Dict[str, Any]]:
    """
    Load one or more Qwen outputs, supporting:
      - old single-file output
      - new sharded outputs
    """
    files = collect_anchor_input_files(anchor_json_path)
    if not files:
        raise FileNotFoundError(f"No valid JSON input files found for: {anchor_json_path}")

    all_records: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()

    print(f"[DINO] Found {len(files)} anchor JSON file(s).")

    for fp in files:
        payload = safe_load_json(fp)
        if payload is None:
            continue

        try:
            records = load_qwen3_records_from_payload(payload)
        except Exception as e:
            print(f"[WARN] Skipping malformed anchor payload: {fp} ({e})")
            continue

        added = 0
        for rec in records:
            rid = str(rec.get("id", "")).strip()
            if rid and rid in seen_ids:
                continue
            if rid:
                seen_ids.add(rid)
            all_records.append(rec)
            added += 1

        print(f"[DINO] Loaded {added} record(s) from {fp}")

    return all_records


# =============================================================================
# NLP LOADING
# =============================================================================
def load_nlp():
    if spacy is None:
        print("[WARN] spaCy is not installed. Falling back to heuristic phrase reduction.")
        return None
    try:
        return spacy.load("en_core_web_sm")
    except Exception as e:
        print(f"[WARN] Could not load en_core_web_sm ({e}). Falling back to heuristic phrase reduction.")
        return None


# =============================================================================
# OPTIONAL VERIFIER MODEL LOADING
# =============================================================================
def load_qwen_verifier(device: str):
    try:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except ImportError as e:
        print(f"[WARN] Qwen verifier dependencies are unavailable ({e}). Skipping crop verification.")
        return None

    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            QWEN_VERIFY_MODEL_ID,
            torch_dtype="auto",
            device_map="auto" if device == "cuda" else None,
        )
        model.eval()
        processor = AutoProcessor.from_pretrained(QWEN_VERIFY_MODEL_ID)
        print(f"[QWEN] Crop verifier ready: {QWEN_VERIFY_MODEL_ID}")
        return {
            "model": model,
            "processor": processor,
            "device": device,
        }
    except Exception as e:
        print(f"[WARN] Could not load Qwen verifier ({e}). Skipping crop verification.")
        return None


def load_clip_verifier(device: str):
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as e:
        print(f"[WARN] CLIP verifier dependencies are unavailable ({e}). Skipping CLIP verification.")
        return None

    try:
        model = CLIPModel.from_pretrained(CLIP_VERIFY_MODEL_ID)
        model.eval()
        if device == "cuda":
            model = model.to(device)
        processor = CLIPProcessor.from_pretrained(CLIP_VERIFY_MODEL_ID)
        print(f"[CLIP] Crop verifier ready: {CLIP_VERIFY_MODEL_ID}")
        return {
            "model": model,
            "processor": processor,
            "device": device,
        }
    except Exception as e:
        print(f"[WARN] Could not load CLIP verifier ({e}). Skipping CLIP verification.")
        return None


# =============================================================================
# VERIFICATION HELPERS
# =============================================================================
def verify_crop_with_qwen(verifier, crop_rgb: np.ndarray, anchor_text: str, category: str) -> Dict[str, Any]:
    image = Image.fromarray(crop_rgb.astype(np.uint8), mode="RGB")
    prompt = f"""
You are verifying whether a grounding crop correctly matches a target anchor.

Anchor text: "{anchor_text}"
Anchor category: "{category}"

Return ONLY valid JSON with exactly these keys:
{{
  "match": true,
  "score": 0.0,
  "dominant_entity": "short noun phrase",
  "reason": "short phrase"
}}

Rules:
- "match" must be true only if the anchor is clearly visible and is the main intended subject of the crop.
- "match" must be false if the crop mainly shows a different object/entity, mostly background/context, or the anchor is absent.
- Use score between 0 and 1.
""".strip()

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]

    processor = verifier["processor"]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(verifier["device"])

    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        generated_ids = verifier["model"].generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )

    in_len = inputs["input_ids"].shape[1]
    out_ids = generated_ids[0, in_len:]

    raw = processor.batch_decode(
        out_ids.unsqueeze(0),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    first = raw.find("{")
    last = raw.rfind("}")
    if first == -1 or last <= first:
        return {"match": False, "score": 0.0, "dominant_entity": "", "reason": "invalid_qwen_output"}

    try:
        parsed = json.loads(raw[first:last + 1])
    except json.JSONDecodeError:
        return {"match": False, "score": 0.0, "dominant_entity": "", "reason": "invalid_qwen_json"}

    return {
        "match": bool(parsed.get("match", False)),
        "score": float(parsed.get("score", 0.0)),
        "dominant_entity": str(parsed.get("dominant_entity", "")).strip(),
        "reason": str(parsed.get("reason", "")).strip(),
    }


def verify_crop_with_clip(verifier, crop_rgb: np.ndarray, anchor_text: str, category: str) -> Dict[str, Any]:
    image = Image.fromarray(crop_rgb.astype(np.uint8), mode="RGB")

    positive_texts = [f"a photo of {anchor_text}"]
    if category and normalize_anchor(category) not in {"", "other"}:
        positive_texts.append(f"a photo of {category}")

    negative_texts = [
        "a photo of background",
        "a photo of scenery",
        "a photo of a different object",
    ]

    labels = positive_texts + negative_texts

    processor = verifier["processor"]
    model = verifier["model"]

    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
    if verifier["device"] == "cuda":
        inputs = {k: v.to(verifier["device"]) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image[0]
        probs = logits.softmax(dim=0).detach().cpu().numpy()

    pos_scores = probs[:len(positive_texts)]
    neg_scores = probs[len(positive_texts):]

    best_pos_idx = int(np.argmax(pos_scores))
    best_pos_score = float(pos_scores[best_pos_idx])
    best_neg_score = float(np.max(neg_scores)) if len(neg_scores) else 0.0

    margin = best_pos_score - best_neg_score
    match = best_pos_score >= 0.30 and margin >= 0.08

    return {
        "match": match,
        "score": best_pos_score,
        "dominant_entity": positive_texts[best_pos_idx],
        "reason": f"clip_margin={margin:.3f}",
    }


# =============================================================================
# MODEL FILE DOWNLOAD
# =============================================================================
def download_dino_files() -> Tuple[str, str]:
    cfg_path = hf_hub_download(repo_id=DINO_REPO, filename=DINO_CFG_NAME)

    last_err: Optional[Exception] = None
    for wname in DINO_WEIGHT_CANDIDATES:
        try:
            wpath = hf_hub_download(repo_id=DINO_REPO, filename=wname)
            return cfg_path, wpath
        except Exception as e:
            last_err = e

    raise RuntimeError(
        f"Could not download any GroundingDINO weights from {DINO_REPO}. "
        f"Tried: {DINO_WEIGHT_CANDIDATES}. Last error: {last_err}"
    )


# =============================================================================
# BOX / NMS UTILITIES
# =============================================================================
def compute_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0
    return inter / union


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
    if len(boxes) == 0:
        return []

    idxs = scores.argsort()[::-1]
    keep: List[int] = []

    while len(idxs) > 0:
        cur = int(idxs[0])
        keep.append(cur)
        if len(idxs) == 1:
            break

        rest = idxs[1:]
        filtered = []
        for j in rest:
            if compute_iou_xyxy(boxes[cur], boxes[int(j)]) <= iou_thresh:
                filtered.append(int(j))
        idxs = np.array(filtered, dtype=np.int64)

    return keep


def clip_box_xyxy(box: List[float], w: int, h: int) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0.0, min(x1, w - 1))
    x2 = max(0.0, min(x2, w - 1))
    y1 = max(0.0, min(y1, h - 1))
    y2 = max(0.0, min(y2, h - 1))

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


# =============================================================================
# PHRASE PROCESSING HELPERS
# =============================================================================
def phrase_tokens_fallback(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", normalize_anchor(text)) if t and t not in STOPWORDS_FALLBACK]


def singularize_token(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ses") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
        return token[:-1]
    return token


def pluralize_token(token: str) -> str:
    if token.endswith("y") and len(token) > 2 and token[-2] not in "aeiou":
        return token[:-1] + "ies"
    if token.endswith("s"):
        return token
    return token + "s"


def tokens_for_match(text: str, nlp) -> Set[str]:
    norm = normalize_anchor(text)
    if not norm:
        return set()

    if nlp is not None:
        doc = nlp(norm)
        toks = {
            (tok.lemma_ or tok.text).lower()
            for tok in doc
            if tok.is_alpha and not tok.is_stop and tok.pos_ in {"NOUN", "PROPN", "ADJ"}
        }
        if toks:
            return toks

    return set(phrase_tokens_fallback(norm))


def head_token_info(text: str, nlp) -> Tuple[str, str]:
    norm = normalize_anchor(text)
    if not norm:
        return "", ""

    if nlp is not None:
        doc = nlp(norm)
        for tok in doc:
            if tok.pos_ in {"NOUN", "PROPN"}:
                return tok.text.lower(), (tok.lemma_ or tok.text).lower()

    toks = phrase_tokens_fallback(norm)
    if not toks:
        return "", ""
    return toks[-1], singularize_token(toks[-1])


def is_plural_phrase(text: str, category: str, nlp) -> bool:
    norm = normalize_anchor(text)
    if not norm:
        return False

    if any(f" {hint} " in f" {norm} " for hint in GROUP_HINTS):
        return True

    if category == "people":
        return True

    if nlp is not None:
        doc = nlp(norm)
        for tok in doc:
            if tok.pos_ in {"NOUN", "PROPN"}:
                if tok.tag_ in {"NNS", "NNPS"}:
                    return True
                if tok.morph.get("Number") == ["Plur"]:
                    return True

    head_text, _ = head_token_info(norm, nlp)
    return bool(head_text.endswith("s") and not head_text.endswith("ss"))


def derive_candidate_queries(phrase: str, category: str, nlp) -> List[str]:
    norm = normalize_anchor(phrase)
    if not norm:
        return []

    candidates: List[str] = [norm]

    if nlp is not None:
        doc = nlp(norm)

        noun_chunks = [normalize_anchor(chunk.text) for chunk in doc.noun_chunks if normalize_anchor(chunk.text)]
        for chunk in noun_chunks:
            candidates.append(chunk)

        head_tok = None
        for tok in doc:
            if tok.pos_ in {"NOUN", "PROPN"}:
                head_tok = tok
                break

        if head_tok is not None:
            left_mods = [
                child.text.lower()
                for child in head_tok.lefts
                if child.dep_ in {"amod", "compound", "nummod", "poss"}
            ]
            compact = normalize_anchor(" ".join(left_mods + [head_tok.text.lower()]))
            if compact:
                candidates.append(compact)

            candidates.append(head_tok.text.lower())

            lemma = (head_tok.lemma_ or head_tok.text).lower()
            if lemma:
                candidates.append(lemma)
                if is_plural_phrase(norm, category, nlp):
                    candidates.append(pluralize_token(lemma))
                else:
                    candidates.append(singularize_token(lemma))

    fallback_tokens = phrase_tokens_fallback(norm)
    if fallback_tokens:
        compact_tokens = []
        for tok in fallback_tokens:
            if tok in SCENE_TOKENS:
                break
            compact_tokens.append(tok)

        if compact_tokens:
            candidates.append(" ".join(compact_tokens))

        candidates.append(fallback_tokens[-1])

        singular = singularize_token(fallback_tokens[-1])
        plural = pluralize_token(fallback_tokens[-1])
        candidates.extend([singular, plural] if is_plural_phrase(norm, category, nlp) else [singular])

    deduped: List[str] = []
    seen: Set[str] = set()
    for cand in candidates:
        cand_norm = normalize_anchor(cand)
        if not cand_norm or cand_norm in seen:
            continue
        seen.add(cand_norm)
        deduped.append(cand_norm)

    return deduped


def query_limit_for_anchor(category: str, is_multi: bool, singular_limit: int, plural_limit: int) -> int:
    if category in SCENE_LEVEL_CATEGORIES:
        return 1
    if is_multi:
        return max(1, plural_limit)
    return max(1, singular_limit)


# =============================================================================
# GROUNDINGDINO QUERY EXECUTION
# =============================================================================
def run_dino_query(
    phrase: str,
    image_tensor,
    w: int,
    h: int,
    dino,
    topk: int,
    nms_thresh: float,
    cache: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    norm = normalize_anchor(phrase)
    if not norm:
        return []

    if norm in cache:
        return cache[norm]

    caption = norm if norm.endswith(".") else norm + "."

    boxes, logits, _ = predict(
        model=dino,
        image=image_tensor,
        caption=caption,
        box_threshold=BOX_THRESHOLD_DEFAULT,
        text_threshold=TEXT_THRESHOLD_DEFAULT,
    )

    if boxes is None or len(boxes) == 0:
        cache[norm] = []
        return []

    scores = logits.detach().cpu().numpy().astype(float)
    boxes_np = boxes.detach().cpu().numpy().astype(float)

    cx, cy, bw, bh = boxes_np[:, 0], boxes_np[:, 1], boxes_np[:, 2], boxes_np[:, 3]
    boxes_px = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    boxes_px *= np.array([w, h, w, h], dtype=float)

    order = scores.argsort()[::-1]
    if topk > 0:
        order = order[:topk]
    boxes_px = boxes_px[order]
    scores = scores[order]

    keep = nms_xyxy(boxes_px, scores, nms_thresh)
    boxes_px = boxes_px[keep]
    scores = scores[keep]

    detections: List[Dict[str, Any]] = []
    for b, s in zip(boxes_px, scores):
        detections.append({
            "box_xyxy": clip_box_xyxy(b.tolist(), w, h),
            "score": float(s),
            "grounding_query": norm,
        })

    cache[norm] = detections
    return detections


# =============================================================================
# PRECISION FILTER CONFIG / EVALUATION
# =============================================================================
def precision_cfg(mode: str) -> Dict[str, float]:
    if mode == "recall":
        return {
            "scene_area_sensitive": 0.90,
            "scene_area_people_group": 0.95,
            "scene_area_large": 0.98,
            "scene_area_background": 1.00,
            "tiny_area_general": 0.0003,
            "tiny_area_building": 0.0008,
            "scene_width": 0.98,
            "scene_height": 0.98,
            "min_score": 0.18,
        }
    if mode == "balanced":
        return {
            "scene_area_sensitive": 0.78,
            "scene_area_people_group": 0.90,
            "scene_area_large": 0.94,
            "scene_area_background": 1.00,
            "tiny_area_general": 0.0005,
            "tiny_area_building": 0.0015,
            "scene_width": 0.95,
            "scene_height": 0.95,
            "min_score": 0.22,
        }

    return {
        "scene_area_sensitive": 0.65,
        "scene_area_people_group": 0.85,
        "scene_area_large": 0.92,
        "scene_area_background": 1.00,
        "tiny_area_general": 0.0007,
        "tiny_area_building": 0.0020,
        "scene_width": 0.93,
        "scene_height": 0.93,
        "min_score": 0.25,
    }


def evaluate_detection(
    det: Dict[str, Any],
    category: str,
    is_multi: bool,
    w: int,
    h: int,
    mode: str,
) -> Tuple[List[str], float]:
    cfg = precision_cfg(mode)

    x1, y1, x2, y2 = [float(v) for v in det["box_xyxy"]]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    area_ratio = (bw * bh) / max(1.0, float(w * h))
    width_ratio = bw / max(1.0, float(w))
    height_ratio = bh / max(1.0, float(h))
    score = float(det.get("score", 0.0))

    reasons: List[str] = []
    adjusted = score

    is_scene_box = (
        area_ratio >= cfg["scene_area_sensitive"]
        or (width_ratio >= cfg["scene_width"] and height_ratio >= cfg["scene_height"])
    )

    if score < cfg["min_score"]:
        reasons.append("low_score")
        adjusted -= 0.20

    if category in ENTITY_SCENE_SENSITIVE:
        if area_ratio >= cfg["scene_area_sensitive"] and not (category in PEOPLE_CATEGORIES and is_multi):
            reasons.append("scene_level_box")
            adjusted -= 0.45
        if category in PEOPLE_CATEGORIES and is_multi and area_ratio >= cfg["scene_area_people_group"]:
            reasons.append("scene_level_box")
            adjusted -= 0.35

    elif category in LARGE_OBJECT_CATEGORIES:
        if area_ratio >= cfg["scene_area_large"] and not is_multi:
            reasons.append("scene_level_box")
            adjusted -= 0.25

    elif category in SCENE_LEVEL_CATEGORIES:
        is_scene_box = False

    if category not in SCENE_LEVEL_CATEGORIES:
        tiny_cutoff = cfg["tiny_area_building"] if category in LARGE_OBJECT_CATEGORIES else cfg["tiny_area_general"]
        if area_ratio <= tiny_cutoff:
            reasons.append("tiny_box")
            adjusted -= 0.18

    if is_multi and category in PEOPLE_CATEGORIES and area_ratio < cfg["tiny_area_general"] * 8:
        reasons.append("too_small_for_group")
        adjusted -= 0.20

    if width_ratio >= 0.98 and height_ratio >= 0.98 and category not in SCENE_LEVEL_CATEGORIES:
        reasons.append("full_frame_box")
        adjusted -= 0.40

    if is_scene_box and category not in SCENE_LEVEL_CATEGORIES:
        adjusted -= 0.10

    det["area_ratio"] = round(area_ratio, 6)
    det["width_ratio"] = round(width_ratio, 6)
    det["height_ratio"] = round(height_ratio, 6)
    det["precision_score"] = round(adjusted, 6)
    det["rejection_reasons"] = reasons

    return reasons, adjusted


# =============================================================================
# IMAGE CROP + VERIFIER STACK
# =============================================================================
def crop_from_box(image_source: np.ndarray, box_xyxy: List[float]) -> Optional[np.ndarray]:
    if image_source is None:
        return None

    h, w = image_source.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in box_xyxy]

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return None

    return image_source[y1:y2, x1:x2].copy()


def run_verifier_stack(
    verifier_bundle: Dict[str, Any],
    crop_rgb: np.ndarray,
    anchor_text: str,
    category: str,
) -> Dict[str, Any]:
    verify_with = verifier_bundle.get("mode", "none")
    results: Dict[str, Dict[str, Any]] = {}
    enabled = verifier_bundle.get("enabled", [])

    if "qwen" in enabled and verifier_bundle.get("qwen") is not None:
        results["qwen"] = verify_crop_with_qwen(verifier_bundle["qwen"], crop_rgb, anchor_text, category)

    if "clip" in enabled and verifier_bundle.get("clip") is not None:
        results["clip"] = verify_crop_with_clip(verifier_bundle["clip"], crop_rgb, anchor_text, category)

    if verify_with == "none" or not results:
        return {"match": True, "score": 1.0, "reason": "verification_disabled", "results": results}

    match = all(bool(r.get("match", False)) for r in results.values())
    score = min(float(r.get("score", 0.0)) for r in results.values())
    reason = "; ".join(f"{name}:{res.get('reason', '')}" for name, res in results.items())
    dominant = "; ".join(f"{name}:{res.get('dominant_entity', '')}" for name, res in results.items())

    return {
        "match": match,
        "score": score,
        "reason": reason,
        "dominant_entity": dominant,
        "results": results,
    }


# =============================================================================
# CORE ANCHOR SELECTION LOGIC
# =============================================================================
def select_detections(
    phrase: str,
    category: str,
    is_multi: bool,
    image_source: Optional[np.ndarray],
    image_tensor,
    w: int,
    h: int,
    dino,
    topk: int,
    nms_thresh: float,
    singular_limit: int,
    plural_limit: int,
    cache: Dict[str, List[Dict[str, Any]]],
    nlp,
    precision_mode: str,
    verifier_bundle: Optional[Dict[str, Any]] = None,
    fallback_policy: str = "drop",
) -> Dict[str, Any]:
    candidate_queries = derive_candidate_queries(phrase, category, nlp)
    query_limit = query_limit_for_anchor(category, is_multi, singular_limit, plural_limit)

    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    fallback_pool: List[Dict[str, Any]] = []

    for idx, query in enumerate(candidate_queries):
        raw_dets = run_dino_query(
            phrase=query,
            image_tensor=image_tensor,
            w=w,
            h=h,
            dino=dino,
            topk=topk,
            nms_thresh=nms_thresh,
            cache=cache,
        )

        for det in raw_dets:
            det_copy = dict(det)
            det_copy["query_rank"] = idx
            det_copy["grounding_query"] = query

            reasons, adjusted = evaluate_detection(det_copy, category, is_multi, w, h, precision_mode)
            fallback_pool.append(det_copy)

            if reasons:
                rejected.append({
                    "grounding_query": query,
                    "score": det_copy["score"],
                    "precision_score": det_copy["precision_score"],
                    "box_xyxy": det_copy["box_xyxy"],
                    "rejection_reasons": reasons,
                })
            else:
                det_copy["selection_mode"] = "precision_filtered"
                det_copy["passed_precision_filters"] = True
                det_copy["fallback_reason"] = None
                accepted.append(det_copy)

    selected = accepted
    fallback_used = False

    if not selected and fallback_pool and fallback_policy == "keep_bad":
        fallback_used = True
        best_bad = max(
            fallback_pool,
            key=lambda d: (float(d.get("precision_score", -1e9)), float(d.get("score", -1e9)))
        )
        best_bad = dict(best_bad)
        best_bad["selection_mode"] = "fallback_bad_box"
        best_bad["passed_precision_filters"] = False
        best_bad["fallback_reason"] = "all_candidates_rejected"
        selected = [best_bad]

    if selected:
        boxes_np = np.array([d["box_xyxy"] for d in selected], dtype=float)
        scores_np = np.array([float(d.get("precision_score", d["score"])) for d in selected], dtype=float)

        keep = nms_xyxy(boxes_np, scores_np, nms_thresh)
        selected = [selected[i] for i in keep]

        selected.sort(
            key=lambda d: (float(d.get("precision_score", -1e9)), float(d.get("score", -1e9))),
            reverse=True,
        )

        verify_pool = selected[: max(query_limit * 3, query_limit)]

        if verifier_bundle and verifier_bundle.get("mode") != "none" and category not in SCENE_LEVEL_CATEGORIES:
            verified: List[Dict[str, Any]] = []
            qwen_rejected: List[Dict[str, Any]] = []

            for det in verify_pool:
                crop_rgb = crop_from_box(image_source, det["box_xyxy"])
                if crop_rgb is None:
                    continue

                try:
                    verify_result = run_verifier_stack(verifier_bundle, crop_rgb, phrase, category)
                except Exception as e:
                    verify_result = {
                        "match": False,
                        "score": 0.0,
                        "dominant_entity": "",
                        "reason": f"verify_error:{type(e).__name__}",
                        "results": {},
                    }

                det["qwen_verify_match"] = bool(verify_result["match"])
                det["qwen_verify_score"] = round(float(verify_result["score"]), 6)
                det["qwen_verify_dominant_entity"] = verify_result["dominant_entity"]
                det["qwen_verify_reason"] = verify_result["reason"]
                det["verifier_results"] = verify_result.get("results", {})

                if det["qwen_verify_match"]:
                    verified.append(det)
                else:
                    qwen_rejected.append({
                        "grounding_query": det.get("grounding_query", ""),
                        "score": det.get("score"),
                        "precision_score": det.get("precision_score"),
                        "box_xyxy": det.get("box_xyxy", []),
                        "rejection_reasons": ["verifier_rejected"],
                        "qwen_verify_dominant_entity": det.get("qwen_verify_dominant_entity", ""),
                        "qwen_verify_reason": det.get("qwen_verify_reason", ""),
                        "verifier_results": det.get("verifier_results", {}),
                    })

            rejected.extend(qwen_rejected)

            if verified:
                verified.sort(
                    key=lambda d: (
                        float(d.get("qwen_verify_score", -1e9)),
                        float(d.get("precision_score", -1e9)),
                        float(d.get("score", -1e9)),
                    ),
                    reverse=True,
                )
                selected = verified[:query_limit]

            elif fallback_policy == "keep_bad" and verify_pool:
                fallback_used = True
                selected = [verify_pool[0]]
                selected[0]["selection_mode"] = "fallback_bad_box"
                selected[0]["passed_precision_filters"] = False
                selected[0]["fallback_reason"] = "verifier_rejected_all"

            else:
                selected = []

        else:
            selected = verify_pool[:query_limit]

    for det in selected:
        det.pop("query_rank", None)

    return {
        "status": "ok" if selected else "no_box",
        "detections": selected,
        "candidate_queries": candidate_queries,
        "rejected_candidates": rejected,
        "grounding_query": selected[0]["grounding_query"] if selected else (candidate_queries[0] if candidate_queries else ""),
        "fallback_used": fallback_used,
        "selection_mode": selected[0]["selection_mode"] if selected else "no_box",
    }


def summarize_rejection_reasons(rejected_candidates: List[Dict[str, Any]]) -> str:
    if not rejected_candidates:
        return "no detections survived candidate generation"

    counts: Dict[str, int] = {}
    verifier_examples: List[str] = []

    for item in rejected_candidates:
        for reason in item.get("rejection_reasons", []) or []:
            counts[reason] = counts.get(reason, 0) + 1
            if reason == "verifier_rejected":
                verifier_reason = str(item.get("qwen_verify_reason", "")).strip()
                if verifier_reason and verifier_reason not in verifier_examples:
                    verifier_examples.append(verifier_reason)

    if not counts:
        return "detections existed, but no accepted box remained"

    parts = [f"{reason} x{counts[reason]}" for reason in sorted(counts)]
    summary = ", ".join(parts)

    if verifier_examples:
        summary += f" | verifier: {verifier_examples[0]}"

    return summary


# =============================================================================
# RELATION MATCHING / RESOLUTION HELPERS
# =============================================================================
def relation_match_score(phrase: str, anchor_entry: Dict[str, Any], nlp) -> float:
    phrase_norm = normalize_anchor(phrase)
    anchor_norm = normalize_anchor(anchor_entry.get("anchor_norm", ""))

    if not phrase_norm or not anchor_norm:
        return -1.0

    if phrase_norm == anchor_norm:
        return 10.0

    phrase_head, phrase_lemma = head_token_info(phrase_norm, nlp)
    anchor_head, anchor_lemma = head_token_info(anchor_norm, nlp)

    score = 0.0
    if phrase_head and phrase_head == anchor_head:
        score += 5.0
    if phrase_lemma and phrase_lemma == anchor_lemma:
        score += 3.0

    phrase_toks = tokens_for_match(phrase_norm, nlp)
    anchor_toks = tokens_for_match(anchor_norm, nlp)
    overlap = phrase_toks & anchor_toks

    score += float(len(overlap))
    if phrase_toks and anchor_toks:
        score += len(overlap) / max(len(anchor_toks), 1)

    return score


def resolve_relation_entity(
    phrase: str,
    object_entries: List[Dict[str, Any]],
    nlp,
) -> Optional[Dict[str, Any]]:
    best_entry = None
    best_score = 0.0

    for entry in object_entries:
        if entry.get("status") != "ok" or not entry.get("detections"):
            continue
        score = relation_match_score(phrase, entry, nlp)
        if score > best_score:
            best_score = score
            best_entry = entry

    if best_entry is None or best_score < 2.5:
        return None

    return best_entry


def ground_relation_entity(
    phrase: str,
    category_hint: str,
    object_entries: List[Dict[str, Any]],
    image_source: Optional[np.ndarray],
    image_tensor,
    w: int,
    h: int,
    dino,
    topk: int,
    nms_thresh: float,
    singular_limit: int,
    plural_limit: int,
    cache: Dict[str, List[Dict[str, Any]]],
    nlp,
    precision_mode: str,
    verifier_bundle: Optional[Dict[str, Any]] = None,
    fallback_policy: str = "keep_bad",
) -> Dict[str, Any]:
    if is_non_groundable(phrase):
        return {
            "status": "non_groundable",
            "detections": [],
            "candidate_queries": [],
            "rejected_candidates": [],
            "grounding_query": "",
            "fallback_used": False,
            "selection_mode": "non_groundable",
            "resolved_from_anchor_norm": None,
        }

    max_boxes = query_limit_for_anchor(
        category_hint or "object",
        is_plural_phrase(phrase, category_hint or "object", nlp),
        singular_limit,
        plural_limit,
    )

    dets = run_dino_query(
        phrase=phrase,
        image_tensor=image_tensor,
        w=w,
        h=h,
        dino=dino,
        topk=topk,
        nms_thresh=nms_thresh,
        cache=cache,
    )[:max_boxes]

    for det in dets:
        det["selection_mode"] = "raw_relation_grounding"
        det["passed_precision_filters"] = None
        det["fallback_reason"] = None

    return {
        "status": "ok" if dets else "no_box",
        "detections": dets,
        "candidate_queries": [normalize_anchor(phrase)] if phrase else [],
        "rejected_candidates": [],
        "grounding_query": normalize_anchor(phrase),
        "fallback_used": False,
        "selection_mode": "raw_relation_grounding" if dets else "no_box",
        "resolved_from_anchor_norm": None,
    }


# =============================================================================
# DEBUG VISUALIZATION
# =============================================================================
def draw_debug_boxes(
    image_bgr: np.ndarray,
    anchor_entries: List[Dict[str, Any]],
    relation_entries: List[Dict[str, Any]],
    out_path: Path,
) -> None:
    if image_bgr.shape[2] == 3:
        vis = cv2.cvtColor(image_bgr, cv2.COLOR_RGB2BGR)
    else:
        vis = image_bgr.copy()

    for a in anchor_entries:
        anchor_norm = a["anchor_norm"]
        for det in a.get("detections", []):
            x1, y1, x2, y2 = [int(v) for v in det["box_xyxy"]]
            color = (0, 180, 255) if det.get("selection_mode") == "fallback_bad_box" else (0, 255, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            label = f"{anchor_norm} {det['score']:.2f}"
            if det.get("selection_mode") == "fallback_bad_box":
                label += " FB"

            cv2.putText(
                vis,
                label[:60],
                (x1, max(12, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

    for rel in relation_entries:
        for det in rel.get("subject_detections", []):
            x1, y1, x2, y2 = [int(v) for v in det["box_xyxy"]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 100, 0), 2)
            label = f"S:{rel['subject'][:30]} {det['score']:.2f}"
            cv2.putText(
                vis, label[:60], (x1, max(12, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 0), 1, cv2.LINE_AA
            )

    for rel in relation_entries:
        for det in rel.get("object_detections", []):
            x1, y1, x2, y2 = [int(v) for v in det["box_xyxy"]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 60, 220), 2)
            label = f"O:{rel['object'][:30]} {det['score']:.2f}"
            cv2.putText(
                vis, label[:60], (x1, min(vis.shape[0] - 4, y2 + 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 60, 220), 1, cv2.LINE_AA
            )

    cv2.imwrite(str(out_path), vis)


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Ground Qwen anchors using GroundingDINO.")

    parser.add_argument(
        "--anchors-json",
        type=str,
        default="",
        help=(
            "Path to Qwen3 anchor JSON, shard directory, or glob pattern. "
            "Defaults to timpact/data/anchors_qwen3.json."
        ),
    )

    parser.add_argument(
        "--max-boxes-per-anchor",
        type=int,
        default=1,
        help="Keep at most N boxes for singular anchors after precision filtering and NMS.",
    )

    parser.add_argument(
        "--max-boxes-plural",
        type=int,
        default=4,
        help="Keep at most N boxes for plural/group anchors after precision filtering and NMS.",
    )

    parser.add_argument(
        "--topk-dino",
        type=int,
        default=20,
        help="Consider top-K raw detections from DINO before NMS."
    )

    parser.add_argument(
        "--nms",
        type=float,
        default=0.7,
        help="NMS IoU threshold for removing near-duplicate boxes."
    )

    parser.add_argument(
        "--precision-mode",
        choices=["high", "balanced", "recall"],
        default="high",
        help="Controls how aggressively scene-level or implausible boxes are filtered.",
    )

    parser.add_argument(
        "--verify-with",
        choices=["none", "qwen", "clip", "qwen_clip"],
        default="qwen_clip",
        help="Verifier stack for object-anchor crops.",
    )

    parser.add_argument(
        "--fallback-policy",
        choices=["drop", "keep_bad"],
        default="drop",
        help="Whether to drop fully rejected anchors or keep one marked bad box.",
    )

    parser.add_argument(
        "--debug-draw",
        action="store_true",
        help="If set, save debug_boxes.jpg with all kept boxes drawn."
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="If >0, process only the first N loaded items after filtering."
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip ids that already have timpact/data/grounding/<id>/boxes.json.",
    )

    args = parser.parse_args()

    device = choose_device()
    nlp = load_nlp()

    verifier_bundle: Dict[str, Any] = {"mode": args.verify_with, "enabled": []}

    if args.verify_with in {"qwen", "qwen_clip"}:
        verifier_bundle["qwen"] = load_qwen_verifier(device)
        if verifier_bundle["qwen"] is not None:
            verifier_bundle["enabled"].append("qwen")
    else:
        verifier_bundle["qwen"] = None

    if args.verify_with in {"clip", "qwen_clip"}:
        verifier_bundle["clip"] = load_clip_verifier(device)
        if verifier_bundle["clip"] is not None:
            verifier_bundle["enabled"].append("clip")
    else:
        verifier_bundle["clip"] = None

    if args.verify_with != "none" and not verifier_bundle["enabled"]:
        print("[WARN] No verifier backend could be loaded. Verification disabled.")
        verifier_bundle["mode"] = "none"

    print(f"[DINO] Using device: {device}")
    print(f"[DINO] Precision mode: {args.precision_mode}")
    print(f"[DINO] Verify with: {verifier_bundle['mode']}")
    print(f"[DINO] Fallback policy: {args.fallback_policy}")

    anchor_json_path = choose_anchor_json_path(args.anchors_json)
    items = load_anchor_records(anchor_json_path)

    filtered_items: List[Dict[str, Any]] = []
    skipped_bad_status = 0
    skipped_missing_outputs = 0
    skipped_missing_anchors = 0

    for item in items:
        item_status = str(item.get("status", "")).strip().lower()
        if item_status != "ok":
            skipped_bad_status += 1
            continue

        outputs = item.get("outputs", {})
        if not isinstance(outputs, dict):
            skipped_missing_outputs += 1
            continue

        anchors_payload = outputs.get("anchors", {})
        if not isinstance(anchors_payload, dict):
            skipped_missing_anchors += 1
            continue

        anchor_list = anchors_payload.get("anchors", [])
        if not isinstance(anchor_list, list) or not anchor_list:
            skipped_missing_anchors += 1
            continue

        filtered_items.append(item)

    if args.max_images and args.max_images > 0:
        filtered_items = filtered_items[: args.max_images]

    print(f"[DINO] Loaded total records: {len(items)}")
    print(
        f"[DINO] Eligible records after filtering: {len(filtered_items)} "
        f"(skipped status!=ok: {skipped_bad_status}, "
        f"missing outputs: {skipped_missing_outputs}, "
        f"missing anchors: {skipped_missing_anchors})"
    )

    print("[DINO] Downloading/locating GroundingDINO files via HF cache...")
    cfg_path, weights_path = download_dino_files()
    print(f"[DINO] cfg: {cfg_path}")
    print(f"[DINO] wts: {weights_path}")

    print("[DINO] Loading model...")
    dino = load_model(cfg_path, weights_path).to(device)
    dino.eval()

    ensure_dir(OUT_BASE)

    processed = 0
    skipped_existing = 0

    for idx, item in enumerate(filtered_items, start=1):
        sid = str(item.get("id", "")).zfill(6)
        img_path = Path(item.get("resolved_image_path", ""))

        out_dir = OUT_BASE / sid
        ensure_dir(out_dir)
        out_json = out_dir / "boxes.json"

        if args.skip_existing and out_json.exists():
            skipped_existing += 1
            print(f"\n[SKIP {idx}/{len(filtered_items)}] id={sid} already exists: {out_json}")
            continue

        print(f"\n[PROCESS {idx}/{len(filtered_items)}] id={sid} image={img_path}")

        outputs = item.get("outputs", None)
        if not isinstance(outputs, dict):
            print(f"[WARN] outputs missing/None for id={sid}. Skipping.")
            continue

        anchors_payload = outputs.get("anchors", {}) if isinstance(outputs.get("anchors"), dict) else {}
        anchor_list = anchors_payload.get("anchors", []) if isinstance(anchors_payload.get("anchors"), list) else []

        if not anchor_list:
            print(f"[WARN] No Qwen3 anchors for id={sid}. Skipping.")
            continue

        if not img_path.exists():
            print(f"[WARN] Missing image file: {img_path}. Skipping.")
            continue

        image_source, image_tensor = load_image(str(img_path))
        h, w = image_source.shape[:2]

        dino_cache: Dict[str, List[Dict[str, Any]]] = {}

        anchor_entries: List[Dict[str, Any]] = []
        anchor_entry_by_id: Dict[str, Dict[str, Any]] = {}

        for a in anchor_list:
            if not isinstance(a, dict):
                continue

            anchor_text = str(a.get("span", "")).strip()
            if not anchor_text:
                continue

            anchor_norm = normalize_anchor(anchor_text)
            if not anchor_norm:
                continue

            anchor_id = str(a.get("anchor_id", "")).strip() or anchor_norm

            semantic_tags = a.get("semantic_tags", []) if isinstance(a.get("semantic_tags"), list) else []
            attributes_list = [str(x).strip() for x in (a.get("attributes") or []) if str(x).strip()]
            localization = a.get("localization", {}) if isinstance(a.get("localization"), dict) else {}
            functional_role = str(a.get("functional_role", "")).strip()
            visibility = str(a.get("visibility", "")).strip()
            salience = str(a.get("salience", "")).strip()
            confidence = str(a.get("confidence", "")).strip()

            category = semantic_tags_to_category(semantic_tags, anchor_text, nlp)
            is_multi = is_plural_phrase(anchor_text, category, nlp)

            result = select_detections(
                phrase=anchor_text,
                category=category,
                is_multi=is_multi,
                image_source=image_source,
                image_tensor=image_tensor,
                w=w,
                h=h,
                dino=dino,
                topk=args.topk_dino,
                nms_thresh=float(args.nms),
                singular_limit=max(1, int(args.max_boxes_per_anchor)),
                plural_limit=max(1, int(args.max_boxes_plural)),
                cache=dino_cache,
                nlp=nlp,
                precision_mode=args.precision_mode,
                verifier_bundle=verifier_bundle,
                fallback_policy=args.fallback_policy,
            )

            entry = {
                "anchor_id": anchor_id,
                "anchor_text": anchor_text,
                "anchor_norm": normalize_anchor(anchor_text),
                "category": category,
                "semantic_tags": semantic_tags,
                "functional_role": functional_role,
                "localization": localization,
                "attributes": "; ".join(attributes_list),
                "attributes_list": attributes_list,
                "visibility": visibility,
                "salience": salience,
                "confidence": confidence,
                "is_multi_instance": is_multi,
                "status": result["status"],
                "grounding_query": result["grounding_query"],
                "candidate_queries": result["candidate_queries"],
                "rejected_candidates": result["rejected_candidates"],
                "fallback_used": result["fallback_used"],
                "selection_mode": result["selection_mode"],
                "detections": result["detections"],
            }

            anchor_entries.append(entry)
            anchor_entry_by_id[anchor_id] = entry

            if entry["status"] != "ok":
                rejection_summary = summarize_rejection_reasons(entry["rejected_candidates"])
                candidate_preview = ", ".join(entry["candidate_queries"][:4]) if entry["candidate_queries"] else "none"
                print(
                    f"[REJECTED] anchor='{anchor_text}' "
                    f"queries=[{candidate_preview}] reasons={rejection_summary}"
                )

        rel_entries: List[Dict[str, Any]] = []
        rel_seen: Set[Tuple[str, str, str]] = set()
        rel_index = 0

        for anchor in anchor_list:
            if not isinstance(anchor, dict):
                continue

            source_id = str(anchor.get("anchor_id", "")).strip()
            source_entry = anchor_entry_by_id.get(source_id)
            if source_entry is None:
                continue

            for rel in anchor.get("relations", []) or []:
                if not isinstance(rel, dict):
                    continue

                target_id = str(rel.get("target_anchor_id", "")).strip()
                predicate = str(rel.get("type", "")).strip()

                if not target_id or not predicate:
                    continue

                target_entry = anchor_entry_by_id.get(target_id)
                if target_entry is None:
                    continue

                key = (source_id, predicate, target_id)
                if key in rel_seen:
                    continue
                rel_seen.add(key)

                rel_entries.append({
                    "relation": f"{source_entry['anchor_text']} {predicate} {target_entry['anchor_text']}",
                    "subject_anchor_id": source_id,
                    "object_anchor_id": target_id,
                    "subject": source_entry["anchor_text"],
                    "predicate": predicate,
                    "object": target_entry["anchor_text"],
                    "type": relation_type_from_qwen3(predicate),
                    "rel_src_index": rel_index,

                    "subject_status": source_entry["status"],
                    "object_status": target_entry["status"],

                    "subject_grounding_query": source_entry["grounding_query"],
                    "object_grounding_query": target_entry["grounding_query"],
                    "subject_candidate_queries": source_entry["candidate_queries"],
                    "object_candidate_queries": target_entry["candidate_queries"],
                    "subject_rejected_candidates": source_entry["rejected_candidates"],
                    "object_rejected_candidates": target_entry["rejected_candidates"],
                    "subject_resolved_from_anchor_norm": source_entry["anchor_norm"],
                    "object_resolved_from_anchor_norm": target_entry["anchor_norm"],

                    "subject_detections": [dict(d) for d in source_entry["detections"]],
                    "object_detections": [dict(d) for d in target_entry["detections"]],
                })
                rel_index += 1

        payload = {
            "id": sid,
            "row_index": item.get("row_index"),
            "image_path": str(img_path),
            "headline": str((item.get("csv_fields") or {}).get("title", "")).strip(),
            "summary": str((item.get("csv_fields") or {}).get("summary", "")).strip(),
            "published": str((item.get("csv_fields") or {}).get("published", "")).strip(),
            "url": str((item.get("csv_fields") or {}).get("url", "")).strip(),
            "grounded_context_caption": str(anchors_payload.get("grounded_context_caption", "")).strip(),
            "anchors": anchor_entries,
            "relation_anchors": rel_entries,
        }

        with out_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print(f"[OK] Wrote {out_json}")

        if args.debug_draw:
            debug_path = out_dir / "debug_boxes.jpg"
            draw_debug_boxes(image_source, anchor_entries, rel_entries, debug_path)
            print(f"[DBG] Wrote {debug_path}")

        a_total = len(anchor_entries)
        a_ok = sum(1 for a in anchor_entries if a.get("status") == "ok" and a.get("detections"))
        a_fb = sum(1 for a in anchor_entries if a.get("fallback_used"))

        r_total = len(rel_entries)
        r_both_ok = sum(
            1 for r in rel_entries
            if r.get("subject_status") == "ok" and r.get("object_status") == "ok"
        )
        r_non_gr = sum(
            1 for r in rel_entries
            if "non_groundable" in (r.get("subject_status"), r.get("object_status"))
        )

        print(
            f"[SUMMARY] object_anchors={a_total} grounded={a_ok} fallback={a_fb} | "
            f"relation_anchors={r_total} both_ok={r_both_ok} has_non_groundable={r_non_gr}"
        )

        processed += 1

    print(f"\n[DONE] Grounding complete. processed={processed} skipped_existing={skipped_existing}")


if __name__ == "__main__":
    main()