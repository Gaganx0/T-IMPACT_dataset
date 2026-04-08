#!/usr/bin/env python

"""
Step 3b: Segment grounded boxes using the original SAM.

Compatible inputs:
  - single grounding JSON file, e.g.:
      timpact/data/grounding/000001/boxes.json
  - grounding root directory, e.g.:
      timpact/data/grounding/
  - glob pattern, e.g.:
      "timpact/data/grounding/*/boxes.json"

Output:
  - timpact/data/segments/<id>/masks/*.png
  - timpact/data/segments/<id>/boxes.json
  - optional: timpact/data/segments/<id>/debug_masks.jpg  (if --debug-draw)

Examples:
    python timpact/scripts/segment_from_boxes_sam.py \
        --sam-model vit_b \
        --grounding-json timpact/data/grounding

    python timpact/scripts/segment_from_boxes_sam.py \
        --sam-model vit_l \
        --grounding-json timpact/data/grounding/000001/boxes.json

    python timpact/scripts/segment_from_boxes_sam.py \
        --sam-model vit_b \
        --grounding-json "/home/Student/s4826850/timpact/data/grounding/*/boxes.json"

Batch / shard examples:
    python timpact/scripts/segment_from_boxes_sam.py \
        --sam-model vit_b \
        --grounding-json timpact/data/grounding \
        --num-shards 20 \
        --shard-index 0 \
        --skip-existing

This script:
  - reads GroundingDINO outputs from boxes.json
  - segments object-anchor detections and relation subject/object detections
  - writes per-mask PNGs plus updated segment-aware boxes.json
  - supports shard-safe processing for Slurm arrays on Rangpur
"""

import argparse
import glob
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download

try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError as e:
    raise ImportError(
        "segment_anything is not installed. Install the official SAM package:\n"
        "  pip install git+https://github.com/facebookresearch/segment-anything.git"
    ) from e


# =============================================================================
# PATHS
# =============================================================================
ROOT = Path(__file__).resolve().parents[1]

IN_BASE = ROOT / "data" / "grounding"
OUT_BASE = ROOT / "data" / "segments"


# =============================================================================
# SAM CHECKPOINTS
# =============================================================================
SAM_REPO = "ybelkada/segment-anything"

SAM_CKPT_BY_MODEL = {
    "vit_b": "checkpoints/sam_vit_b_01ec64.pth",
    "vit_l": "checkpoints/sam_vit_l_0b3195.pth",
}


# =============================================================================
# GENERAL HELPERS
# =============================================================================
def choose_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def slugify(text: str, maxlen: int = 48) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(text).lower().strip())
    s = s.strip("_")
    return s[:maxlen] if s else "item"


def safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Could not parse JSON: {path} ({e})")
        return None


def resolve_repo_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (ROOT / p).resolve()


def download_sam_checkpoint(model_type: str) -> str:
    if model_type not in SAM_CKPT_BY_MODEL:
        raise ValueError(f"Unknown SAM model type: {model_type}")
    return hf_hub_download(repo_id=SAM_REPO, filename=SAM_CKPT_BY_MODEL[model_type])


# =============================================================================
# INPUT COLLECTION / SHARDING
# =============================================================================
def choose_grounding_json_path(explicit_path: str = "") -> Path:
    if explicit_path:
        return Path(explicit_path)
    return IN_BASE


def collect_grounding_input_files(grounding_json_path: Path) -> List[Path]:
    """
    Accept:
      - single boxes.json file
      - directory containing grounding/<id>/boxes.json
      - glob pattern such as grounding/*/boxes.json
    """
    grounding_json_str = str(grounding_json_path)

    if any(ch in grounding_json_str for ch in ["*", "?", "["]):
        files = sorted(Path(p) for p in glob.glob(grounding_json_str))
        return [p for p in files if p.is_file() and p.name == "boxes.json"]

    if grounding_json_path.is_file():
        if grounding_json_path.name == "boxes.json":
            return [grounding_json_path]
        return []

    if grounding_json_path.is_dir():
        direct = grounding_json_path / "boxes.json"
        if direct.is_file():
            return [direct]

        files = sorted(
            p for p in grounding_json_path.glob("*/boxes.json")
            if p.is_file()
        )
        return files

    return []


def apply_sharding(
    files: List[Path],
    num_shards: int,
    shard_index: int,
) -> List[Path]:
    if num_shards <= 1:
        return files
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"Invalid shard_index={shard_index} for num_shards={num_shards}")
    return [fp for i, fp in enumerate(files) if (i % num_shards) == shard_index]


# =============================================================================
# MASK IO
# =============================================================================
def save_mask_png(mask_bool: np.ndarray, out_path: Path) -> None:
    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    cv2.imwrite(str(out_path), mask_u8)


def save_union_mask_png(mask_paths: List[str], out_path: Path) -> bool:
    union = None
    for mp_str in mask_paths:
        if not mp_str:
            continue

        mp = Path(mp_str)
        if not mp.is_absolute():
            mp = ROOT / mp

        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue

        union = m if union is None else np.maximum(union, m)

    if union is None:
        return False

    cv2.imwrite(str(out_path), union)
    return True


# =============================================================================
# SAM CORE
# =============================================================================
def sam_segment(
    predictor: "SamPredictor",
    box: List[float],
    sid: str,
    label: str,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    input_box = np.array(box, dtype=np.float32)

    try:
        masks, scores_sam, _ = predictor.predict(
            box=input_box,
            multimask_output=True,
        )

        if masks is None or len(masks) == 0:
            raise RuntimeError("SAM returned no masks")

        best_idx = int(np.argmax(scores_sam))
        return masks[best_idx], None

    except Exception as e:
        reason = f"{type(e).__name__}: {e}"
        print(f"[WARN] SAM failed for id={sid} label={label}: {reason}")
        return None, reason


def summarize_sam_failure(failure_reasons: List[str], failed_boxes: List[List[float]]) -> str:
    if not failure_reasons:
        return "no SAM mask produced"

    unique_reasons: List[str] = []
    for reason in failure_reasons:
        if reason and reason not in unique_reasons:
            unique_reasons.append(reason)

    box_preview = ", ".join(
        str([round(float(v), 1) for v in box]) for box in failed_boxes[:2]
    ) if failed_boxes else "none"

    return f"reasons={'; '.join(unique_reasons[:3])} boxes={box_preview}"


# =============================================================================
# DEBUG VIS
# =============================================================================
def overlay_masks_for_debug(
    image_bgr: np.ndarray,
    anchor_entries: List[Dict[str, Any]],
    relation_entries: List[Dict[str, Any]],
) -> np.ndarray:
    vis = image_bgr.copy()
    overlay = image_bgr.copy()

    def load_mask(mask_path: str) -> Optional[np.ndarray]:
        if not mask_path:
            return None

        mp = Path(mask_path)
        if not mp.is_absolute():
            mp = ROOT / mp

        if not mp.exists():
            return None

        return cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)

    # Object masks: red tint
    for a in anchor_entries:
        for det in a.get("detections", []):
            mask = load_mask(det.get("mask_path", ""))
            if mask is None:
                continue
            ys, xs = np.where(mask > 0)
            overlay[ys, xs] = (overlay[ys, xs] * 0.5 + np.array([0, 0, 220]) * 0.5).astype(np.uint8)

    # Relation subject masks: blue tint
    for rel in relation_entries:
        for det in rel.get("subject_detections", []):
            mask = load_mask(det.get("mask_path", ""))
            if mask is None:
                continue
            ys, xs = np.where(mask > 0)
            overlay[ys, xs] = (overlay[ys, xs] * 0.5 + np.array([220, 60, 0]) * 0.5).astype(np.uint8)

    # Relation object masks: yellow tint
    for rel in relation_entries:
        for det in rel.get("object_detections", []):
            mask = load_mask(det.get("mask_path", ""))
            if mask is None:
                continue
            ys, xs = np.where(mask > 0)
            overlay[ys, xs] = (overlay[ys, xs] * 0.5 + np.array([0, 220, 220]) * 0.5).astype(np.uint8)

    vis = cv2.addWeighted(overlay, 0.7, vis, 0.3, 0)

    for a in anchor_entries:
        anchor_norm = a.get("anchor_norm", "")
        for det in a.get("detections", []):
            box = det.get("box_xyxy")
            if not box:
                continue
            x1, y1, x2, y2 = [int(float(v)) for v in box]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                vis,
                anchor_norm[:60],
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

    for rel in relation_entries:
        for det in rel.get("subject_detections", []):
            box = det.get("box_xyxy")
            if not box:
                continue
            x1, y1, x2, y2 = [int(float(v)) for v in box]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 100, 0), 2)
            cv2.putText(
                vis,
                f"S:{rel.get('subject', '')[:40]}",
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 100, 0),
                1,
                cv2.LINE_AA,
            )

    for rel in relation_entries:
        for det in rel.get("object_detections", []):
            box = det.get("box_xyxy")
            if not box:
                continue
            x1, y1, x2, y2 = [int(float(v)) for v in box]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 60, 220), 2)
            cv2.putText(
                vis,
                f"O:{rel.get('object', '')[:40]}",
                (x1, min(vis.shape[0] - 4, y2 + 13)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 60, 220),
                1,
                cv2.LINE_AA,
            )

    return vis


# =============================================================================
# DETECTION SEGMENT HELPERS
# =============================================================================
def segment_detection_list(
    predictor: "SamPredictor",
    sid: str,
    label_prefix: str,
    detections: List[Dict[str, Any]],
    masks_dir: Path,
    fallback_grounding_query: str = "",
    keep_extra_fields: bool = True,
) -> Tuple[List[Dict[str, Any]], List[str], List[List[float]]]:
    det_out_list: List[Dict[str, Any]] = []
    failure_reasons: List[str] = []
    failed_boxes: List[List[float]] = []

    for k, det in enumerate(detections):
        box = det.get("box_xyxy")
        score = det.get("score", None)

        if not box or len(box) != 4:
            failure_reasons.append("invalid_box")
            continue

        x1, y1, x2, y2 = [float(v) for v in box]

        best_mask, sam_failure_reason = sam_segment(
            predictor=predictor,
            box=[x1, y1, x2, y2],
            sid=sid,
            label=f"{label_prefix}_{k}",
        )

        if best_mask is None:
            failure_reasons.append(sam_failure_reason or "no_mask")
            failed_boxes.append([x1, y1, x2, y2])
            continue

        safe_label = slugify(label_prefix, maxlen=48)
        mask_path = masks_dir / f"{safe_label}_{k}.png"
        save_mask_png(best_mask, mask_path)

        det_out = {
            "box_xyxy": [x1, y1, x2, y2],
            "score": float(score) if score is not None else None,
            "grounding_query": det.get("grounding_query", fallback_grounding_query),
            "mask_path": str(mask_path.relative_to(ROOT)),
        }

        if keep_extra_fields:
            for key in [
                "selection_mode",
                "passed_precision_filters",
                "fallback_reason",
                "precision_score",
                "area_ratio",
                "width_ratio",
                "height_ratio",
                "qwen_verify_match",
                "qwen_verify_score",
                "qwen_verify_dominant_entity",
                "qwen_verify_reason",
                "verifier_results",
                "resolved_via_object_anchor",
            ]:
                if key in det:
                    det_out[key] = det.get(key)

        det_out_list.append(det_out)

    return det_out_list, failure_reasons, failed_boxes


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Segment GroundingDINO boxes using original SAM.")

    parser.add_argument(
        "--sam-model",
        choices=["vit_b", "vit_l"],
        required=True,
        help="Which SAM backbone checkpoint to use.",
    )

    parser.add_argument(
        "--grounding-json",
        type=str,
        default="",
        help=(
            "Path to grounding boxes.json, grounding directory, or glob pattern. "
            "Defaults to timpact/data/grounding/"
        ),
    )

    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of shards/jobs splitting the input list.",
    )

    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Which shard index this job should process (0-based).",
    )

    parser.add_argument(
        "--debug-draw",
        action="store_true",
        help="If set, save debug_masks.jpg with masks overlaid and boxes drawn.",
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip ids that already have timpact/data/segments/<id>/boxes.json.",
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="If >0, process only the first N selected grounding items after sharding/filtering.",
    )

    args = parser.parse_args()

    device = choose_device()
    print(f"[SAM] Using device: {device}")
    print(f"[SAM] Model type: {args.sam_model}")
    print(f"[SAM] num_shards={args.num_shards} shard_index={args.shard_index}")

    ensure_dir(OUT_BASE)

    grounding_json_path = choose_grounding_json_path(args.grounding_json)
    box_files = collect_grounding_input_files(grounding_json_path)

    if not box_files:
        raise FileNotFoundError(f"No valid grounding boxes.json files found for: {grounding_json_path}")

    box_files = apply_sharding(box_files, args.num_shards, args.shard_index)

    if args.max_images and args.max_images > 0:
        box_files = box_files[: args.max_images]

    print(f"[SAM] Found {len(box_files)} grounding output(s) for this shard.")

    print("[SAM] Downloading/locating SAM checkpoint via HF cache...")
    sam_ckpt = download_sam_checkpoint(args.sam_model)
    print(f"[SAM] ckpt: {sam_ckpt}")

    print("[SAM] Loading SAM...")
    sam = sam_model_registry[args.sam_model](checkpoint=sam_ckpt)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    processed = 0
    skipped_existing = 0

    for idx, boxes_json in enumerate(box_files, start=1):
        payload = safe_load_json(boxes_json)
        if payload is None:
            continue

        sid = str(payload.get("id", "")).strip() or boxes_json.parent.name
        sid = sid.zfill(6)

        out_dir = OUT_BASE / sid
        out_json = out_dir / "boxes.json"

        if args.skip_existing and out_json.exists():
            skipped_existing += 1
            print(f"\n[SKIP {idx}/{len(box_files)}] id={sid} already exists: {out_json}")
            continue

        print(f"\n[PROCESS {idx}/{len(box_files)}] id={sid} boxes={boxes_json}")

        image_path_raw = str(payload.get("image_path", "")).strip()
        image_path = resolve_repo_path(image_path_raw)

        if not image_path.exists():
            print(f"[WARN] Missing image file: {image_path}. Skipping id={sid}.")
            continue

        anchors = payload.get("anchors", []) or []
        relation_anchors = payload.get("relation_anchors", []) or []

        if not anchors and not relation_anchors:
            print(f"[WARN] No anchors in grounding boxes.json for id={sid}. Skipping.")
            continue

        img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"[WARN] cv2 failed to read: {image_path}. Skipping id={sid}.")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        predictor.set_image(img_rgb)

        masks_dir = out_dir / "masks"
        ensure_dir(masks_dir)

        # ---------------------------------------------------------------------
        # OBJECT ANCHORS
        # ---------------------------------------------------------------------
        out_anchors: List[Dict[str, Any]] = []

        for a in anchors:
            anchor_text = a.get("anchor_text", "")
            anchor_norm = a.get("anchor_norm", "")
            status = a.get("status", "no_box")
            detections = a.get("detections", []) or []

            out_entry = {
                "anchor_id": a.get("anchor_id", ""),
                "anchor_text": anchor_text,
                "anchor_norm": anchor_norm,
                "category": a.get("category", "object"),
                "semantic_tags": a.get("semantic_tags", []),
                "functional_role": a.get("functional_role", ""),
                "localization": a.get("localization", {}),
                "attributes": a.get("attributes", ""),
                "attributes_list": a.get("attributes_list", []),
                "visibility": a.get("visibility", ""),
                "salience": a.get("salience", ""),
                "confidence": a.get("confidence", ""),
                "is_multi_instance": a.get("is_multi_instance", False),
                "status": status,
                "grounding_query": a.get("grounding_query", ""),
                "candidate_queries": a.get("candidate_queries", []),
                "rejected_candidates": a.get("rejected_candidates", []),
                "fallback_used": a.get("fallback_used", False),
                "selection_mode": a.get("selection_mode", status),
                "union_mask_path": None,
                "detections": [],
            }

            if status != "ok" or not detections:
                out_anchors.append(out_entry)
                continue

            label_prefix = anchor_norm or anchor_text or f"anchor_{a.get('anchor_id', 'x')}"
            det_out_list, failure_reasons, failed_boxes = segment_detection_list(
                predictor=predictor,
                sid=sid,
                label_prefix=label_prefix,
                detections=detections,
                masks_dir=masks_dir,
                fallback_grounding_query=a.get("grounding_query", ""),
                keep_extra_fields=True,
            )

            out_entry["detections"] = det_out_list

            if out_entry["status"] == "ok" and len(det_out_list) == 0:
                out_entry["status"] = "sam_failed"

            elif det_out_list:
                union_path = masks_dir / f"{slugify(label_prefix)}_union.png"
                if save_union_mask_png([d["mask_path"] for d in det_out_list], union_path):
                    out_entry["union_mask_path"] = str(union_path.relative_to(ROOT))

            if out_entry["status"] == "sam_failed":
                failure_summary = summarize_sam_failure(failure_reasons, failed_boxes)
                print(f"[SAM-FAILED] anchor='{anchor_text}' {failure_summary}")

            out_anchors.append(out_entry)

        # ---------------------------------------------------------------------
        # RELATION ANCHORS
        # ---------------------------------------------------------------------
        out_rels: List[Dict[str, Any]] = []

        for r_i, rel in enumerate(relation_anchors):
            subj_status = rel.get("subject_status", "no_box")
            obj_status = rel.get("object_status", "no_box")

            subj_dets = rel.get("subject_detections", []) or []
            obj_dets = rel.get("object_detections", []) or []

            subj_det_out: List[Dict[str, Any]] = []
            obj_det_out: List[Dict[str, Any]] = []

            if subj_status == "ok" and subj_dets:
                subj_det_out, subj_failure_reasons, subj_failed_boxes = segment_detection_list(
                    predictor=predictor,
                    sid=sid,
                    label_prefix=f"rel{r_i:03d}_subj",
                    detections=subj_dets,
                    masks_dir=masks_dir,
                    fallback_grounding_query=rel.get("subject_grounding_query", ""),
                    keep_extra_fields=True,
                )

                if subj_status == "ok" and len(subj_det_out) == 0:
                    subj_status = "sam_failed"

                if subj_status == "sam_failed":
                    failure_summary = summarize_sam_failure(subj_failure_reasons, subj_failed_boxes)
                    print(f"[SAM-FAILED] relation_subject='{rel.get('subject', '')}' {failure_summary}")

            if obj_status == "ok" and obj_dets:
                obj_det_out, obj_failure_reasons, obj_failed_boxes = segment_detection_list(
                    predictor=predictor,
                    sid=sid,
                    label_prefix=f"rel{r_i:03d}_obj",
                    detections=obj_dets,
                    masks_dir=masks_dir,
                    fallback_grounding_query=rel.get("object_grounding_query", ""),
                    keep_extra_fields=True,
                )

                if obj_status == "ok" and len(obj_det_out) == 0:
                    obj_status = "sam_failed"

                if obj_status == "sam_failed":
                    failure_summary = summarize_sam_failure(obj_failure_reasons, obj_failed_boxes)
                    print(f"[SAM-FAILED] relation_object='{rel.get('object', '')}' {failure_summary}")

            union_mask_path = None
            all_mask_paths = (
                [d.get("mask_path", "") for d in subj_det_out] +
                [d.get("mask_path", "") for d in obj_det_out]
            )

            if any(all_mask_paths):
                union_path = masks_dir / f"rel{r_i:03d}_union.png"
                if save_union_mask_png(all_mask_paths, union_path):
                    union_mask_path = str(union_path.relative_to(ROOT))

            out_rels.append({
                "relation": rel.get("relation", ""),
                "subject_anchor_id": rel.get("subject_anchor_id", ""),
                "object_anchor_id": rel.get("object_anchor_id", ""),
                "subject": rel.get("subject", ""),
                "predicate": rel.get("predicate", ""),
                "object": rel.get("object", ""),
                "type": rel.get("type", "other"),
                "rel_src_index": rel.get("rel_src_index", r_i),

                "subject_grounding_query": rel.get("subject_grounding_query", ""),
                "object_grounding_query": rel.get("object_grounding_query", ""),
                "subject_candidate_queries": rel.get("subject_candidate_queries", []),
                "object_candidate_queries": rel.get("object_candidate_queries", []),
                "subject_rejected_candidates": rel.get("subject_rejected_candidates", []),
                "object_rejected_candidates": rel.get("object_rejected_candidates", []),
                "subject_resolved_from_anchor_norm": rel.get("subject_resolved_from_anchor_norm"),
                "object_resolved_from_anchor_norm": rel.get("object_resolved_from_anchor_norm"),

                "subject_status": subj_status,
                "object_status": obj_status,

                "subject_detections": subj_det_out,
                "object_detections": obj_det_out,

                "union_mask_path": union_mask_path,
            })

        ensure_dir(out_dir)

        out_payload = {
            "id": payload.get("id", sid),
            "row_index": payload.get("row_index"),
            "image_path": str(image_path),
            "headline": payload.get("headline", ""),
            "summary": payload.get("summary", ""),
            "published": payload.get("published", ""),
            "url": payload.get("url", ""),
            "grounded_context_caption": payload.get("grounded_context_caption", ""),
            "anchors": out_anchors,
            "relation_anchors": out_rels,
        }

        with out_json.open("w", encoding="utf-8") as f:
            json.dump(out_payload, f, indent=2, ensure_ascii=False)

        print(f"[OK] Wrote {out_json}")
        print(f"[OK] Masks dir: {masks_dir}")

        if args.debug_draw:
            dbg = overlay_masks_for_debug(img_bgr, out_anchors, out_rels)
            dbg_path = out_dir / "debug_masks.jpg"
            cv2.imwrite(str(dbg_path), dbg)
            print(f"[DBG] Wrote {dbg_path}")

        a_total = len(out_anchors)
        a_ok = sum(1 for a in out_anchors if a.get("status") == "ok" and a.get("detections"))
        a_failed = sum(1 for a in out_anchors if a.get("status") == "sam_failed")
        a_nobox = sum(1 for a in out_anchors if a.get("status") == "no_box")

        r_total = len(out_rels)
        r_both_ok = sum(
            1 for r in out_rels
            if r.get("subject_status") == "ok" and r.get("object_status") == "ok"
        )
        r_non_gr = sum(
            1 for r in out_rels
            if "non_groundable" in (r.get("subject_status"), r.get("object_status"))
        )

        print(
            f"[SUMMARY] anchors={a_total} ok={a_ok} no_box={a_nobox} sam_failed={a_failed} | "
            f"relation_anchors={r_total} both_ok={r_both_ok} has_non_groundable={r_non_gr}"
        )

        processed += 1

    print(f"\n[DONE] Segmentation complete. processed={processed} skipped_existing={skipped_existing}")


if __name__ == "__main__":
    main()