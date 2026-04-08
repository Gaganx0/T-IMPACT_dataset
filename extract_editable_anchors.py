#!/usr/bin/env python3

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

ROOT = Path("/home/Student/s4826850/timpact")
MANIFEST_DIR = ROOT / "data" / "manifests"
OUTPUT_DIR = ROOT / "data" / "anchors"
OUTPUT_FILE = OUTPUT_DIR / "anchors_step1.jsonl"

MIN_MASK_RATIO = 0.002
MAX_MASK_RATIO = 0.60
MIN_GROUNDING_SCORE = 0.35
MIN_VERIFY_SCORE = 0.30

# Only reject truly vague anchors here.
VAGUE_TERMS = {
    "area",
    "background",
    "surface",
    "region",
    "scene",
}


def safe_str(x: Any) -> str:
    return x if isinstance(x, str) else ""


def safe_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []


def safe_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def is_vague_anchor(text: str) -> bool:
    t = safe_str(text).strip().lower()
    if not t:
        return True
    return any(term == t or f" {term} " in f" {t} " for term in VAGUE_TERMS)


def resolve_mask_path(mask_path: Optional[str]) -> Optional[str]:
    if not mask_path:
        return None
    p = Path(mask_path)
    if p.is_absolute():
        return str(p)
    return str((ROOT / mask_path).resolve())


def get_mask_ratio_and_image_area(anchor: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    stats = safe_dict(anchor.get("union_mask_stats"))
    w = stats.get("width")
    h = stats.get("height")
    fg = stats.get("foreground_area_px")

    if w is None or h is None or fg is None:
        return None, None

    try:
        image_area = float(w) * float(h)
        if image_area <= 0:
            return None, None
        return float(fg) / image_area, image_area
    except Exception:
        return None, None


def get_size_band(mask_ratio: Optional[float]) -> Optional[str]:
    if mask_ratio is None:
        return None
    if mask_ratio < 0.05:
        return "small"
    if mask_ratio < 0.25:
        return "medium"
    return "large"


def get_best_detection(anchor: Dict[str, Any]) -> Dict[str, Any]:
    best = anchor.get("best_detection")
    return best if isinstance(best, dict) else {}


def count_relations_for_anchor(sample: Dict[str, Any], anchor_id: str) -> int:
    rels = safe_list(sample.get("relations"))
    count = 0
    for r in rels:
        if not isinstance(r, dict):
            continue
        if r.get("subject_anchor_id") == anchor_id or r.get("object_anchor_id") == anchor_id:
            count += 1
    return count


def compute_technical_validity(anchor_row: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    status = anchor_row.get("status")
    grounding_status = anchor_row.get("grounding_status")
    union_mask_path = anchor_row.get("union_mask_path")
    mask_ratio = anchor_row.get("mask_ratio")
    grounding_score = anchor_row.get("grounding_score")
    qwen_verify_score = anchor_row.get("qwen_verify_score")
    anchor_text = anchor_row.get("anchor_text", "")

    if status != "ok" or grounding_status != "ok":
        return False, "status_not_ok"

    if not union_mask_path:
        return False, "no_union_mask"

    if mask_ratio is None:
        return False, "missing_mask_stats"

    if mask_ratio < MIN_MASK_RATIO:
        return False, "mask_too_small"

    if mask_ratio > MAX_MASK_RATIO:
        return False, "mask_too_large"

    if grounding_score is None or grounding_score < MIN_GROUNDING_SCORE:
        return False, "low_grounding_score"

    if qwen_verify_score is not None and qwen_verify_score < MIN_VERIFY_SCORE:
        return False, "low_verify_score"

    if is_vague_anchor(anchor_text):
        return False, "vague_anchor"

    return True, None


def normalize_anchor_norm(anchor_norm: Optional[str], anchor_text: str) -> str:
    a = safe_str(anchor_norm).strip().lower()
    if a:
        return a
    return safe_str(anchor_text).strip().lower()


def make_dedup_group(sample_id: str, anchor_norm: str, union_mask_path: Optional[str]) -> str:
    return f"{sample_id}::{anchor_norm}::{union_mask_path or 'NO_MASK'}"


def flatten_manifest(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        return []

    # Skip summary/index-style files if they sneak in
    if "sample_id" not in data or "anchors" not in data:
        return []

    sample_id = safe_str(data.get("sample_id"))
    image_path = safe_str(data.get("image_path"))
    headline = safe_str(data.get("headline"))
    summary = safe_str(data.get("summary"))

    caption = safe_dict(data.get("caption"))
    anchors_meta = safe_dict(data.get("anchors_meta"))

    literal_caption = safe_str(caption.get("image_caption_literal"))
    grounded_context_caption = safe_str(anchors_meta.get("grounded_context_caption"))

    rows: List[Dict[str, Any]] = []
    anchors = safe_list(data.get("anchors"))

    for idx, anchor in enumerate(anchors):
        if not isinstance(anchor, dict):
            continue

        anchor_id = safe_str(anchor.get("anchor_id")) or f"a{idx+1}"
        anchor_text = safe_str(anchor.get("anchor_text"))
        if not anchor_text:
            anchor_text = safe_str(safe_dict(anchor.get("qwen")).get("span"))

        anchor_norm = normalize_anchor_norm(anchor.get("anchor_norm"), anchor_text)

        category = safe_str(anchor.get("category")).lower()
        semantic_tags = safe_list(anchor.get("semantic_tags"))
        functional_role = safe_str(anchor.get("functional_role"))
        attributes_text = safe_str(anchor.get("attributes_text"))
        status = safe_str(anchor.get("status"))
        grounding_status = safe_str(anchor.get("grounding_status"))
        union_mask_path = resolve_mask_path(anchor.get("union_mask_path"))

        mask_ratio, image_area_px = get_mask_ratio_and_image_area(anchor)
        size_band = get_size_band(mask_ratio)

        best_det = get_best_detection(anchor)
        grounding_score = best_det.get("score")
        precision_score = best_det.get("precision_score")
        qwen_verify_score = best_det.get("qwen_verify_score")
        bbox_xyxy = best_det.get("box_xyxy")

        relation_count = count_relations_for_anchor(data, anchor_id)
        mask_area_px = safe_dict(anchor.get("union_mask_stats")).get("foreground_area_px")

        row: Dict[str, Any] = {
            "sample_id": sample_id,
            "image_path": image_path,
            "headline": headline,
            "summary": summary,
            "literal_caption": literal_caption,
            "grounded_context_caption": grounded_context_caption,
            "anchor_id": anchor_id,
            "anchor_text": anchor_text,
            "anchor_norm": anchor_norm,
            "category": category,
            "semantic_tags": semantic_tags,
            "functional_role": functional_role,
            "attributes_text": attributes_text,
            "status": status,
            "grounding_status": grounding_status,
            "union_mask_path": union_mask_path,
            "mask_area_px": mask_area_px,
            "image_area_px": image_area_px,
            "mask_ratio": mask_ratio,
            "size_band": size_band,
            "bbox_xyxy": bbox_xyxy,
            "grounding_score": grounding_score,
            "precision_score": precision_score,
            "qwen_verify_score": qwen_verify_score,
            "num_detections": len(safe_list(anchor.get("detections"))),
            "relation_count": relation_count,
        }

        technical_valid, technical_reject_reason = compute_technical_validity(row)
        row["technical_valid"] = technical_valid
        row["technical_reject_reason"] = technical_reject_reason

        row["dedup_group"] = make_dedup_group(sample_id, anchor_norm, union_mask_path)
        row["is_duplicate"] = False
        row["keep_for_step2"] = technical_valid

        rows.append(row)

    return rows


def mark_duplicates(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    If multiple rows in the same sample share the same dedup_group,
    keep the best one by:
      1. higher grounding_score
      2. higher qwen_verify_score
      3. larger mask_area_px
    """
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["dedup_group"], []).append(dict(row))  # defensive copy

    output_rows: List[Dict[str, Any]] = []

    for _, group_rows in groups.items():
        if len(group_rows) == 1:
            output_rows.append(group_rows[0])
            continue

        def sort_key(r: Dict[str, Any]):
            return (
                r["grounding_score"] if r["grounding_score"] is not None else -1.0,
                r["qwen_verify_score"] if r["qwen_verify_score"] is not None else -1.0,
                r["mask_area_px"] if r["mask_area_px"] is not None else -1,
            )

        group_rows_sorted = sorted(group_rows, key=sort_key, reverse=True)

        keeper = dict(group_rows_sorted[0])
        keeper["is_duplicate"] = False
        output_rows.append(keeper)

        for dup in group_rows_sorted[1:]:
            dup = dict(dup)
            dup["is_duplicate"] = True
            dup["technical_valid"] = False
            dup["technical_reject_reason"] = "duplicate"
            dup["keep_for_step2"] = False
            output_rows.append(dup)

    return output_rows


def collect_manifest_files(manifest_dir: Path) -> List[Path]:
    files = []
    for p in sorted(manifest_dir.glob("*.json")):
        if p.name == "index.json":
            continue
        if p.stem.isdigit():
            files.append(p)
    return files


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_files = collect_manifest_files(MANIFEST_DIR)
    if not manifest_files:
        raise FileNotFoundError(f"No per-sample manifest JSON files found in {MANIFEST_DIR}")

    all_rows: List[Dict[str, Any]] = []

    for mf in manifest_files:
        try:
            all_rows.extend(flatten_manifest(mf))
        except Exception as e:
            print(f"[WARN] Skipping malformed manifest {mf}: {e}")

    all_rows = mark_duplicates(all_rows)

    with OUTPUT_FILE.open("w", encoding="utf-8") as writer:
        for row in all_rows:
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")

    kept = sum(1 for r in all_rows if r.get("keep_for_step2"))
    duplicates = sum(1 for r in all_rows if r.get("is_duplicate"))

    print(f"Done. Wrote: {OUTPUT_FILE}")
    print(f"Total rows: {len(all_rows)}")
    print(f"Keep for step2: {kept}")
    print(f"Duplicates: {duplicates}")


if __name__ == "__main__":
    main()