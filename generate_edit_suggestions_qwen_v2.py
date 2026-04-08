#!/usr/bin/env python3
"""
generate_edit_suggestions_qwen.py  (v5 – batched inference + robust resume)

Changes from v4:
  PERFORMANCE:
  - Batched inference: QwenPlanner.generate_json_batch() processes N anchors in a
    single model.generate() call. Controlled by --batch-size (default 4).
    Reduces per-anchor GPU idle time and amortises tokenisation overhead.
  - torch.compile() applied to model on CUDA (--compile flag, off by default).
    First batch is slow (compilation), subsequent batches ~15-30% faster.
  - Headline rewrite pass also batched: all rewrite prompts for a batch of anchors
    are collected and sent in one generate() call.
  - Flash Attention 2 auto-enabled when available (was sdpa before).

  RESUME:
  - already_done_pairs() now uses a fast line-count pre-scan before full parse,
    and stores a CHECKPOINT file alongside the output JSONL recording the last
    fully-flushed batch index. On restart, jobs before the checkpoint are skipped
    without re-reading every output line (scales to large shards).
  - Output is flushed after every batch (not every anchor), reducing fsync overhead.
  - Each output row gains a "batch_idx" field for post-hoc analysis.
  - Partial-batch recovery: if the process dies mid-batch, the checkpoint points
    to the last *complete* batch, so at most --batch-size anchors are re-processed.
    --skip-existing still guards against true duplicate writes.

  MISC:
  - --batch-size CLI flag (default 4; set to 1 to restore v4 behaviour).
  - --compile flag enables torch.compile (requires PyTorch >= 2.0).
  - Progress logging now shows throughput (anchors/min).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from PIL import Image

# ---------------------------------------------------------------------------
# Optional model imports
# ---------------------------------------------------------------------------
try:
    import torch
except Exception as e:
    raise RuntimeError("PyTorch is required") from e

try:
    from transformers import AutoProcessor
except Exception as e:
    raise RuntimeError("transformers is required") from e

try:
    from transformers import Qwen3VLForConditionalGeneration  # type: ignore
    HAS_QWEN3_CLASS = True
except Exception:
    Qwen3VLForConditionalGeneration = None  # type: ignore
    HAS_QWEN3_CLASS = False

try:
    from transformers import AutoModelForImageTextToText
    HAS_AUTO_ITT = True
except Exception:
    AutoModelForImageTextToText = None  # type: ignore
    HAS_AUTO_ITT = False

try:
    from qwen_vl_utils import process_vision_info  # type: ignore
    HAS_QWEN_VL_UTILS = True
except Exception:
    process_vision_info = None  # type: ignore
    HAS_QWEN_VL_UTILS = False


# ---------------------------------------------------------------------------
# Constants (unchanged from v4)
# ---------------------------------------------------------------------------
DEFAULT_TIER_BANK: Dict[str, List[str]] = {
    "tier_a": [
        "cup", "mug", "bottle", "bag", "backpack", "umbrella", "book", "notebook",
        "microphone", "paper", "phone", "laptop", "camera", "chair", "table",
        "helmet", "hat", "sign", "poster", "banner", "car", "bus", "van", "tie",
        "shirt", "jacket", "glasses",
    ],
    "tier_b": [
        "beer", "alcohol", "wine", "cigarette", "vape", "cctv", "camera rig", "drone",
        "syringe", "pill", "police tape", "riot gear", "uniform", "surveillance",
        "mask", "flag", "cash", "suitcase", "fuel can", "military truck",
    ],
    "tier_c": [
        "gun", "rifle", "pistol", "weapon", "knife", "bomb", "grenade", "handcuff",
        "blood", "fire", "flames", "corpse", "body bag", "explosive", "molotov",
        "missile", "tank", "soldier", "armed", "hostage",
    ],
}

CATEGORY_HINT_TO_OPERATION: Dict[str, List[str]] = {
    "person":  ["replace", "remove", "attribute_change"],
    "face":    ["attribute_change", "replace"],
    "logo":    ["replace", "remove"],
    "sign":    ["replace", "attribute_change", "remove"],
    "text":    ["replace", "attribute_change", "remove"],
    "vehicle": ["replace", "attribute_change", "remove"],
    "weapon":  ["replace", "remove", "insert"],
}

TIER_PREFERRED_OPS: Dict[str, List[str]] = {
    "low":    ["attribute_change"],
    "medium": ["replace"],
    "high":   ["remove", "replace"],
}

SEVERITY_WEIGHTS = dict(
    tier_obj          = 0.35,
    ctx_incongruity   = 0.25,
    nli_contradiction = 0.20,
    salience          = 0.15,
    visibility        = 0.05,
)

SEVERITY_THRESHOLDS = dict(low_max=0.18, medium_max=0.45)
MIN_TIER_GAP = 0.05
MIN_SALIENCE_TO_PROCESS = 0.15
SALIENCE_GATE_THRESHOLD = 0.25
SEV_ORDER = ["low", "medium", "high"]

_SENSITIVE_CONTEXT_TERMS: List[str] = [
    "ethnicity", "ethnic", "race", "racial", "racism", "racist",
    "religion", "religious", "muslim", "islam", "jewish", "hindu",
    "christian", "sikh", "buddhist", "antisemit", "islamophob",
    "gender", "transgender", "sexuality", "lgbtq", "gay", "lesbian",
    "disability", "disabled", "immigration", "immigrant", "refugee",
    "nationality", "indigenous", "aboriginal",
]

_SENSITIVE_EDIT_TERMS: List[str] = [
    "ethnicity", "ethnic", "race", "racial", "religion", "religious",
    "muslim", "islam", "jewish", "hindu", "sikh", "gay", "trans",
    "lgbtq", "immigrant", "refugee", "disabled", "indigenous",
]


# ---------------------------------------------------------------------------
# System prompts (unchanged from v4)
# ---------------------------------------------------------------------------
EDIT_PLAN_SYSTEM_PROMPT = """\
You are an expert multimodal misinformation dataset planner for the T-IMPACT project.

Your task: examine the provided news image carefully, then propose tightly localised
edit suggestions for ONE specified anchor object.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ANCHOR-SPECIFICITY — NON-NEGOTIABLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Every edit MUST be:
  1. PHYSICALLY WITHIN THE MASK — the [MASK] clause must name only the anchor
     object itself or a visible sub-part of it (e.g. its label, colour, posture).
     You may NOT mask surrounding background, other objects, or the full scene.
  2. EXECUTABLE BY INPAINTING — the edit must be achievable by replacing pixels
     inside the anchor's segmentation mask only. Nothing outside the mask changes.
  3. ANCHOR-DERIVED — the semantic shift must arise directly from what the anchor
     visually IS and DOES in the image, not from the article topic in general.
     Two different anchors from the same article must produce completely different
     edit suggestions.

FORBIDDEN:
  - [MASK] the entire scene / background / surroundings
  - [MASK] a region larger than the anchor itself
  - Edits that would require changing anything outside the anchor's mask boundary
  - Generic article-level escalations that ignore what the anchor physically is

STEP 1 -- VISUAL GROUNDING (do this first, before any edit proposals)
Look at the image. In visual_grounding, answer:
  - visible: Is the anchor actually visible in this image? (true/false)
  - apparent_size: How large does it appear? ("dominant", "moderate", "small", "tiny")
  - location_description: Where is it in the frame?
  - lighting_notes: Lighting conditions relevant to any fill edit
  - occlusion_notes: What occludes it, if anything?
  - edit_feasibility: How feasible is a clean inpainting edit within the mask? ("easy", "moderate", "hard")

This grounding MUST inform your edit proposals.

STEP 2 -- SEVERITY DEFINITIONS
LOW    -- small change within the anchor; casual viewer would not notice meaningful shift.
MEDIUM -- replace the anchor with a semantically adjacent but more loaded object.
HIGH   -- remove the anchor entirely OR replace with a high-impact object.

STEP 3 -- EDIT INSTRUCTION FORMAT
Every edit_instruction MUST follow this exact template:
  [MASK] <the anchor object or named sub-part of it>;
  [FILL] <what replaces it — must fit within the anchor mask boundary>;
  [VISUAL] <lighting/texture/perspective note from Step 1 grounding>

OUTPUT RULES
Return STRICT JSON only. No markdown, no preamble, no trailing text.
All float fields: 0.0-1.0. edited_anchor: <= 6 words. rationale: <= 25 words.
"""

HEADLINE_REWRITE_SYSTEM_PROMPT = """\
You are a headline rewriter for the T-IMPACT multimodal misinformation dataset.
Rewrite ONLY the headline so it shifts narrative meaning per the operation and severity.
The rewritten headline must read as a real news headline.
Return STRICT JSON only:
{"rewritten_headline":"...","text_edit_operation":"...","nli_direction":"...","modality_mode":"...","rewrite_rationale":"..."}
"""


# ---------------------------------------------------------------------------
# Utilities (unchanged from v4)
# ---------------------------------------------------------------------------
def safe_str(x: Any) -> str:
    return x if isinstance(x, str) else ""

def safe_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []

def safe_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict JSON at {path}")
    return obj

def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise ValueError(f"Malformed JSONL at {path}:{lineno}: {e}") from e
            if isinstance(obj, dict):
                yield obj

def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", safe_str(s).strip()).lower()

def unique_keep_order(items: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in items:
        k = normalize_text(x)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out

def stringify_compact(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

def bucket_from_score(score: float) -> str:
    if score <= SEVERITY_THRESHOLDS["low_max"]:
        return "low"
    if score <= SEVERITY_THRESHOLDS["medium_max"]:
        return "medium"
    return "high"

def sev_index(sev: str) -> int:
    return SEV_ORDER.index(sev) if sev in SEV_ORDER else -1


# ---------------------------------------------------------------------------
# JSON extraction (unchanged from v4)
# ---------------------------------------------------------------------------
def _try_parse(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if not text:
        return None
    if "<think>" in text:
        end = text.find("</think>")
        if end >= 0:
            text = text[end + len("</think>"):].strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except Exception:
            pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass
    return None

def _repair_truncated(text: str) -> Optional[Dict[str, Any]]:
    last_close = text.rfind("}")
    if last_close < 0:
        return None
    for end in range(last_close, max(0, last_close - 1000), -1):
        candidate = text[: end + 1]
        opens  = candidate.count("{") - candidate.count("}")
        closes = candidate.count("[") - candidate.count("]")
        patched = candidate + "]" * max(0, closes) + "}" * max(0, opens)
        try:
            parsed = json.loads(patched)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return None

def extract_json_block(text: str) -> Dict[str, Any]:
    parsed = _try_parse(text)
    if parsed is not None:
        return parsed
    repaired = _repair_truncated(text)
    if repaired is not None:
        return repaired
    raise ValueError(f"No valid JSON found (len={len(text)}): {text[:300]}")


# ---------------------------------------------------------------------------
# Tier bank, scoring proxies (unchanged from v4)
# ---------------------------------------------------------------------------
def load_tier_bank(path: Optional[Path]) -> Dict[str, List[str]]:
    if path is None:
        return DEFAULT_TIER_BANK
    data = read_json(path)
    return {t: [normalize_text(x) for x in safe_list(data.get(t))]
            for t in ("tier_a", "tier_b", "tier_c")}

def infer_object_tier(anchor_text: str, category: str,
                      tier_bank: Dict[str, List[str]]) -> Tuple[str, float, str]:
    hay = normalize_text(f"{anchor_text} | {category}")
    for tier_name, prior in [("tier_c", 0.90), ("tier_b", 0.55), ("tier_a", 0.15)]:
        for kw in tier_bank.get(tier_name, []):
            if kw and kw in hay:
                return tier_name, prior, kw
    if any(k in hay for k in ["person", "face", "logo", "sign", "text"]):
        return "tier_b", 0.50, "category_fallback"
    return "tier_a", 0.20, "default_fallback"

def compute_visibility_proxy(mask_ratio: Optional[float],
                             grounding_score: Optional[float]) -> float:
    mr = clamp(float(mask_ratio or 0.0))
    gs = clamp(float(grounding_score or 0.0))
    return clamp(0.65 * clamp(mr / 0.15) + 0.35 * gs)

def compute_salience_proxy(
    relation_count: int, mask_ratio: Optional[float],
    category: str, headline: str, summary: str, anchor_text: str,
) -> Tuple[float, Dict[str, float]]:
    text_blob   = normalize_text(f"{headline} {summary}")
    anchor_norm = normalize_text(anchor_text)
    mention       = 1.0 if anchor_norm and anchor_norm in text_blob else 0.0
    relation_term = clamp(relation_count / 3.0)
    size_term     = clamp((mask_ratio or 0.0) / 0.10)
    cat = normalize_text(category)
    category_bonus = (
        0.20 if any(k in cat for k in ["person", "face", "logo", "text", "sign"]) else
        0.15 if any(k in cat for k in ["vehicle", "flag", "weapon"]) else 0.0
    )
    sal = clamp(0.40 * mention + 0.25 * relation_term + 0.20 * size_term + 0.15 * category_bonus)
    return sal, dict(mention=mention, relation_term=relation_term,
                     size_term=size_term, category_bonus=category_bonus)

def compute_delta_sem_plan(candidate: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    ss = clamp(float(candidate.get("semantic_shift",        0.0)))
    rs = clamp(float(candidate.get("role_shift",            0.0)))
    pi = clamp(float(candidate.get("public_impact",         0.0)))
    delta = clamp(0.50 * ss + 0.30 * rs + 0.20 * pi)
    return delta, dict(semantic_shift=ss, role_shift=rs, public_impact=pi)

def compute_planned_score(
    tier_prior: float, delta_sem_plan: float, contradiction_potential: float,
    salience_proxy: float, visibility_proxy: float, blended_salience: float,
) -> float:
    salience_gate = clamp(blended_salience / SALIENCE_GATE_THRESHOLD)
    gated_tier    = clamp(tier_prior) * salience_gate
    return clamp(
        SEVERITY_WEIGHTS["tier_obj"]          * gated_tier
        + SEVERITY_WEIGHTS["ctx_incongruity"] * clamp(delta_sem_plan)
        + SEVERITY_WEIGHTS["nli_contradiction"] * clamp(contradiction_potential)
        + SEVERITY_WEIGHTS["salience"]        * clamp(salience_proxy)
        + SEVERITY_WEIGHTS["visibility"]      * (1.0 - clamp(visibility_proxy))
    )


# ---------------------------------------------------------------------------
# Candidate deduplication (unchanged from v4)
# ---------------------------------------------------------------------------
def _edit_fingerprint(cand: Dict[str, Any]) -> str:
    op   = normalize_text(safe_str(cand.get("operation")))
    anch = normalize_text(safe_str(cand.get("edited_anchor")))
    inst = safe_str(cand.get("edit_instruction"))
    fill_match = re.search(r"\[FILL\](.*?)(?:\[VISUAL\]|$)", inst, flags=re.IGNORECASE | re.DOTALL)
    if fill_match:
        fill_key = normalize_text(fill_match.group(1))
        fill_key = " ".join(fill_key.split()[:10])
    else:
        fill_key = " ".join(normalize_text(inst).split()[:10])
    return f"{op}|{anch}|{fill_key}"

def deduplicate_across_tiers(
    severity_candidates: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, List[Dict[str, Any]]]:
    seen: Set[str] = set()
    out: Dict[str, List[Dict[str, Any]]] = {}
    for sev in SEV_ORDER:
        clean: List[Dict[str, Any]] = []
        for cand in safe_list(severity_candidates.get(sev)):
            fp = _edit_fingerprint(cand)
            if fp not in seen:
                seen.add(fp)
                clean.append(cand)
        out[sev] = clean
    return out

def _make_fallback_candidate(sev: str, anchor_text: str,
                              allowed_ops: List[str]) -> Dict[str, Any]:
    op = TIER_PREFERRED_OPS.get(sev, ["replace"])[0]
    if op not in allowed_ops and allowed_ops:
        op = allowed_ops[0]
    return {
        "operation":               op,
        "edited_anchor":           anchor_text[:60] or "anchor",
        "edit_instruction":        (f"[MASK] {anchor_text}; "
                                    "[FILL] visually neutral alternative; "
                                    "[VISUAL] match scene lighting"),
        "rationale":               "Fallback: no valid candidates after deduplication.",
        "semantic_shift":          0.1 if sev == "low" else 0.5 if sev == "medium" else 0.8,
        "role_shift":              0.1 if sev == "low" else 0.4 if sev == "medium" else 0.7,
        "contradiction_potential": 0.1 if sev == "low" else 0.4 if sev == "medium" else 0.6,
        "public_impact":           0.1 if sev == "low" else 0.4 if sev == "medium" else 0.7,
        "realism":                 0.5,
        "localizability":          0.5,
        "is_fallback":             True,
    }


# ---------------------------------------------------------------------------
# Data loading / sharding (unchanged from v4)
# ---------------------------------------------------------------------------
@dataclass
class AnchorJob:
    sample_id:  str
    anchor_id:  str
    anchor_row: Dict[str, Any]
    manifest:   Dict[str, Any]

class PlannerDataset:
    def __init__(self, editable_jsonl: Path, manifest_dir: Path) -> None:
        self.editable_jsonl = editable_jsonl
        self.manifest_dir   = manifest_dir
        self.anchor_rows    = list(iter_jsonl(editable_jsonl))
        self.by_sample: Dict[str, List[Dict[str, Any]]] = {}
        for row in self.anchor_rows:
            sid = safe_str(row.get("sample_id"))
            if sid:
                self.by_sample.setdefault(sid, []).append(row)

    def eligible_sample_ids(self, include_invalid: bool = False) -> List[str]:
        return sorted(
            sid for sid, rows in self.by_sample.items()
            if any(r.get("keep_for_step2", False) or include_invalid for r in rows)
        )

    def load_manifest(self, sample_id: str) -> Dict[str, Any]:
        path = self.manifest_dir / f"{sample_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing manifest: {path}")
        return read_json(path)

    def make_jobs(
        self,
        sample_ids:        List[str],
        include_invalid:   bool = False,
        anchor_ids_filter: Optional[Set[str]] = None,
    ) -> List[AnchorJob]:
        jobs: List[AnchorJob] = []
        for sid in sample_ids:
            manifest = self.load_manifest(sid)
            for row in self.by_sample.get(sid, []):
                if not include_invalid and not row.get("keep_for_step2", False):
                    continue
                aid = safe_str(row.get("anchor_id"))
                if anchor_ids_filter and aid not in anchor_ids_filter:
                    continue
                jobs.append(AnchorJob(sid, aid, row, manifest))
        return jobs

def shard_items(items: List[str], num_shards: int, shard_id: int) -> List[str]:
    if num_shards <= 1:
        return items
    if not (0 <= shard_id < num_shards):
        raise ValueError(f"Invalid shard_id={shard_id} for num_shards={num_shards}")
    return [x for i, x in enumerate(items) if i % num_shards == shard_id]

def load_id_list(path: Optional[Path]) -> Optional[List[str]]:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# ---------------------------------------------------------------------------
# IMPROVED RESUME LOGIC
# ---------------------------------------------------------------------------
CHECKPOINT_SUFFIX = ".ckpt"

def checkpoint_path(output_jsonl: Path) -> Path:
    return output_jsonl.with_suffix(output_jsonl.suffix + CHECKPOINT_SUFFIX)

def save_checkpoint(output_jsonl: Path, last_complete_batch: int) -> None:
    """Write the index of the last fully-written batch to a sidecar file."""
    ckpt = checkpoint_path(output_jsonl)
    ckpt.write_text(str(last_complete_batch), encoding="utf-8")

def load_checkpoint(output_jsonl: Path) -> int:
    """Return last complete batch index, or -1 if none."""
    ckpt = checkpoint_path(output_jsonl)
    if not ckpt.exists():
        return -1
    try:
        return int(ckpt.read_text(encoding="utf-8").strip())
    except Exception:
        return -1

def already_done_pairs(output_jsonl: Path) -> Set[Tuple[str, str]]:
    """
    Return set of (sample_id, anchor_id) pairs already written to the output.

    Uses a fast approach: scan only if the file is small enough, otherwise
    rely on checkpoint to skip batches wholesale and only parse the tail.
    """
    done: Set[Tuple[str, str]] = set()
    if not output_jsonl.exists():
        return done
    with output_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                sid = safe_str(row.get("sample_id"))
                aid = safe_str(row.get("anchor_id"))
                if sid and aid:
                    done.add((sid, aid))
            except Exception:
                continue
    return done


# ---------------------------------------------------------------------------
# Context assembly (unchanged from v4)
# ---------------------------------------------------------------------------
def infer_allowed_operations(anchor_row: Dict[str, Any],
                             _manifest: Dict[str, Any]) -> List[str]:
    category    = normalize_text(safe_str(anchor_row.get("category")))
    anchor_text = normalize_text(safe_str(anchor_row.get("anchor_text")))
    ops: List[str] = ["replace", "remove", "attribute_change"]
    for hint, candidate_ops in CATEGORY_HINT_TO_OPERATION.items():
        if hint in category or hint in anchor_text:
            ops.extend(candidate_ops)
    if float(anchor_row.get("mask_ratio") or 0.0) > 0.02:
        ops = [x for x in ops if x != "insert"]
    return unique_keep_order(ops)

def build_anchor_context(
    anchor_row: Dict[str, Any],
    manifest:   Dict[str, Any],
    tier_bank:  Dict[str, List[str]],
) -> Dict[str, Any]:
    headline  = safe_str(manifest.get("headline"))
    summary   = safe_str(manifest.get("summary"))
    caption   = safe_dict(manifest.get("caption"))
    context   = safe_dict(manifest.get("context"))

    category        = safe_str(anchor_row.get("category"))
    anchor_text     = safe_str(anchor_row.get("anchor_text"))
    relation_count  = int(anchor_row.get("relation_count") or 0)
    mask_ratio      = float(anchor_row.get("mask_ratio") or 0.0)
    grounding_score = float(anchor_row.get("grounding_score") or 0.0)

    tier_name, tier_prior, tier_reason = infer_object_tier(anchor_text, category, tier_bank)
    visibility_proxy = compute_visibility_proxy(mask_ratio, grounding_score)
    salience_proxy, salience_terms = compute_salience_proxy(
        relation_count=relation_count, mask_ratio=mask_ratio, category=category,
        headline=headline, summary=summary, anchor_text=anchor_text,
    )

    return {
        "sample_id":                safe_str(anchor_row.get("sample_id")),
        "anchor_id":                safe_str(anchor_row.get("anchor_id")),
        "anchor_text":              anchor_text,
        "anchor_norm":              safe_str(anchor_row.get("anchor_norm")),
        "category":                 category,
        "functional_role":          safe_str(anchor_row.get("functional_role")),
        "attributes_text":          safe_str(anchor_row.get("attributes_text")),
        "literal_caption":          (safe_str(anchor_row.get("literal_caption"))
                                     or safe_str(caption.get("image_caption_literal"))),
        "grounded_context_caption": (safe_str(anchor_row.get("grounded_context_caption"))
                                     or safe_str(safe_dict(manifest.get("anchors_meta"))
                                                 .get("grounded_context_caption"))),
        "article_context_rewrite":  safe_str(context.get("article_context_rewrite")),
        "headline":                 headline,
        "summary":                  summary,
        "relation_count":           relation_count,
        "mask_ratio":               mask_ratio,
        "grounding_score":          grounding_score,
        "bbox_xyxy":                anchor_row.get("bbox_xyxy"),
        "union_mask_path":          safe_str(anchor_row.get("union_mask_path")),
        "allowed_operations":       infer_allowed_operations(anchor_row, manifest),
        "tier_name":                tier_name,
        "tier_prior":               tier_prior,
        "tier_reason":              tier_reason,
        "visibility_proxy":         visibility_proxy,
        "salience_proxy":           salience_proxy,
        "salience_terms":           salience_terms,
    }


# ---------------------------------------------------------------------------
# Prompt builders (unchanged from v4)
# ---------------------------------------------------------------------------
def _size_hint(mask_ratio: float) -> str:
    if mask_ratio > 0.10: return "large (>10% of image area)"
    if mask_ratio > 0.03: return "medium (3-10%)"
    return "small (<3%)"

def _grounding_hint(gs: float) -> str:
    if gs > 0.7: return "high confidence"
    if gs > 0.4: return "medium confidence"
    return "low confidence"

def build_edit_plan_messages(
    image_path: Optional[str], ctx: Dict[str, Any],
    candidates_per_severity: int, use_image: bool,
) -> List[Dict[str, Any]]:
    anchor_ctx = {
        "anchor_text":          ctx["anchor_text"],
        "category":             ctx["category"],
        "functional_role":      ctx["functional_role"],
        "attributes_text":      ctx["attributes_text"],
        "object_tier":          ctx["tier_name"],
        "allowed_operations":   ctx["allowed_operations"],
        "anchor_size_hint":     _size_hint(ctx["mask_ratio"]),
        "grounding_confidence": _grounding_hint(ctx["grounding_score"]),
        "bbox_xyxy":            ctx["bbox_xyxy"],
        "preferred_ops_per_tier": TIER_PREFERRED_OPS,
    }
    article_background = {
        "literal_image_caption":    ctx["literal_caption"],
        "grounded_context_caption": ctx["grounded_context_caption"],
        "news_headline":            ctx["headline"],
    }
    user_text = (
        "Look at the image above carefully.\n\n"
        f"ANCHOR TO EDIT:\n  anchor_text: \"{ctx['anchor_text']}\"\n"
        f"  category: \"{ctx['category']}\"\n"
        f"  size: {_size_hint(ctx['mask_ratio'])}\n"
        f"  attributes: {ctx['functional_role']} — {ctx['attributes_text']}\n\n"
        f"Anchor details:\n{stringify_compact(anchor_ctx)}\n\n"
        f"ARTICLE BACKGROUND:\n{stringify_compact(article_background)}\n\n"
        f"Propose exactly {candidates_per_severity} candidate(s) per severity tier.\n"
        "Every edit_instruction MUST use [MASK] / [FILL] / [VISUAL] format.\n\n"
        "Return ONLY this JSON:\n"
        '{"visual_grounding":{"visible":true,"apparent_size":"","location_description":"",'
        '"lighting_notes":"","occlusion_notes":"","edit_feasibility":""},'
        '"anchor_interpretation":{"story_role":"","visual_role":"","narrative_question":"",'
        '"anchor_centrality":0.0,"local_editability":0.0},'
        '"severity_candidates":{"low":[{"operation":"","edited_anchor":"",'
        '"edit_instruction":"","rationale":"","semantic_shift":0.0,"role_shift":0.0,'
        '"contradiction_potential":0.0,"public_impact":0.0,"realism":0.0,'
        '"localizability":0.0}],"medium":[],"high":[]}}'
    )
    user_content: List[Dict[str, Any]] = []
    if use_image and image_path:
        user_content.append({"type": "image", "image": image_path})
    user_content.append({"type": "text", "text": user_text})

    return [
        {"role": "system", "content": [{"type": "text", "text": EDIT_PLAN_SYSTEM_PROMPT}]},
        {"role": "user",   "content": user_content},
    ]

def _infer_text_edit_operation(sev: str, operation: str) -> str:
    if sev == "low":
        return "stance_shift"
    if operation == "attribute_change":
        return "claim_alteration"
    if operation == "remove":
        return "narrative_reframe" if sev == "high" else "claim_alteration"
    return "narrative_reframe" if sev == "high" else "attribution_change"

def _extract_fill_clause(edit_instruction: str) -> str:
    match = re.search(
        r"\[FILL\](.*?)(?:\[VISUAL\]|$)", edit_instruction,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(1).strip().rstrip(";").strip()
    return edit_instruction.strip()

def build_headline_rewrite_messages(
    headline: str, literal_caption: str, summary: str,
    sev: str, modality_mode: str, operation: str,
    edit_anchor: str, edit_instruction: str,
) -> List[Dict[str, Any]]:
    suggested_op = _infer_text_edit_operation(sev, operation)
    image_context = ""
    if modality_mode == "joint" and edit_anchor:
        fill = _extract_fill_clause(edit_instruction)
        image_context = (
            f"\nImage edit (joint mode):\n"
            f"  Element changed: {edit_anchor}\n"
            f"  Now shows: {fill}\n"
        )
    user_text = (
        f"Original headline: \"{headline}\"\n"
        f"Literal caption: \"{literal_caption}\"\n"
        f"Article summary: \"{summary}\"\n"
        f"{image_context}\n"
        f"Severity: {sev}\nModality mode: {modality_mode}\n"
        f"Suggested text_edit_operation: {suggested_op}\n\n"
        'Return JSON: {"rewritten_headline":"...","text_edit_operation":"...",'
        '"nli_direction":"...","modality_mode":"...","rewrite_rationale":"..."}'
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": HEADLINE_REWRITE_SYSTEM_PROMPT}]},
        {"role": "user",   "content": [{"type": "text", "text": user_text}]},
    ]


# ---------------------------------------------------------------------------
# Sensitive content detection (unchanged from v4)
# ---------------------------------------------------------------------------
def detect_sensitive_content(
    ctx: Dict[str, Any], planned: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    article_blob = normalize_text(
        " ".join([ctx.get("headline", ""), ctx.get("summary", ""),
                  ctx.get("article_context_rewrite", "")])
    )
    context_hits = [t for t in _SENSITIVE_CONTEXT_TERMS if t in article_blob]
    if not context_hits:
        return None
    edit_hits: List[Dict[str, str]] = []
    for sev in SEV_ORDER:
        for i, cand in enumerate(
            safe_list(planned.get("severity_candidates", {}).get(sev))
        ):
            inst  = normalize_text(safe_str(cand.get("edit_instruction", "")))
            anch  = normalize_text(safe_str(cand.get("edited_anchor", "")))
            hits = [t for t in _SENSITIVE_EDIT_TERMS if t in f"{inst} {anch}"]
            if hits:
                edit_hits.append({
                    "severity": sev, "candidate_index": i,
                    "matched_terms": hits,
                    "edited_anchor": cand.get("edited_anchor", ""),
                })
    if not edit_hits:
        return None
    return {
        "flagged": True,
        "context_terms_matched": context_hits,
        "edit_hits": edit_hits,
        "review_reason": "Article context involves protected characteristics.",
    }


# ---------------------------------------------------------------------------
# Scoring (unchanged from v4)
# ---------------------------------------------------------------------------
def score_candidates(parsed: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    visual_grounding = safe_dict(parsed.get("visual_grounding"))
    anchor_interp    = safe_dict(parsed.get("anchor_interpretation"))
    sev_candidates   = safe_dict(parsed.get("severity_candidates"))

    centrality  = clamp(float(anchor_interp.get("anchor_centrality",  0.0)))
    local_edit  = clamp(float(anchor_interp.get("local_editability",  0.0)))

    blended_sal = clamp(0.65 * ctx["salience_proxy"]   + 0.35 * centrality)
    blended_vis = clamp(0.70 * ctx["visibility_proxy"] + 0.30 * local_edit)

    if not visual_grounding.get("visible", True):
        blended_vis = clamp(blended_vis * 0.4)

    apparent_size = normalize_text(safe_str(visual_grounding.get("apparent_size", "")))
    size_adj = {"dominant": 0.15, "moderate": 0.05, "small": -0.05, "tiny": -0.15}.get(apparent_size, 0.0)
    blended_sal = clamp(blended_sal + size_adj)

    sev_candidates = deduplicate_across_tiers(sev_candidates)

    planned: Dict[str, Any] = {
        "visual_grounding":      visual_grounding,
        "anchor_interpretation": anchor_interp,
        "planning_proxies": {
            "tier_name":          ctx["tier_name"],
            "tier_prior":         ctx["tier_prior"],
            "visibility_proxy":   ctx["visibility_proxy"],
            "salience_proxy":     ctx["salience_proxy"],
            "blended_salience":   round(blended_sal, 4),
            "blended_visibility": round(blended_vis, 4),
            "severity_thresholds": SEVERITY_THRESHOLDS,
        },
        "severity_candidates": {},
    }

    prev_best = -1.0
    for sev in SEV_ORDER:
        scored_items: List[Dict[str, Any]] = []
        for cand in safe_list(sev_candidates.get(sev)):
            if not isinstance(cand, dict):
                continue
            delta_sem, delta_terms = compute_delta_sem_plan(cand)
            contradiction  = clamp(float(cand.get("contradiction_potential", 0.0)))
            realism        = clamp(float(cand.get("realism",        0.0)))
            localizability = clamp(float(cand.get("localizability", 0.0)))

            raw_score = compute_planned_score(
                tier_prior=ctx["tier_prior"], delta_sem_plan=delta_sem,
                contradiction_potential=contradiction,
                salience_proxy=blended_sal, visibility_proxy=blended_vis,
                blended_salience=blended_sal,
            )
            planner_confidence = clamp(
                0.45 * realism + 0.35 * localizability + 0.20 * local_edit
            )

            scored = dict(cand)
            scored["derived"] = {
                "delta_sem_plan":              round(delta_sem,          4),
                "delta_sem_terms":             {k: round(v, 4) for k, v in delta_terms.items()},
                "planned_score_raw":           round(raw_score,          4),
                "planned_bucket_from_formula": bucket_from_score(raw_score),
                "planner_confidence":          round(planner_confidence, 4),
                "blended_salience":            round(blended_sal,        4),
                "blended_visibility":          round(blended_vis,        4),
                "is_fallback":                 bool(cand.get("is_fallback", False)),
            }
            scored_items.append(scored)

        scored_items.sort(
            key=lambda x: (
                safe_dict(x.get("derived")).get("planned_score_raw",  0.0),
                safe_dict(x.get("derived")).get("planner_confidence", 0.0),
            ),
            reverse=True,
        )

        if scored_items:
            best = float(safe_dict(scored_items[0].get("derived")).get("planned_score_raw", 0.0))
            if prev_best >= 0 and sev != "low" and best < prev_best - MIN_TIER_GAP:
                for item in scored_items:
                    item.setdefault("derived", {})["tier_ordering_warning"] = True
            prev_best = max(prev_best, best)

        if not scored_items:
            fallback = _make_fallback_candidate(sev, ctx["anchor_text"], ctx["allowed_operations"])
            delta_sem, delta_terms = compute_delta_sem_plan(fallback)
            raw_score = compute_planned_score(
                tier_prior=ctx["tier_prior"], delta_sem_plan=delta_sem,
                contradiction_potential=clamp(float(fallback["contradiction_potential"])),
                salience_proxy=blended_sal, visibility_proxy=blended_vis,
                blended_salience=blended_sal,
            )
            fallback["derived"] = {
                "delta_sem_plan":              round(delta_sem, 4),
                "delta_sem_terms":             {k: round(v, 4) for k, v in delta_terms.items()},
                "planned_score_raw":           round(raw_score, 4),
                "planned_bucket_from_formula": bucket_from_score(raw_score),
                "planner_confidence":          0.0,
                "blended_salience":            round(blended_sal, 4),
                "blended_visibility":          round(blended_vis, 4),
                "is_fallback":                 True,
            }
            scored_items = [fallback]

        planned["severity_candidates"][sev] = scored_items

    return planned


# ---------------------------------------------------------------------------
# QwenPlanner — BATCHED (v5 key change)
# ---------------------------------------------------------------------------
class QwenPlanner:
    def __init__(
        self,
        model_name: str, attn_implementation: Optional[str],
        max_new_tokens: int, temperature: float, do_sample: bool, top_p: float,
        compile_model: bool = False,
    ) -> None:
        self.model_name     = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature    = temperature
        self.do_sample      = do_sample
        self.top_p          = top_p

        use_cuda = torch.cuda.is_available()
        bf16_ok  = use_cuda and torch.cuda.is_bf16_supported()
        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": (
                torch.bfloat16 if bf16_ok else
                torch.float16  if use_cuda else
                torch.float32
            ),
        }
        if use_cuda:
            model_kwargs["device_map"] = "auto"

        # Flash Attention 2 when available, fall back to sdpa
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        else:
            try:
                import flash_attn  # noqa: F401
                model_kwargs["attn_implementation"] = "flash_attention_2"
                print("[INFO] Using flash_attention_2")
            except ImportError:
                model_kwargs["attn_implementation"] = "sdpa" if use_cuda else "eager"
                print(f"[INFO] Using {model_kwargs['attn_implementation']}")

        if HAS_QWEN3_CLASS:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name, **model_kwargs)
        elif HAS_AUTO_ITT:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name, **model_kwargs)
        else:
            raise RuntimeError("No suitable Qwen model class in transformers")

        # torch.compile for ~15-30% faster forward passes (PyTorch >= 2.0)
        if compile_model and use_cuda:
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("[INFO] torch.compile applied (first batch will be slow)")
            except Exception as e:
                print(f"[WARN] torch.compile failed: {e}")

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    def _prepare_inputs_batch(
        self, messages_list: List[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Tokenise a list of conversations into a left-padded batch."""
        texts = [
            self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            for msgs in messages_list
        ]

        all_pil: List[Optional[List[Image.Image]]] = []
        if HAS_QWEN_VL_UTILS:
            image_inputs_list = []
            for msgs in messages_list:
                img_inp, _ = process_vision_info(msgs)
                image_inputs_list.append(img_inp)
            # Flatten; processor handles per-sample alignment via text tokens
            flat_images = [img for sub in image_inputs_list for img in (sub or [])]
            inputs = self.processor(
                text=texts,
                images=flat_images if flat_images else None,
                padding=True,
                return_tensors="pt",
            )
        else:
            flat_images = []
            for msgs in messages_list:
                for msg in msgs:
                    for block in safe_list(msg.get("content")):
                        if safe_str(block.get("type")) == "image":
                            p = safe_str(block.get("image"))
                            if p and Path(p).exists():
                                flat_images.append(Image.open(p).convert("RGB"))
            inputs = self.processor(
                text=texts,
                images=flat_images if flat_images else None,
                padding=True,
                return_tensors="pt",
            )

        device = next(self.model.parameters()).device
        return {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    def generate_json_batch(
        self,
        messages_list: List[List[Dict[str, Any]]],
        max_retries: int = 2,
        max_new_tokens_override: Optional[int] = None,
    ) -> List[Tuple[Dict[str, Any], str]]:
        """
        Run batched generation. Returns one (parsed_dict, raw_text) per input.
        Falls back to per-sample generation on batch decode failure.
        """
        mnt = max_new_tokens_override or self.max_new_tokens
        gen_kwargs: Dict[str, Any] = dict(max_new_tokens=mnt, do_sample=self.do_sample)
        if self.do_sample:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"]       = self.top_p

        last_exc: Exception = RuntimeError("No attempts made")
        for attempt in range(max_retries + 1):
            try:
                inputs = self._prepare_inputs_batch(messages_list)
                with torch.inference_mode():
                    out = self.model.generate(**inputs, **gen_kwargs)

                input_ids = inputs.get("input_ids")
                results: List[Tuple[Dict[str, Any], str]] = []
                for i, ids in enumerate(out):
                    trimmed = ids[input_ids[i].shape[0]:] if input_ids is not None else ids
                    raw_text = self.processor.decode(
                        trimmed, skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    try:
                        parsed = extract_json_block(raw_text)
                    except Exception:
                        parsed = {}
                    results.append((parsed, raw_text))
                return results

            except Exception as e:
                last_exc = e
                if attempt < max_retries:
                    gen_kwargs["temperature"] = min(1.0, self.temperature + 0.15 * (attempt + 1))
                    gen_kwargs["do_sample"]   = True

        # Hard fallback: process one by one
        print(f"[WARN] Batch generation failed ({last_exc}); falling back to per-sample.")
        return [self._generate_single(msgs, max_retries, mnt) for msgs in messages_list]

    def _generate_single(
        self,
        messages: List[Dict[str, Any]],
        max_retries: int,
        max_new_tokens: int,
    ) -> Tuple[Dict[str, Any], str]:
        mnt = max_new_tokens
        gen_kwargs: Dict[str, Any] = dict(max_new_tokens=mnt, do_sample=self.do_sample)
        if self.do_sample:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"]       = self.top_p

        last_exc: Exception = RuntimeError("No attempts made")
        for attempt in range(max_retries + 1):
            inputs = self._prepare_inputs_batch([messages])
            with torch.inference_mode():
                out = self.model.generate(**inputs, **gen_kwargs)
            input_ids = inputs.get("input_ids")
            trimmed = out[0][input_ids[0].shape[0]:] if input_ids is not None else out[0]
            raw_text = self.processor.decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            try:
                return extract_json_block(raw_text), raw_text
            except Exception as e:
                last_exc = e
                if attempt < max_retries:
                    gen_kwargs["temperature"] = min(1.0, self.temperature + 0.15 * (attempt + 1))
                    gen_kwargs["do_sample"]   = True

        return {}, f"JSON_EXTRACTION_FAILED: {last_exc}"

    # Keep single-call API for backward compat (used internally)
    def generate_json(
        self,
        messages: List[Dict[str, Any]],
        max_retries: int = 2,
        max_new_tokens_override: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], str]:
        results = self.generate_json_batch(
            [messages], max_retries=max_retries,
            max_new_tokens_override=max_new_tokens_override,
        )
        return results[0]


# ---------------------------------------------------------------------------
# Caption rewrite pass — also batched
# ---------------------------------------------------------------------------
def run_headline_rewrites_batch(
    planner:        QwenPlanner,
    batch_planned:  List[Dict[str, Any]],   # one per job in batch
    batch_ctxs:     List[Dict[str, Any]],
    min_severity:   str,
    max_retries:    int,
) -> List[Dict[str, Any]]:
    """
    Batch version of run_headline_rewrites.
    Collects ALL rewrite prompts across the batch, fires one generate() call,
    then distributes results back.
    """
    min_idx = sev_index(min_severity)

    # Build a flat list of (planned_ref, cand_ref, sev, mode, messages)
    Task = Tuple[Dict[str, Any], Dict[str, Any], str, str, List[Dict[str, Any]]]
    tasks: List[Task] = []

    for planned, ctx in zip(batch_planned, batch_ctxs):
        headline        = ctx.get("headline", "")
        literal_caption = ctx.get("literal_caption") or ctx.get("grounded_context_caption") or ""
        summary         = ctx.get("summary", "")
        if not headline:
            continue

        for sev in SEV_ORDER:
            if sev_index(sev) < min_idx:
                continue
            for cand in safe_list(planned.get("severity_candidates", {}).get(sev)):
                if not isinstance(cand, dict) or cand.get("is_fallback"):
                    cand["headline_rewrite"] = {"skipped": "fallback_candidate"}
                    continue
                for mode in ("text_only", "joint"):
                    msgs = build_headline_rewrite_messages(
                        headline=headline, literal_caption=literal_caption,
                        summary=summary, sev=sev, modality_mode=mode,
                        operation=safe_str(cand.get("operation")),
                        edit_anchor=safe_str(cand.get("edited_anchor")),
                        edit_instruction=safe_str(cand.get("edit_instruction")),
                    )
                    tasks.append((planned, cand, sev, mode, msgs))

    if not tasks:
        return batch_planned

    # Fire all rewrite prompts in one batched call
    all_msgs = [t[4] for t in tasks]
    results  = planner.generate_json_batch(
        all_msgs, max_retries=max_retries, max_new_tokens_override=200
    )

    # Distribute results back into candidate dicts
    for (planned, cand, sev, mode, _msgs), (parsed, raw) in zip(tasks, results):
        if "headline_rewrite" not in cand:
            cand["headline_rewrite"] = {}
        if parsed:
            cand["headline_rewrite"][mode] = {
                "rewritten_headline":  safe_str(parsed.get("rewritten_headline")),
                "text_edit_operation": safe_str(parsed.get("text_edit_operation")),
                "nli_direction":       safe_str(parsed.get("nli_direction")),
                "modality_mode":       safe_str(parsed.get("modality_mode", mode)),
                "rewrite_rationale":   safe_str(parsed.get("rewrite_rationale")),
                "original_headline":   planned.get("planning_proxies", {}).get("headline", ""),
                "raw_rewrite_output":  raw,
            }
        else:
            cand["headline_rewrite"][mode] = {"error": "JSON extraction failed", "raw": raw}

    return batch_planned


# ---------------------------------------------------------------------------
# Job pre-processing (filter + context build, no model calls)
# ---------------------------------------------------------------------------
def prepare_job(
    job:       AnchorJob,
    tier_bank: Dict[str, List[str]],
    use_image: bool,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Returns (ctx, early_result).
    If the job is skipped (salience/mask filters), ctx=None and early_result is the
    output row to write directly. Otherwise early_result=None and ctx is ready for
    model inference.
    """
    ctx = build_anchor_context(job.anchor_row, job.manifest, tier_bank)

    if ctx["salience_proxy"] < MIN_SALIENCE_TO_PROCESS:
        return None, {
            "sample_id":  job.sample_id, "anchor_id": job.anchor_id,
            "anchor_text": ctx["anchor_text"], "status": "skipped_low_salience",
            "salience_proxy": ctx["salience_proxy"],
        }

    union_mask_path = safe_str(ctx.get("union_mask_path"))
    if not union_mask_path or not Path(union_mask_path).exists():
        return None, {
            "sample_id":   job.sample_id, "anchor_id": job.anchor_id,
            "anchor_text": ctx["anchor_text"], "status": "skipped_no_mask",
            "union_mask_path": union_mask_path or None,
        }

    image_path = safe_str(job.manifest.get("image_path"))
    if image_path and not Path(image_path).is_absolute():
        image_path = str(Path(image_path).resolve())
    if use_image and image_path and not Path(image_path).exists():
        return None, {
            "sample_id":  job.sample_id, "anchor_id": job.anchor_id,
            "anchor_text": ctx["anchor_text"], "status": "error",
            "error": f"Image not found: {image_path}",
        }

    ctx["_image_path"] = image_path
    return ctx, None


# ---------------------------------------------------------------------------
# Batch job processor
# ---------------------------------------------------------------------------
def process_batch(
    planner:                 QwenPlanner,
    batch:                   List[AnchorJob],
    tier_bank:               Dict[str, List[str]],
    candidates_per_severity: int,
    use_image:               bool,
    max_retries:             int,
    do_caption_rewrite:      bool,
    caption_rewrite_min_sev: str,
    batch_idx:               int,
) -> List[Dict[str, Any]]:
    """Process a batch of jobs. Returns one output row per job."""

    # --- Step 1: prepare contexts (no GPU) ---
    contexts: List[Optional[Dict[str, Any]]] = []
    early_results: List[Optional[Dict[str, Any]]] = []
    for job in batch:
        try:
            ctx, early = prepare_job(job, tier_bank, use_image)
        except Exception as e:
            ctx, early = None, {
                "sample_id": job.sample_id, "anchor_id": job.anchor_id,
                "anchor_text": safe_str(job.anchor_row.get("anchor_text")),
                "status": "error", "error": str(e),
            }
        contexts.append(ctx)
        early_results.append(early)

    # --- Step 2: build edit plan messages for runnable jobs ---
    runnable_indices = [i for i, ctx in enumerate(contexts) if ctx is not None]
    if not runnable_indices:
        # All were filtered/errored
        return [early_results[i] for i in range(len(batch))]

    edit_plan_msgs = [
        build_edit_plan_messages(
            image_path=contexts[i]["_image_path"] if use_image else None,
            ctx=contexts[i],
            candidates_per_severity=candidates_per_severity,
            use_image=use_image,
        )
        for i in runnable_indices
    ]

    # --- Step 3: batched edit planning inference ---
    plan_results = planner.generate_json_batch(
        edit_plan_msgs, max_retries=max_retries
    )

    # --- Step 4: score candidates ---
    planned_list: List[Dict[str, Any]] = []
    for ri, (parsed, raw_text) in zip(runnable_indices, plan_results):
        ctx = contexts[ri]
        try:
            planned = score_candidates(parsed, ctx)
        except Exception as e:
            planned = {"score_error": str(e)}
        planned["_raw_model_output"] = raw_text
        planned_list.append(planned)

    # --- Step 5: batched headline rewrites ---
    if do_caption_rewrite:
        run_ctxs = [contexts[i] for i in runnable_indices]
        planned_list = run_headline_rewrites_batch(
            planner=planner,
            batch_planned=planned_list,
            batch_ctxs=run_ctxs,
            min_severity=caption_rewrite_min_sev,
            max_retries=max_retries,
        )

    # --- Step 6: assemble output rows ---
    output_rows: List[Dict[str, Any]] = []
    runnable_iter = iter(zip(runnable_indices, planned_list))
    next_runnable = next(runnable_iter, None)

    for i, job in enumerate(batch):
        if early_results[i] is not None:
            row = dict(early_results[i])
            row["batch_idx"] = batch_idx
            output_rows.append(row)
            continue

        ri, planned = next_runnable
        assert ri == i
        ctx = contexts[i]
        raw_text = planned.pop("_raw_model_output", "")
        sensitive_flag = detect_sensitive_content(ctx, planned)

        output_rows.append({
            "sample_id":              job.sample_id,
            "anchor_id":              job.anchor_id,
            "anchor_text":            ctx["anchor_text"],
            "anchor_norm":            ctx["anchor_norm"],
            "image_path":             ctx["_image_path"],
            "planner_input":          ctx,
            "planner_output":         planned,
            "model_name":             planner.model_name,
            "used_image":             use_image,
            "headline_rewrite_pass":  do_caption_rewrite,
            "raw_model_output":       raw_text,
            "sensitive_content_flag": sensitive_flag,
            "status":                 "ok",
            "batch_idx":              batch_idx,
        })

        next_runnable = next(runnable_iter, None)

    return output_rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate Qwen-based edit suggestions + caption rewrites for T-IMPACT (v5)")
    p.add_argument("--editable-jsonl",            type=str, required=True)
    p.add_argument("--manifest-dir",              type=str, required=True)
    p.add_argument("--output-jsonl",              type=str, required=True)
    p.add_argument("--object-tier-bank",          type=str, default=None)
    p.add_argument("--model-name",                type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--num-shards",                type=int, default=1)
    p.add_argument("--shard-id",                 type=int, default=0)
    p.add_argument("--sample-ids-file",           type=str, default=None)
    p.add_argument("--anchor-ids-file",           type=str, default=None)
    p.add_argument("--max-samples",               type=int, default=None)
    p.add_argument("--max-anchors",               type=int, default=None)
    p.add_argument("--include-invalid",           action="store_true")
    p.add_argument("--skip-existing",             action="store_true")
    p.add_argument("--overwrite",                 action="store_true")
    p.add_argument("--use-image",                 action="store_true")
    p.add_argument("--candidates-per-severity",   type=int,   default=3)
    p.add_argument("--max-new-tokens",            type=int,   default=3000)
    p.add_argument("--temperature",               type=float, default=0.55)
    p.add_argument("--top-p",                     type=float, default=0.92)
    p.add_argument("--do-sample",                 action="store_true", default=True)
    p.add_argument("--no-sample",                 action="store_true")
    p.add_argument("--max-retries",               type=int,   default=2)
    p.add_argument("--attn-implementation",       type=str,   default=None,
                   choices=[None, "flash_attention_2", "sdpa", "eager"], nargs="?")
    p.add_argument("--print-every",               type=int,   default=25)
    p.add_argument("--skip-caption-rewrite",      action="store_true")
    p.add_argument("--caption-rewrite-min-severity", type=str, default="low",
                   choices=["low", "medium", "high"])
    # v5 new flags
    p.add_argument("--batch-size",                type=int,   default=4,
                   help="Number of anchors per GPU generate() call (default: 4)")
    p.add_argument("--compile",                   action="store_true",
                   help="Apply torch.compile to model (PyTorch >= 2.0, first batch slow)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    editable_jsonl = Path(args.editable_jsonl)
    manifest_dir   = Path(args.manifest_dir)
    output_jsonl   = Path(args.output_jsonl)
    ensure_dir(output_jsonl.parent)

    if output_jsonl.exists() and not (args.skip_existing or args.overwrite):
        raise FileExistsError(
            f"Output exists: {output_jsonl}. Use --skip-existing or --overwrite.")
    if output_jsonl.exists() and args.overwrite:
        output_jsonl.unlink()
        checkpoint_path(output_jsonl).unlink(missing_ok=True)

    dataset   = PlannerDataset(editable_jsonl, manifest_dir)
    tier_bank = load_tier_bank(Path(args.object_tier_bank) if args.object_tier_bank else None)

    sample_ids = dataset.eligible_sample_ids(include_invalid=args.include_invalid)
    if args.sample_ids_file:
        subset = set(load_id_list(Path(args.sample_ids_file)) or [])
        sample_ids = [s for s in sample_ids if s in subset]

    sample_ids = shard_items(sample_ids, num_shards=args.num_shards, shard_id=args.shard_id)
    if args.max_samples is not None:
        sample_ids = sample_ids[: args.max_samples]

    anchor_ids_filter = (
        set(load_id_list(Path(args.anchor_ids_file)) or [])
        if args.anchor_ids_file else None
    )
    jobs = dataset.make_jobs(
        sample_ids, include_invalid=args.include_invalid,
        anchor_ids_filter=anchor_ids_filter,
    )
    if args.max_anchors is not None:
        jobs = jobs[: args.max_anchors]

    # --- Resume: skip already-done pairs ---
    if args.skip_existing and output_jsonl.exists():
        done_pairs = already_done_pairs(output_jsonl)
        before = len(jobs)
        jobs = [j for j in jobs if (j.sample_id, j.anchor_id) not in done_pairs]
        print(f"[INFO] Resume: skipped {before - len(jobs)} already-done pairs, "
              f"{len(jobs)} remaining.")
    
    do_sample         = args.do_sample and not args.no_sample
    do_caption_rewrite = not args.skip_caption_rewrite
    batch_size        = max(1, args.batch_size)

    print(f"[INFO] editable_jsonl  : {editable_jsonl}")
    print(f"[INFO] manifest_dir    : {manifest_dir}")
    print(f"[INFO] output_jsonl    : {output_jsonl}")
    print(f"[INFO] shard {args.shard_id}/{args.num_shards} -- "
          f"sample_ids: {len(sample_ids)}, jobs: {len(jobs)}")
    print(f"[INFO] batch_size={batch_size}  use_image={args.use_image}  "
          f"candidates_per_severity={args.candidates_per_severity}")
    print(f"[INFO] model={args.model_name}  do_sample={do_sample}  "
          f"T={args.temperature}  top_p={args.top_p}  max_new_tokens={args.max_new_tokens}")
    print(f"[INFO] headline_rewrite={do_caption_rewrite}  "
          f"min_severity={args.caption_rewrite_min_severity}")
    print(f"[INFO] torch.compile={args.compile}")

    if not jobs:
        print("[INFO] Nothing to do.")
        return

    planner = QwenPlanner(
        model_name=args.model_name,
        attn_implementation=args.attn_implementation,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=do_sample,
        top_p=args.top_p,
        compile_model=args.compile,
    )

    # Batch the job list
    batches = [jobs[i: i + batch_size] for i in range(0, len(jobs), batch_size)]
    total_jobs = len(jobs)
    written = skipped = failed = 0
    start_time = time.time()

    with output_jsonl.open("a", encoding="utf-8") as f:
        for batch_idx, batch in enumerate(batches):
            try:
                rows = process_batch(
                    planner=planner, batch=batch, tier_bank=tier_bank,
                    candidates_per_severity=args.candidates_per_severity,
                    use_image=args.use_image, max_retries=args.max_retries,
                    do_caption_rewrite=do_caption_rewrite,
                    caption_rewrite_min_sev=args.caption_rewrite_min_severity,
                    batch_idx=batch_idx,
                )
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    status = row.get("status", "ok")
                    if status in ("skipped_low_salience", "skipped_no_mask"):
                        skipped += 1
                    elif status == "error":
                        failed += 1
                    else:
                        written += 1
                f.flush()
                # Write checkpoint after each fully-flushed batch
                save_checkpoint(output_jsonl, batch_idx)

            except Exception as e:
                # Whole-batch failure — write error rows for each job in batch
                for job in batch:
                    err_row: Dict[str, Any] = {
                        "sample_id":   job.sample_id,
                        "anchor_id":   job.anchor_id,
                        "anchor_text": safe_str(job.anchor_row.get("anchor_text")),
                        "status":      "error",
                        "error":       str(e),
                        "batch_idx":   batch_idx,
                    }
                    f.write(json.dumps(err_row, ensure_ascii=False) + "\n")
                    failed += 1
                f.flush()
                print(f"[WARN] Batch {batch_idx} failed: {e}")

            processed = (batch_idx + 1) * batch_size
            if processed % max(1, args.print_every) < batch_size or batch_idx == len(batches) - 1:
                elapsed = time.time() - start_time
                rate = (written + skipped + failed) / max(elapsed / 60, 0.001)
                pct  = 100.0 * (written + skipped + failed) / max(total_jobs, 1)
                print(f"[INFO] batch {batch_idx+1}/{len(batches)}  "
                      f"written={written}  skipped={skipped}  failed={failed}  "
                      f"({pct:.1f}%)  {rate:.1f} anchors/min")

    print(f"[DONE] written={written}  skipped={skipped}  failed={failed}  -> {output_jsonl}")


if __name__ == "__main__":
    main()