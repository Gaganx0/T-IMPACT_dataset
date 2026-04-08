#!/usr/bin/env python3
"""
run_image_edits.py

Execute image manipulations for T-IMPACT using edit plans produced by
generate_edit_suggestions_qwen_v2.py.

Updated routing policy:
  - remove              → LaMa only for safe, recoverable regions
  - attribute_change    → SDXL local in-place edit
  - replace             → SDXL local in-place edit only when structurally safe
  - replace (unsafe)    → route to attribute_change or skip
  - insert / fill       → SDXL local insertion

Key fixes applied over original:
  - [BUG FIX] LaMa API: correct batch format — full image (not pre-masked),
    mask binarised to 0/1, padded to mod=8, output unpadded to original size
  - [BUG FIX] LaMa API: model.freeze() retained (correct for Lightning modules);
    guard added so it is only called when model is a Lightning module
  - [BUG FIX] SDXL: default model switched to the inpainting-specific checkpoint
    (diffusers/stable-diffusion-xl-1.0-inpainting-0.1)
  - [BUG FIX] SDXL: resolution rounding changed from ×8 to ×64 (VAE requirement)
  - [BUG FIX] SDXL: seed forwarded from _do_sdxl → SDXLInpainter.inpaint
  - [BUG FIX] _do_sdxl_local: crop result size verified and matched before paste
  - [BUG FIX] torch.load: weights_only=False added for legacy checkpoint compat
  - [BUG FIX] _replace_strength_and_pad: meta parameter now used to lower
    strength and increase pad for person-attached regions
  - [BUG FIX] bbox convention: closed [x1,y1,x2,y2] normalised consistently
  - [BUG FIX] --overwrite and --skip-existing are now mutually exclusive
  - [BUG FIX] output filenames now include both requested and effective
    operations so rerouting is visible on disk
  - [BUG FIX] routing thresholds relaxed for scene-level replace/remove
    so broad but plausible edits are not skipped as aggressively
  - [BUG FIX] LaMa subprocess now tries Hydra-safe command variants and remove
    jobs can fall back to SDXL fill instead of failing hard
  - [BUG FIX] SDXL prompts shortened to reduce CLIP truncation
  - [BUG FIX] colour correction made more conservative to reduce halos/tint shifts

  Quality / stylistic consistency improvements:
  - SDXL now uses padding_mask_crop via the built-in pipeline feature rather
    than a custom manual crop, eliminating the paste-size mismatch entirely;
    the _do_sdxl_local crop path is retained only as a fallback for very small
    crops where padding_mask_crop is not beneficial
  - Colour/tone matching: after SDXL inpainting, a LAB-space histogram-matched
    colour correction is applied inside the masked region so the edit adopts
    the surrounding image's tonal palette
  - Soft feathering applied consistently to the paste mask using a distance
    transform rather than a flat Gaussian, giving a physically correct gradient
  - Lighting normalisation: the mean luminance of the masked region in the
    output is nudged toward the mean luminance of a border ring around the mask
  - ban whole-person replace
  - hard-cap generic replace area ratio
  - disallow replace on strip-like / border-touching / sprawling masks
  - disallow remove when it would reveal hidden anatomy/clothing structure
  - remove LaMa->SDXL chained large replace for unstable cases
  - stronger preserve-identity / preserve-geometry prompting
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import tempfile
import time
import traceback
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger("run_image_edits")

# ---------------------------------------------------------------------------
# Optional heavy imports
# ---------------------------------------------------------------------------
try:
    import torch
except ImportError as e:
    raise ImportError("PyTorch is required: pip install torch") from e

try:
    from diffusers import StableDiffusionXLInpaintPipeline
except ImportError as e:
    raise ImportError("diffusers is required: pip install diffusers") from e

try:
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF
except ImportError as e:
    raise ImportError("torchvision is required: pip install torchvision") from e

try:
    from saicinpainting.training.trainers import load_checkpoint  # type: ignore
    from saicinpainting.evaluation.utils import move_to_device    # type: ignore
    HAS_LAMA_API = True
except Exception:
    HAS_LAMA_API = False
    logger.warning(
        "[WARN] LaMa Python API not found — will call lama/bin/predict.py via subprocess"
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Hard cap for generic direct replace.
LARGE_REPLACE_AREA_RATIO = 0.28

# Stricter cap for person-like / body-attached replacements.
PERSON_REPLACE_AREA_RATIO = 0.14

# Remove is only safe when the model can plausibly recover the region from context.
MAX_REMOVE_AREA_RATIO = 0.26
MAX_BODY_ATTACHED_REMOVE_AREA_RATIO = 0.07

# Additional relaxed caps for broad scene-level edits.
SCENE_REPLACE_AREA_RATIO = 0.40
SCENE_REMOVE_AREA_RATIO = 0.30

SDXL_GUIDANCE_SCALE = 8.0
SDXL_GUIDANCE_SCALE_INSERTION = 12.0
SDXL_NUM_INFERENCE_STEPS = 42
SDXL_NUM_INFERENCE_STEPS_INSERTION = 52

SDXL_STRENGTH_BY_OP: Dict[str, float] = {
    "attribute_change":      0.42,
    "attribute_change_face": 0.24,
    "replace_small":         0.46,
    "replace_medium":        0.54,
    "insert":                0.84,
    "fill":                  0.82,
    "default":               0.58,
}
# SDXL requires multiples of 64 for stable VAE encoding (NOT 8).
SDXL_RESOLUTION_MULTIPLE = 64

MASK_DILATION_PX = 6
MASK_DILATION_FACE_PX = 2

# Use erosion for replace so edit stays inside the object.
REPLACE_MASK_EROSION_PX = 3

DEFAULT_REALISM_THRESHOLD = 0.45
MAX_CANDIDATES_PER_SEVERITY = 2

# Conservative fallback behaviour when LaMa fails on remove jobs.
REMOVE_FALLBACK_TO_FILL = True
REMOVE_FALLBACK_MAX_AREA_RATIO = 0.28
PROMPT_MAX_CHARS = 360

MIN_MASKED_CHANGE_RATIO: Dict[str, float] = {
    "low": 0.010,
    "medium": 0.018,
    "high": 0.030,
}
MIN_MASKED_MEAN_DELTA: Dict[str, float] = {
    "low": 3.0,
    "medium": 5.0,
    "high": 8.0,
}
MAX_SDXL_ATTEMPTS = 3

# Padding (pixels) around the mask bbox used by padding_mask_crop pipeline feature.
# The pipeline will auto-crop and upscale the masked region to 1024 for quality.
SDXL_PADDING_MASK_CROP = 32

FACE_ANCHOR_KEYWORDS: FrozenSet[str] = frozenset([
    "face", "expression", "eye", "eyes", "eyebrow", "eyebrows",
    "mouth", "smile", "frown", "nose", "chin", "forehead", "skin",
    "complexion", "beard", "moustache", "mustache", "hair colour",
    "hair color", "lip", "lips",
])

PERSON_KEYWORDS: FrozenSet[str] = frozenset([
    "person", "man", "woman", "boy", "girl", "human", "subject", "individual",
    "people", "adult", "child", "pedestrian", "worker", "soldier", "police",
    "officer", "officer uniform", "face", "head", "torso", "body", "arm", "hand",
    "leg", "foot", "helmet", "cap", "hat", "shirt", "jacket", "vest", "uniform",
])

BODY_ATTACHED_KEYWORDS: FrozenSet[str] = frozenset([
    "shirt", "jacket", "vest", "uniform", "cap", "hat", "helmet", "badge", "patch",
    "strap", "backpack", "bag", "watch", "glove", "glasses", "belt", "radio",
    "bodycam", "device", "shoulder", "chest", "torso", "arm", "hand",
])

IDENTITY_SENSITIVE_KEYWORDS: FrozenSet[str] = frozenset([
    "face", "head", "hair", "eye", "eyes", "nose", "mouth", "beard", "mustache",
    "moustache", "skin", "expression", "forehead", "chin", "ear",
])

BACKGROUND_SAFE_REMOVE_KEYWORDS: FrozenSet[str] = frozenset([
    "logo", "sign", "poster", "graffiti", "sticker", "text", "marking", "label",
    "symbol", "wall object", "small object", "cone", "bin", "box",
])

VAGUE_PERSON_SWAP_PREFIXES = (
    "man ", "woman ", "person ", "police", "officer", "soldier", "figure",
    "individual", "military", "guard",
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class EditJob:
    sample_id: str
    anchor_id: str
    image_path: Path
    union_mask_path: Optional[Path]
    bbox_xyxy: Optional[List[float]]
    operation: str
    severity: str
    candidate_index: int
    edit_instruction: str
    visual_prompt: str
    fill_clause: str
    edited_anchor: str
    tier: str
    planning_score: float
    rewritten_caption: Optional[str]
    nli_direction: Optional[str]
    text_edit_operation: Optional[str]


@dataclass
class EditResult:
    job: EditJob
    output_path: Optional[Path] = None
    realism_score: Optional[float] = None
    accepted: bool = False
    reject_reason: Optional[str] = None
    error: Optional[str] = None
    duration_s: float = 0.0
    lama_intermediate_path: Optional[Path] = None
    effective_operation: Optional[str] = None
    route_reason: Optional[str] = None
    masked_mean_delta: Optional[float] = None
    masked_changed_ratio: Optional[float] = None


@dataclass
class MaskStats:
    area_ratio: float
    bbox: Optional[Tuple[int, int, int, int]]   # half-open: x1,y1,x2,y2
    bbox_area_ratio: float
    bbox_aspect_ratio: float
    touches_border_count: int
    compactness: float
    is_empty: bool


@dataclass
class AnchorMeta:
    is_face_edit: bool
    is_person_like: bool
    is_whole_person: bool
    is_body_attached: bool
    is_identity_sensitive: bool
    is_background_safe_remove: bool
    implies_role_swap: bool
    reveals_hidden_structure_when_removed: bool


@dataclass
class RouteDecision:
    effective_operation: str
    route_reason: str


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_anchors_index(
    anchors_jsonl: Path,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    idx: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in load_jsonl(anchors_jsonl):
        key = (row.get("sample_id", ""), row.get("anchor_id", ""))
        idx[key] = row
    return idx


def bbox_to_mask(
    bbox: List[float], w: int, h: int, dilation: int = 0
) -> np.ndarray:
    """
    bbox is closed [x1, y1, x2, y2] in pixel coordinates.
    Converts to a uint8 mask of shape (h, w) with 255 inside the box.
    """
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    # Clamp and apply dilation
    x1 = max(0, x1 - dilation)
    y1 = max(0, y1 - dilation)
    x2 = min(w, x2 + dilation)
    y2 = min(h, y2 + dilation)
    mask = np.zeros((h, w), dtype=np.uint8)
    # bbox is closed so +1 on slice end converts to half-open array indexing
    mask[y1:y2 + 1, x1:x2 + 1] = 255
    return mask


def load_mask(
    mask_path: Optional[Path],
    bbox: Optional[List[float]],
    img_w: int,
    img_h: int,
    dilation: int = MASK_DILATION_PX,
) -> np.ndarray:
    if mask_path and mask_path.exists():
        arr = np.array(Image.open(mask_path).convert("L"))
        arr = (arr > 0).astype(np.uint8) * 255
        if dilation > 0:
            from scipy.ndimage import binary_dilation
            struct = np.ones((dilation * 2 + 1, dilation * 2 + 1), dtype=bool)
            arr = (binary_dilation(arr > 0, structure=struct).astype(np.uint8) * 255)
        return arr

    if bbox:
        return bbox_to_mask(bbox, img_w, img_h, dilation=dilation)

    logger.warning("No mask and no bbox — returning empty mask")
    return np.zeros((img_h, img_w), dtype=np.uint8)


def erode_mask(mask: np.ndarray, erosion_px: int) -> np.ndarray:
    if erosion_px <= 0:
        return mask
    from scipy.ndimage import binary_erosion
    structure = np.ones((erosion_px * 2 + 1, erosion_px * 2 + 1), dtype=bool)
    eroded = binary_erosion(mask > 0, structure=structure)
    out = (eroded.astype(np.uint8) * 255)
    if out.sum() == 0:
        return mask
    return out


def mask_area_ratio(mask: np.ndarray) -> float:
    h, w = mask.shape[:2]
    if h == 0 or w == 0:
        return 0.0
    return float((mask > 0).sum()) / float(h * w)


def _contains_any(text: str, keywords: FrozenSet[str]) -> bool:
    t = f" {text.lower().strip()} "
    return any(f" {kw} " in t for kw in keywords)


def _is_face_edit(edited_anchor: str) -> bool:
    return _contains_any(edited_anchor, FACE_ANCHOR_KEYWORDS)


def _parse_edit_instruction(edit_instruction: str) -> Tuple[str, str]:
    fill_match = re.search(r"\[FILL\](.*?)(?:\[VISUAL\]|$)", edit_instruction, re.DOTALL)
    visual_match = re.search(r"\[VISUAL\](.*?)$", edit_instruction, re.DOTALL)
    fill_clause = (
        fill_match.group(1).strip().rstrip(";").strip() if fill_match else edit_instruction
    )
    visual_prompt = visual_match.group(1).strip() if visual_match else ""
    return fill_clause, visual_prompt


def _normalize_text(*parts: Optional[str]) -> str:
    return " ".join(p for p in parts if p).lower().strip()


def already_done_keys(output_jsonl: Path) -> set:
    done: set = set()
    if not output_jsonl.exists():
        return done
    for row in load_jsonl(output_jsonl):
        key = (
            row.get("sample_id"),
            row.get("anchor_id"),
            row.get("requested_operation"),
            row.get("severity"),
            row.get("candidate_index"),
        )
        done.add(key)
    return done


def compute_mask_stats(mask: np.ndarray) -> MaskStats:
    h, w = mask.shape[:2]
    fg = mask > 0
    fg_count = int(fg.sum())
    area_ratio = float(fg_count) / float(max(1, h * w))

    if fg_count == 0:
        return MaskStats(
            area_ratio=0.0,
            bbox=None,
            bbox_area_ratio=0.0,
            bbox_aspect_ratio=1.0,
            touches_border_count=0,
            compactness=0.0,
            is_empty=True,
        )

    ys, xs = np.where(fg)
    # Store as closed bbox (inclusive max pixel)
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()), int(ys.max())   # closed: last included pixel
    bw = max(1, x2 - x1 + 1)
    bh = max(1, y2 - y1 + 1)
    bbox_area = bw * bh
    bbox_area_ratio = float(bbox_area) / float(max(1, h * w))
    aspect = float(bw) / float(max(1, bh))

    touches = 0
    if x1 <= 0:
        touches += 1
    if y1 <= 0:
        touches += 1
    if x2 >= w - 1:
        touches += 1
    if y2 >= h - 1:
        touches += 1

    compactness = float(fg_count) / float(max(1, bbox_area))

    return MaskStats(
        area_ratio=area_ratio,
        bbox=(x1, y1, x2, y2),   # closed bbox
        bbox_area_ratio=bbox_area_ratio,
        bbox_aspect_ratio=aspect,
        touches_border_count=touches,
        compactness=compactness,
        is_empty=False,
    )


def infer_anchor_meta(job: EditJob) -> AnchorMeta:
    text = _normalize_text(
        job.edited_anchor, job.fill_clause, job.visual_prompt, job.edit_instruction
    )
    edited_anchor = (job.edited_anchor or "").lower().strip()
    fill_lower = (job.fill_clause or "").lower().strip()

    is_face_edit = _is_face_edit(job.edited_anchor)
    is_person_like = _contains_any(text, PERSON_KEYWORDS)
    is_body_attached = _contains_any(text, BODY_ATTACHED_KEYWORDS)
    is_identity_sensitive = is_face_edit or _contains_any(text, IDENTITY_SENSITIVE_KEYWORDS)
    is_background_safe_remove = _contains_any(text, BACKGROUND_SAFE_REMOVE_KEYWORDS)

    is_whole_person = False
    if is_person_like:
        if any(
            tok in edited_anchor
            for tok in ["person", "man", "woman", "officer", "soldier", "individual", "subject"]
        ):
            is_whole_person = True
        if edited_anchor in {
            "person", "man", "woman", "officer", "soldier", "individual", "subject"
        }:
            is_whole_person = True

    implies_role_swap = any(fill_lower.startswith(p) for p in VAGUE_PERSON_SWAP_PREFIXES)
    reveals_hidden_structure_when_removed = is_person_like and (
        is_body_attached
        or is_identity_sensitive
        or ("vest" in text)
        or ("shirt" in text)
        or ("uniform" in text)
    )

    return AnchorMeta(
        is_face_edit=is_face_edit,
        is_person_like=is_person_like,
        is_whole_person=is_whole_person,
        is_body_attached=is_body_attached,
        is_identity_sensitive=is_identity_sensitive,
        is_background_safe_remove=is_background_safe_remove,
        implies_role_swap=implies_role_swap,
        reveals_hidden_structure_when_removed=reveals_hidden_structure_when_removed,
    )



def _compact_prompt_text(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip())
    s = re.sub(r"\s+,", ",", s)
    return s


def _join_prompt_parts(*parts: Optional[str], max_chars: int = PROMPT_MAX_CHARS) -> str:
    items = [_compact_prompt_text(p) for p in parts if p and _compact_prompt_text(p)]
    out = ", ".join(items)
    if len(out) <= max_chars:
        return out
    shortened = []
    cur = 0
    for item in items:
        add = len(item) + (2 if shortened else 0)
        if cur + add > max_chars:
            break
        shortened.append(item)
        cur += add
    return ", ".join(shortened) if shortened else out[:max_chars].rstrip(", ")


def _build_remove_fill_prompt(edited_anchor: str, severity: str) -> Tuple[str, str]:
    severity_hint = {
        "low": "subtle but visible",
        "medium": "clear localized",
        "high": "strong localized",
    }.get(severity, "localized")
    anchor_text = _compact_prompt_text(edited_anchor or "target object")
    positive = _join_prompt_parts(
        f"remove only the masked {anchor_text}",
        f"{severity_hint} cleanup",
        "fill with matching surrounding background",
        "keep scene geometry lighting and camera unchanged",
        "photorealistic",
    )
    negative = _join_prompt_parts(
        "extra object",
        "ghost outline",
        "duplicate subject",
        "obvious seam",
        "halo",
        "new viewpoint",
        "cartoon",
        "text or watermark",
    )
    return positive, negative


def build_sdxl_prompt(
    fill_clause: str,
    visual_prompt: str,
    edited_anchor: str,
    operation: str,
    meta: AnchorMeta,
    severity: str,
) -> Tuple[str, str]:
    fill_clause = _compact_prompt_text(fill_clause)
    visual_prompt = _compact_prompt_text(visual_prompt)
    anchor_text = _compact_prompt_text(edited_anchor)

    severity_hint = {
        "low": "subtle but visible",
        "medium": "clear localized",
        "high": "strong obvious",
    }.get(severity, "localized")

    keep_scene = "keep unmasked scene camera layout lighting and scale unchanged"
    keep_person = "keep same identity pose body and clothing outside mask"
    visible = "change must be visible to the naked eye"

    neg_common = _join_prompt_parts(
        "extra object or person",
        "duplicate subject",
        "wrong perspective or scale",
        "weak imperceptible edit",
        "halo seam blur",
        "cartoon illustration",
        "text watermark logo unless requested",
    )

    if meta.is_face_edit:
        positive = _join_prompt_parts(
            f"edit only masked region of {anchor_text}",
            severity_hint,
            fill_clause,
            visual_prompt,
            visible,
            "photorealistic",
            keep_scene,
            keep_person,
        )
        negative = _join_prompt_parts(
            "different person",
            "identity drift",
            "uncanny face",
            "plastic skin",
            neg_common,
        )
        return positive, negative

    if operation == "attribute_change":
        positive = _join_prompt_parts(
            f"edit only masked region of {anchor_text}",
            severity_hint,
            fill_clause,
            visual_prompt,
            visible,
            "photorealistic local attribute change",
            keep_scene,
            keep_person if (meta.is_person_like or meta.is_body_attached) else None,
        )
        negative = _join_prompt_parts(
            "whole-subject replacement",
            "role change",
            "new accessory unless requested",
            neg_common,
        )
        return positive, negative

    if operation == "replace":
        positive = _join_prompt_parts(
            f"replace only masked region of {anchor_text}",
            severity_hint,
            fill_clause,
            visual_prompt,
            visible,
            "photorealistic natural integration",
            keep_scene,
            keep_person if (meta.is_person_like or meta.is_body_attached) else None,
        )
        negative = _join_prompt_parts(
            "whole-person replacement",
            "identity drift" if (meta.is_person_like or meta.is_body_attached) else None,
            "role swap" if (meta.is_person_like or meta.is_body_attached) else None,
            neg_common,
        )
        return positive, negative

    if operation in ("fill", "insert"):
        positive = _join_prompt_parts(
            severity_hint,
            fill_clause,
            visual_prompt,
            "photorealistic natural integration",
            keep_scene,
        )
        return positive, neg_common

    positive = _join_prompt_parts(
        severity_hint,
        fill_clause,
        visual_prompt,
        "photorealistic",
        keep_scene,
    )
    return positive, neg_common


# ---------------------------------------------------------------------------
# Colour / tonal consistency helpers
# ---------------------------------------------------------------------------

def _lab_colour_match_masked(
    edited: Image.Image,
    original: Image.Image,
    mask: np.ndarray,
    blend: float = 0.6,
) -> Image.Image:
    """
    Correct the tonal palette of the edited region so that its LAB histogram
    statistics match those of the surrounding (un-masked) context in the
    original image.  The correction is blended with `blend` weight so it is
    not forced — this preserves intentional colour changes while making the
    overall tonal feel consistent with the scene.
    """
    try:
        from skimage import color as skcolor
    except ImportError:
        return edited   # graceful no-op if skimage not installed

    orig_arr = np.array(original.convert("RGB")).astype(np.float32) / 255.0
    edit_arr = np.array(edited.convert("RGB")).astype(np.float32) / 255.0

    fg = (mask > 0)
    bg = ~fg

    if bg.sum() < 100 or fg.sum() < 10:
        return edited

    orig_lab = skcolor.rgb2lab(orig_arr)
    edit_lab = skcolor.rgb2lab(edit_arr)

    # Compute statistics over the background (unmasked context) in original.
    for ch in range(3):
        src = edit_lab[:, :, ch][fg]
        ref = orig_lab[:, :, ch][bg]

        src_mean, src_std = src.mean(), src.std() + 1e-6
        ref_mean, ref_std = ref.mean(), ref.std() + 1e-6

        corrected = (edit_lab[:, :, ch] - src_mean) * (ref_std / src_std) + ref_mean
        # Blend correction into edited LAB — preserve intentional changes
        edit_lab[:, :, ch] = np.where(
            fg,
            edit_lab[:, :, ch] * (1.0 - blend) + corrected * blend,
            edit_lab[:, :, ch],
        )

    result_rgb = (np.clip(skcolor.lab2rgb(edit_lab), 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(result_rgb)


def _luminance_match_border(
    edited: Image.Image,
    original: Image.Image,
    mask: np.ndarray,
    border_px: int = 12,
    strength: float = 0.5,
) -> Image.Image:
    """
    Nudge the mean luminance of the edited masked region toward the mean
    luminance of the ring of pixels surrounding the mask boundary in the
    original image.  This avoids the edit looking over/under-exposed
    relative to its immediate neighbourhood.
    """
    from scipy.ndimage import binary_dilation
    struct = np.ones((border_px * 2 + 1, border_px * 2 + 1), dtype=bool)
    border_mask = binary_dilation(mask > 0, structure=struct) & ~(mask > 0)

    if border_mask.sum() < 20:
        return edited

    orig_arr = np.array(original.convert("RGB")).astype(np.float32)
    edit_arr = np.array(edited.convert("RGB")).astype(np.float32)

    # Luminance via standard rec601 weights
    def lum(a: np.ndarray) -> np.ndarray:
        return 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]

    border_lum = lum(orig_arr)[border_mask].mean()
    edit_lum_in_mask = lum(edit_arr)[mask > 0].mean()

    if edit_lum_in_mask < 1.0:
        return edited

    scale = 1.0 + strength * ((border_lum / edit_lum_in_mask) - 1.0)
    scale = float(np.clip(scale, 0.6, 1.6))

    corrected = edit_arr.copy()
    corrected[mask > 0] = np.clip(edit_arr[mask > 0] * scale, 0, 255)
    return Image.fromarray(corrected.astype(np.uint8))



def _distance_feather_mask(mask: np.ndarray, radius: int) -> Image.Image:
    """
    Build a soft paste mask with a hard interior and a smooth falloff only in a
    narrow boundary band. This avoids washing out the full edit while still
    preventing visible seams at the edge of the edited region.
    """
    from scipy.ndimage import binary_erosion, distance_transform_edt

    binary = (mask > 0)
    if radius <= 0 or binary.sum() == 0:
        return Image.fromarray((binary.astype(np.uint8) * 255), mode="L")

    structure = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=bool)
    interior = binary_erosion(binary, structure=structure)
    boundary_band = binary & (~interior)

    dist_in = distance_transform_edt(binary.astype(np.uint8))
    alpha = np.zeros(binary.shape, dtype=np.float32)
    alpha[interior] = 1.0
    alpha[boundary_band] = np.clip(dist_in[boundary_band] / max(1, radius), 0.0, 1.0)

    return Image.fromarray((alpha * 255).astype(np.uint8), mode="L")


# ---------------------------------------------------------------------------
# Job extraction
# ---------------------------------------------------------------------------
def extract_jobs_from_plan_row(
    row: Dict[str, Any],
    anchor_index: Dict[Tuple[str, str], Dict[str, Any]],
    images_root: Path,
    max_candidates_per_severity: Optional[int] = None,
) -> List[EditJob]:
    if row.get("status") in ("error", "skipped_low_salience"):
        return []

    sample_id = row.get("sample_id", "")
    anchor_id = row.get("anchor_id", "")

    image_path_str = row.get("image_path", "")
    image_path = Path(image_path_str) if image_path_str else None
    if image_path and not image_path.is_absolute():
        image_path = (images_root / image_path).resolve()
    if not image_path or not image_path.exists():
        logger.warning(f"[SKIP] {sample_id}/{anchor_id}: image not found at {image_path}")
        return []

    planner_input = row.get("planner_input", {}) or {}
    anchor_row = anchor_index.get((sample_id, anchor_id), {})

    mask_path_str = planner_input.get("union_mask_path") or anchor_row.get("union_mask_path")
    union_mask_path = Path(mask_path_str) if mask_path_str else None
    bbox_xyxy = planner_input.get("bbox_xyxy") or anchor_row.get("bbox_xyxy")

    mask_file_exists = union_mask_path and union_mask_path.exists()
    if not mask_file_exists and not bbox_xyxy:
        logger.warning(
            f"[SKIP] {sample_id}/{anchor_id}: no union_mask_path and no bbox_xyxy — "
            f"cannot localise edit, skipping all candidates for this anchor"
        )
        return []

    tier = planner_input.get("tier_name", "tier_a") or "tier_a"

    planned = row.get("planner_output", {})
    severity_candidates = planned.get("severity_candidates", {})
    if not severity_candidates:
        logger.warning(
            f"[SKIP] {sample_id}/{anchor_id}: no severity_candidates in planner_output"
        )
        return []

    MIN_SCORE_BY_SEVERITY: Dict[str, float] = {"low": 0.20, "medium": 0.15, "high": 0.0}
    _max_cands = (
        max_candidates_per_severity
        if max_candidates_per_severity is not None
        else MAX_CANDIDATES_PER_SEVERITY
    )

    mask_ratio = planner_input.get("mask_ratio", 1.0)
    occlusion = planned.get("visual_grounding", {}).get("occlusion_notes", "").lower()
    anchor_occluded = any(
        w in occlusion for w in ("covered", "occluded", "hidden", "partial")
    )

    jobs: List[EditJob] = []

    for severity in ("low", "medium", "high"):
        candidates = severity_candidates.get(severity, [])
        if not isinstance(candidates, list):
            logger.warning(
                f"[SKIP] {sample_id}/{anchor_id} sev={severity}: "
                f"severity_candidates['{severity}'] is not a list"
            )
            continue

        if not candidates:
            continue

        seen_anchors: set = set()
        accepted_this_severity = 0

        for raw_idx, cand in enumerate(candidates):
            if not isinstance(cand, dict):
                continue

            if accepted_this_severity >= _max_cands:
                break

            operation = cand.get("operation", "replace")
            edit_instruction = cand.get("edit_instruction", "")
            edited_anchor = cand.get("edited_anchor", "")

            anchor_key = edited_anchor.lower().strip()
            if anchor_key in seen_anchors:
                continue
            seen_anchors.add(anchor_key)

            fill_clause, visual_prompt = _parse_edit_instruction(edit_instruction)

            derived = cand.get("derived", {}) or {}
            planning_score = float(derived.get("planned_score_raw", 0.0))

            min_score = MIN_SCORE_BY_SEVERITY.get(severity, 0.0)
            if planning_score < min_score:
                continue

            if severity == "low" and anchor_occluded and mask_ratio > 0.15:
                continue

            headline_rewrite = cand.get("headline_rewrite", {}) or {}
            rewrite_block = (
                headline_rewrite.get("joint")
                or headline_rewrite.get("text_only")
                or {}
            )
            rewritten_caption = rewrite_block.get("rewritten_headline")
            nli_direction = rewrite_block.get("nli_direction")
            text_edit_operation = rewrite_block.get("text_edit_operation")

            jobs.append(EditJob(
                sample_id=sample_id,
                anchor_id=anchor_id,
                image_path=image_path,
                union_mask_path=union_mask_path,
                bbox_xyxy=bbox_xyxy,
                operation=operation,
                severity=severity,
                candidate_index=accepted_this_severity,
                edit_instruction=edit_instruction,
                visual_prompt=visual_prompt,
                fill_clause=fill_clause,
                edited_anchor=edited_anchor,
                tier=tier,
                planning_score=planning_score,
                rewritten_caption=rewritten_caption,
                nli_direction=nli_direction,
                text_edit_operation=text_edit_operation,
            ))
            accepted_this_severity += 1

    return jobs


# ---------------------------------------------------------------------------
# LaMa inpainting
# ---------------------------------------------------------------------------
def _pad_tensor_to_modulo(x: "torch.Tensor", mod: int) -> "torch.Tensor":
    """Pad tensor height/width to be divisible by `mod` (required by LaMa)."""
    _, _, h, w = x.shape
    pad_h = (mod - h % mod) % mod
    pad_w = (mod - w % mod) % mod
    if pad_h == 0 and pad_w == 0:
        return x
    return torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")


class LamaInpainter:
    def __init__(
        self,
        config_path: Optional[str],
        checkpoint_path: Optional[str],
        device: str,
    ):
        self.device = device
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self._model = None

        self.lama_root: Optional[Path] = None
        if checkpoint_path:
            cp = Path(checkpoint_path).resolve()
            candidate = cp.parent
            for _ in range(4):
                if (candidate / "bin" / "predict.py").exists():
                    self.lama_root = candidate
                    break
                candidate = candidate.parent
            if self.lama_root is None:
                logger.warning(
                    f"[LaMa] Could not find bin/predict.py relative to {checkpoint_path}. "
                    f"Subprocess fallback will try current working directory."
                )

        if HAS_LAMA_API and config_path and checkpoint_path:
            self._load_model()

    def _load_model(self) -> None:
        import yaml
        from omegaconf import OmegaConf

        try:
            with open(self.config_path) as f:
                cfg = OmegaConf.create(yaml.safe_load(f))
            # predict_only avoids unnecessary discriminator init
            cfg.training_model.predict_only = True
            cfg.visualizer.kind = "noop"
            self._model = load_checkpoint(
                cfg, self.checkpoint_path, strict=False, map_location="cpu"
            )
            # freeze() is a PyTorch Lightning method — guard before calling
            if hasattr(self._model, "freeze"):
                self._model.freeze()
            else:
                self._model.eval()
                for p in self._model.parameters():
                    p.requires_grad_(False)
            self._model.to(self.device)
            logger.info("[LaMa] Model loaded via Python API")
        except Exception as e:
            logger.warning(f"[LaMa] Python API load failed ({e}); will use subprocess fallback")
            self._model = None

    def inpaint(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        if self._model is not None:
            return self._inpaint_api(image, mask)
        elif self.checkpoint_path:
            return self._inpaint_subprocess(image, mask)
        else:
            raise RuntimeError(
                "LaMa is not configured. Provide --lama-config and --lama-checkpoint."
            )

    def _inpaint_api(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        """
        Correct LaMa batch format (sourced from Inpaint-Anything reference impl):
          - batch['image']: float32 [0,1] tensor (B, 3, H, W) — full image, NOT masked
          - batch['mask']:  float32 {0,1} tensor (B, 1, H, W) — binarised
          - both padded to mod=8 before inference
          - output is batch["inpainted"] (B, 3, H_padded, W_padded), unpadded after
        """
        orig_h, orig_w = mask.shape[:2]

        img_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        msk_np = (mask > 0).astype(np.float32)   # 0/1 float, NOT 0/255

        # (H, W, 3) → (1, 3, H, W)
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        # (H, W) → (1, 1, H, W)
        msk_t = torch.from_numpy(msk_np).unsqueeze(0).unsqueeze(0)

        # LaMa requires dimensions divisible by 8
        img_t = _pad_tensor_to_modulo(img_t, 8)
        msk_t = _pad_tensor_to_modulo(msk_t, 8)

        batch = {
            "image": img_t.to(self.device),
            "mask":  msk_t.to(self.device),
        }

        with torch.no_grad():
            batch = self._model(batch)

        result_t = batch["inpainted"][0].cpu().clamp(0, 1)   # (3, H_pad, W_pad)
        # Unpad back to original size
        result_t = result_t[:, :orig_h, :orig_w]
        return TF.to_pil_image(result_t)

    def _inpaint_subprocess(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        import subprocess
        import sys

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image.save(tmp / "image.png")
            Image.fromarray(mask).save(tmp / "image_mask.png")

            out_dir = tmp / "out"
            out_dir.mkdir()

            if self.lama_root is not None:
                predict_script = str(self.lama_root / "bin" / "predict.py")
                cwd = str(self.lama_root)
            else:
                predict_script = "bin/predict.py"
                cwd = None

            base_cmd = [
                sys.executable,
                predict_script,
                f"model.path={Path(self.checkpoint_path).resolve()}",
                f"indir={tmp}",
                f"outdir={out_dir}",
                "dataset.img_suffix=.png",
            ]
            variants = [
                base_cmd + ["+dataset.mask_suffix=_mask.png"],
                base_cmd + ["dataset.mask_suffix=_mask.png"],
                base_cmd,
            ]

            errors = []
            for cmd in variants:
                logger.debug(f"[LaMa] subprocess cmd: {' '.join(cmd)}  cwd={cwd}")
                proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
                if proc.returncode == 0:
                    result_files = sorted(out_dir.glob("*.png"))
                    if result_files:
                        return Image.open(result_files[0]).convert("RGB")
                    errors.append(
                        f"cmd={' '.join(cmd)} produced no png output; stdout tail={proc.stdout[-500:]} stderr tail={proc.stderr[-500:]}"
                    )
                else:
                    errors.append(
                        f"cmd={' '.join(cmd)} exit={proc.returncode} stdout tail={proc.stdout[-1200:]} stderr tail={proc.stderr[-1200:]}"
                    )

            raise RuntimeError("LaMa predict.py failed for all command variants:\n" + "\n---\n".join(errors))


# ---------------------------------------------------------------------------
# SDXL inpainting
# ---------------------------------------------------------------------------
class SDXLInpainter:
    def __init__(self, model_id: str, device: str):
        self.device = device
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        logger.info(f"[SDXL] Loading {model_id}…")
        kwargs: Dict[str, Any] = {
            "torch_dtype": dtype,
            "use_safetensors": True,
        }
        if device.startswith("cuda"):
            kwargs["variant"] = "fp16"

        self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            model_id,
            **kwargs,
        ).to(device)
        self.pipe.set_progress_bar_config(disable=True)
        logger.info("[SDXL] Ready")

    def inpaint(
        self,
        image: Image.Image,
        mask: np.ndarray,
        prompt: str,
        negative_prompt: str,
        strength: float = 0.80,
        seed: Optional[int] = None,
        guidance_scale: float = SDXL_GUIDANCE_SCALE,
        num_inference_steps: int = SDXL_NUM_INFERENCE_STEPS,
        use_padding_crop: bool = True,
    ) -> Image.Image:
        """
        Inpaint using SDXL.

        When use_padding_crop=True (the default), the pipeline's built-in
        padding_mask_crop feature is used: it crops to the masked region with
        padding, upscales to 1024 internally, inpaints at full quality, then
        pastes back.  This eliminates the manual crop/paste size mismatch.

        use_padding_crop=False falls back to a direct call on the full image
        after rounding to a multiple of 64.
        """
        orig_w, orig_h = image.size

        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)

        msk_pil = Image.fromarray(mask).convert("L")

        if use_padding_crop:
            # The pipeline handles the crop/resize/paste internally.
            # Image must be PIL; output is returned at original resolution.
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                mask_image=msk_pil,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                strength=strength,
                generator=generator,
                padding_mask_crop=SDXL_PADDING_MASK_CROP,
            ).images[0]
            # pipeline returns at original size when padding_mask_crop is used
            return result

        # Fallback: resize to nearest multiple of SDXL_RESOLUTION_MULTIPLE (64)
        target_w = ((orig_w + SDXL_RESOLUTION_MULTIPLE - 1) // SDXL_RESOLUTION_MULTIPLE) * SDXL_RESOLUTION_MULTIPLE
        target_h = ((orig_h + SDXL_RESOLUTION_MULTIPLE - 1) // SDXL_RESOLUTION_MULTIPLE) * SDXL_RESOLUTION_MULTIPLE

        img_resized = image.resize((target_w, target_h), Image.LANCZOS)
        msk_resized = msk_pil.resize((target_w, target_h), Image.NEAREST)

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=img_resized,
            mask_image=msk_resized,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            strength=strength,
            generator=generator,
        ).images[0]

        # Resize back to original; LANCZOS for image, NEAREST for mask-edge crispness
        return result.resize((orig_w, orig_h), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Realism filter
# ---------------------------------------------------------------------------
class RealismFilter:
    def __init__(self, checkpoint: Optional[str], device: str):
        self.device = device
        self._model = None
        self._transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        if checkpoint and Path(checkpoint).exists():
            self._load(checkpoint)

    def _load(self, checkpoint: str) -> None:
        try:
            import timm
            self._model = timm.create_model(
                "vit_base_patch16_224", pretrained=False, num_classes=2
            )
            # weights_only=False required for arbitrary checkpoint formats
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self._model.load_state_dict(state.get("model", state), strict=False)
            self._model.eval().to(self.device)
            logger.info("[Realism] Loaded SIDA-style ViT checkpoint")
        except Exception as e:
            logger.warning(
                f"[Realism] Failed to load checkpoint ({e}); using pass-through scorer"
            )
            self._model = None

    def score(self, image: Image.Image) -> float:
        if self._model is None:
            return 1.0
        img_t = self._transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self._model(img_t)
            prob = torch.softmax(logits, dim=-1)[0, 1].item()
        return float(prob)


# ---------------------------------------------------------------------------
# Core editing logic
# ---------------------------------------------------------------------------
class EditExecutor:
    def __init__(
        self,
        lama: LamaInpainter,
        sdxl: SDXLInpainter,
        realism: RealismFilter,
        realism_threshold: float,
        output_images_root: Path,
    ):
        self.lama = lama
        self.sdxl = sdxl
        self.realism = realism
        self.realism_threshold = realism_threshold
        self.output_images_root = output_images_root


    def _get_mask(
        self, job: EditJob, image: Image.Image, effective_operation: str
    ) -> np.ndarray:
        """
        Prefer the precise union/SAM mask when available. The previous version
        replaced attribute-change masks with the raw bbox, which made edits
        bleed into background pixels and often reduced the visible edit to a
        weak colour wash over a large rectangle.
        """
        w, h = image.size
        meta = infer_anchor_meta(job)
        is_face = effective_operation == "attribute_change" and meta.is_face_edit
        dilation = MASK_DILATION_FACE_PX if is_face else MASK_DILATION_PX

        precise_mask = load_mask(
            job.union_mask_path, None, w, h, dilation=dilation
        ) if job.union_mask_path else np.zeros((h, w), dtype=np.uint8)

        bbox_mask = bbox_to_mask(job.bbox_xyxy, w, h, dilation=max(1, dilation // 2)) if job.bbox_xyxy else None

        if effective_operation == "attribute_change":
            if precise_mask.sum() > 0 and bbox_mask is not None:
                # keep only the object pixels inside the bbox region
                mixed = np.where((precise_mask > 0) & (bbox_mask > 0), 255, 0).astype(np.uint8)
                if mixed.sum() > 0:
                    precise_mask = mixed
            elif precise_mask.sum() == 0 and bbox_mask is not None:
                precise_mask = bbox_mask

            # tighten slightly so background edges do not get repainted
            if precise_mask.sum() > 0:
                precise_mask = erode_mask(precise_mask, 1 if is_face else 2)

            if precise_mask.sum() > 0:
                return precise_mask

            if bbox_mask is not None:
                return bbox_mask

        base_mask = precise_mask
        if base_mask.sum() == 0 and bbox_mask is not None:
            base_mask = bbox_mask

        if effective_operation == "replace":
            return self._tighten_replace_mask(base_mask, job, image)

        return base_mask

    def _job_seed(self, job: EditJob, attempt: int = 0) -> int:
        payload = f"{job.sample_id}|{job.anchor_id}|{job.operation}|{job.severity}|{job.candidate_index}|{attempt}"
        return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)

    def _masked_change_stats(self, original: Image.Image, edited: Image.Image, mask: np.ndarray) -> Tuple[float, float]:
        fg = mask > 0
        if fg.sum() == 0:
            return 0.0, 0.0
        orig = np.array(original.convert("RGB")).astype(np.float32)
        edit = np.array(edited.convert("RGB")).astype(np.float32)
        delta = np.abs(edit - orig).mean(axis=2)
        mean_delta = float(delta[fg].mean())
        changed_ratio = float((delta[fg] >= 6.0).mean())
        return mean_delta, changed_ratio

    def _edit_is_visible(self, original: Image.Image, edited: Image.Image, mask: np.ndarray, severity: str) -> bool:
        mean_delta, changed_ratio = self._masked_change_stats(original, edited, mask)
        return (
            mean_delta >= MIN_MASKED_MEAN_DELTA.get(severity, 4.0)
            and changed_ratio >= MIN_MASKED_CHANGE_RATIO.get(severity, 0.01)
        )

    def _attempt_schedule(self, effective_operation: str, severity: str, meta: AnchorMeta, stats: MaskStats) -> List[Dict[str, Any]]:
        if effective_operation == "attribute_change":
            base_strength = SDXL_STRENGTH_BY_OP["attribute_change_face"] if meta.is_face_edit else SDXL_STRENGTH_BY_OP["attribute_change"]
            if severity == "low":
                multipliers = [1.0, 1.15]
            elif severity == "medium":
                multipliers = [1.10, 1.25, 1.40]
            else:
                multipliers = [1.25, 1.45, 1.65]
            feather = 3 if meta.is_face_edit else 5
            return [
                {
                    "strength": min(0.92, base_strength * m),
                    "guidance_scale": SDXL_GUIDANCE_SCALE + (0.0 if i == 0 else 0.5 * i),
                    "num_inference_steps": SDXL_NUM_INFERENCE_STEPS + 4 * i,
                    "feather_radius": feather,
                }
                for i, m in enumerate(multipliers)
            ][:MAX_SDXL_ATTEMPTS]

        if effective_operation in ("fill", "insert"):
            base = SDXL_STRENGTH_BY_OP.get(effective_operation, SDXL_STRENGTH_BY_OP["default"])
            return [
                {
                    "strength": min(0.98, base * m),
                    "guidance_scale": SDXL_GUIDANCE_SCALE_INSERTION + 0.5 * i,
                    "num_inference_steps": SDXL_NUM_INFERENCE_STEPS_INSERTION + 4 * i,
                    "feather_radius": 6,
                }
                for i, m in enumerate([1.0, 1.08, 1.16])
            ][:MAX_SDXL_ATTEMPTS]

        if effective_operation == "replace":
            base_strength, _ = self._replace_strength_and_pad(stats, meta)
            return [
                {
                    "strength": min(0.95, base_strength * m),
                    "guidance_scale": SDXL_GUIDANCE_SCALE + 0.5 * i,
                    "num_inference_steps": SDXL_NUM_INFERENCE_STEPS + 4 * i,
                    "feather_radius": 5,
                }
                for i, m in enumerate([1.0, 1.12, 1.24])
            ][:MAX_SDXL_ATTEMPTS]

        return []

    def _output_path(
        self, job: EditJob, effective_operation: str, suffix: str = ""
    ) -> Path:
        sample_dir = self.output_images_root / job.sample_id
        ensure_dir(sample_dir)
        requested = (job.operation or "unknown").lower().strip().replace(" ", "_")
        effective = (effective_operation or "unknown").lower().strip().replace(" ", "_")
        fname = (
            f"{job.anchor_id}_req-{requested}_eff-{effective}_{job.severity}"
            f"_{job.candidate_index}{suffix}.png"
        )
        return sample_dir / fname

    def _reject_path(self, job: EditJob, effective_operation: str) -> Path:
        reject_dir = self.output_images_root / job.sample_id / "rejects"
        ensure_dir(reject_dir)
        requested = (job.operation or "unknown").lower().strip().replace(" ", "_")
        effective = (effective_operation or "unknown").lower().strip().replace(" ", "_")
        fname = (
            f"{job.anchor_id}_req-{requested}_eff-{effective}_{job.severity}"
            f"_{job.candidate_index}.png"
        )
        return reject_dir / fname

    def _mask_bbox(self, mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Return closed bbox (x1,y1,x2,y2) of the mask foreground."""
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    def _expand_box(
        self,
        box: Tuple[int, int, int, int],
        image_size: Tuple[int, int],
        pad_ratio: float = 0.12,
    ) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = box   # closed
        w, h = image_size
        bw = max(1, x2 - x1 + 1)
        bh = max(1, y2 - y1 + 1)
        px = max(4, int(bw * pad_ratio))
        py = max(4, int(bh * pad_ratio))
        return (
            max(0, x1 - px),
            max(0, y1 - py),
            min(w - 1, x2 + px),   # keep closed, clamped
            min(h - 1, y2 + py),
        )

    def _tighten_replace_mask(
        self, mask: np.ndarray, job: EditJob, image: Image.Image
    ) -> np.ndarray:
        w, h = image.size
        base = mask

        if job.bbox_xyxy:
            bbox_mask = bbox_to_mask(
                job.bbox_xyxy, w, h, dilation=max(1, MASK_DILATION_PX // 3)
            )
            base = np.where((mask > 0) & (bbox_mask > 0), 255, 0).astype(np.uint8)
            if base.sum() == 0:
                base = bbox_mask

        return erode_mask(base, REPLACE_MASK_EROSION_PX)

    def _replace_strength_and_pad(
        self, stats: MaskStats, meta: AnchorMeta
    ) -> Tuple[float, float]:
        """
        Choose SDXL strength and crop pad ratio for replace operations.
        Person-attached replacements use lower strength (less structural change)
        and wider padding (more context for the model).
        """
        if meta.is_person_like or meta.is_body_attached:
            if stats.area_ratio <= 0.06:
                return SDXL_STRENGTH_BY_OP["replace_small"] * 0.85, 0.14
            return SDXL_STRENGTH_BY_OP["replace_medium"] * 0.85, 0.18
        # Non-person
        if stats.area_ratio <= 0.06:
            return SDXL_STRENGTH_BY_OP["replace_small"], 0.10
        if stats.area_ratio <= 0.12:
            return SDXL_STRENGTH_BY_OP["replace_small"], 0.12
        return SDXL_STRENGTH_BY_OP["replace_medium"], 0.14

    def _route_operation(self, job: EditJob, mask: np.ndarray) -> RouteDecision:
        requested = job.operation.lower()
        stats = compute_mask_stats(mask)
        meta = infer_anchor_meta(job)

        if stats.is_empty:
            return RouteDecision("skip", "empty_mask")

        if requested == "attribute_change":
            return RouteDecision("attribute_change", "requested_attribute_change")

        if requested in ("fill", "insert"):
            return RouteDecision(requested, f"requested_{requested}")

        if requested == "remove":
            if meta.reveals_hidden_structure_when_removed:
                if meta.is_person_like or meta.is_body_attached:
                    return RouteDecision(
                        "attribute_change", "remove_reveals_hidden_structure_rerouted"
                    )
                if stats.area_ratio <= SCENE_REMOVE_AREA_RATIO:
                    return RouteDecision("remove", "scene_remove_despite_hidden_structure")
                return RouteDecision("skip", "remove_reveals_hidden_structure")

            if meta.is_body_attached and stats.area_ratio > MAX_BODY_ATTACHED_REMOVE_AREA_RATIO:
                return RouteDecision(
                    "attribute_change", "body_attached_remove_unstable_rerouted"
                )

            if stats.compactness < 0.10:
                return RouteDecision("skip", "remove_irregular_mask")

            if meta.is_person_like or meta.is_body_attached:
                if stats.area_ratio > MAX_REMOVE_AREA_RATIO:
                    return RouteDecision(
                        "attribute_change", "person_remove_too_large_rerouted"
                    )
                return RouteDecision("remove", "safe_person_remove")

            if stats.area_ratio > SCENE_REMOVE_AREA_RATIO:
                return RouteDecision("skip", "remove_too_large")

            return RouteDecision("remove", "safe_remove")

        if requested == "replace":
            if meta.implies_role_swap and (meta.is_person_like or meta.is_body_attached):
                return RouteDecision(
                    "attribute_change", "role_swap_rerouted_to_local_attribute_change"
                )

            if meta.is_whole_person:
                return RouteDecision("attribute_change", "whole_person_replace_disallowed")

            if stats.bbox_aspect_ratio > 4.5 or stats.bbox_aspect_ratio < 0.20:
                return RouteDecision("skip", "replace_strip_like_region")

            if stats.compactness < 0.10:
                return RouteDecision("skip", "replace_irregular_mask")

            if meta.is_identity_sensitive:
                return RouteDecision(
                    "attribute_change", "identity_sensitive_replace_rerouted"
                )

            if meta.is_person_like or meta.is_body_attached:
                if stats.area_ratio > PERSON_REPLACE_AREA_RATIO:
                    return RouteDecision(
                        "attribute_change", "person_like_replace_too_large_rerouted"
                    )
                return RouteDecision("replace", "safe_person_local_replace")

            if stats.touches_border_count >= 3 and stats.area_ratio > 0.45:
                return RouteDecision("skip", "replace_border_touching_extreme")

            if stats.area_ratio > SCENE_REPLACE_AREA_RATIO:
                return RouteDecision("skip", "replace_too_large")

            if stats.touches_border_count >= 2:
                return RouteDecision("replace", "replace_border_touching_allowed")

            if stats.area_ratio > LARGE_REPLACE_AREA_RATIO:
                return RouteDecision("replace", "safe_large_scene_replace")

            return RouteDecision("replace", "safe_replace")

        return RouteDecision("skip", f"unknown_operation_{requested}")

    def _do_sdxl(
        self,
        image: Image.Image,
        mask: np.ndarray,
        prompt: str,
        negative_prompt: str,
        strength: float,
        guidance_scale: float,
        num_inference_steps: int,
        seed: Optional[int] = None,
        use_padding_crop: bool = True,
    ) -> Image.Image:
        return self.sdxl.inpaint(
            image,
            mask,
            prompt,
            negative_prompt,
            strength=strength,
            seed=seed,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            use_padding_crop=use_padding_crop,
        )


    def _apply_colour_consistency(
        self,
        edited: Image.Image,
        original: Image.Image,
        mask: np.ndarray,
        effective_operation: str,
        meta: AnchorMeta,
    ) -> Image.Image:
        """
        Keep colour correction conservative. Heavy LAB matching was causing the
        visible tint/halo artefacts you noticed.
        """
        if effective_operation in ("attribute_change", "remove"):
            return edited

        if meta.is_person_like or meta.is_body_attached:
            return _luminance_match_border(edited, original, mask, border_px=8, strength=0.12)

        # For broader non-person scene edits, use only a light touch.
        edited = _lab_colour_match_masked(edited, original, mask, blend=0.18)
        edited = _luminance_match_border(edited, original, mask, border_px=8, strength=0.12)
        return edited

    def _do_sdxl_and_paste(
        self,
        image: Image.Image,
        mask: np.ndarray,
        prompt: str,
        negative_prompt: str,
        strength: float,
        guidance_scale: float,
        num_inference_steps: int,
        feather_radius: int,
        effective_operation: str,
        meta: AnchorMeta,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        SDXL already pastes the crop back into the full image when
        padding_mask_crop is used. We keep that result as the base image and only
        do a light edge blend to suppress seams, instead of re-mixing the entire
        masked region against the original. That preserves the edit strength.
        """
        inpainted = self._do_sdxl(
            image, mask, prompt, negative_prompt,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
            use_padding_crop=True,
        )

        inpainted = self._apply_colour_consistency(
            inpainted, image, mask, effective_operation=effective_operation, meta=meta
        )

        soft_mask = _distance_feather_mask(mask, radius=feather_radius)
        result = image.copy()
        result.paste(inpainted, (0, 0), mask=soft_mask)
        return result

    def _do_remove(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        return self.lama.inpaint(image, mask)

    def _remove_fallback_candidate(
        self,
        image: Image.Image,
        mask: np.ndarray,
        job: EditJob,
        meta: AnchorMeta,
        stats: MaskStats,
    ) -> Optional[Image.Image]:
        if not REMOVE_FALLBACK_TO_FILL:
            return None
        if stats.area_ratio > REMOVE_FALLBACK_MAX_AREA_RATIO:
            return None
        if meta.is_person_like and not meta.is_background_safe_remove and stats.area_ratio > 0.12:
            return None

        positive_prompt, negative_prompt = _build_remove_fill_prompt(job.edited_anchor, job.severity)
        schedule = [
            {"strength": 0.82, "guidance_scale": SDXL_GUIDANCE_SCALE_INSERTION, "num_inference_steps": 42, "feather_radius": 5},
            {"strength": 0.90, "guidance_scale": SDXL_GUIDANCE_SCALE_INSERTION + 0.5, "num_inference_steps": 48, "feather_radius": 6},
        ]
        edited = None
        for attempt_idx, params in enumerate(schedule):
            candidate = self._do_sdxl_and_paste(
                image,
                mask,
                positive_prompt,
                negative_prompt,
                strength=params["strength"],
                guidance_scale=params["guidance_scale"],
                num_inference_steps=params["num_inference_steps"],
                feather_radius=params["feather_radius"],
                effective_operation="fill",
                meta=meta,
                seed=self._job_seed(job, attempt=100 + attempt_idx),
            )
            edited = candidate
            if self._edit_is_visible(image, candidate, mask, job.severity):
                break
        return edited

    def execute(self, job: EditJob) -> EditResult:
        t0 = time.time()
        result = EditResult(job=job)

        try:
            image = Image.open(job.image_path).convert("RGB")

            raw_mask = load_mask(
                job.union_mask_path,
                job.bbox_xyxy,
                image.size[0],
                image.size[1],
                dilation=(
                    MASK_DILATION_FACE_PX
                    if _is_face_edit(job.edited_anchor)
                    else MASK_DILATION_PX
                ),
            )

            route = self._route_operation(job, raw_mask)
            result.effective_operation = route.effective_operation
            result.route_reason = route.route_reason

            if route.effective_operation == "skip":
                result.accepted = False
                result.reject_reason = route.route_reason
                result.duration_s = time.time() - t0
                return result

            effective_operation = route.effective_operation
            mask = self._get_mask(job, image, effective_operation)
            stats = compute_mask_stats(mask)
            meta = infer_anchor_meta(job)

            positive_prompt, negative_prompt = build_sdxl_prompt(
                job.fill_clause,
                job.visual_prompt,
                job.edited_anchor,
                operation=effective_operation,
                meta=meta,
                severity=job.severity,
            )

            if effective_operation == "remove":
                try:
                    edited = self._do_remove(image, mask)
                except Exception as remove_exc:
                    logger.warning(
                        f"[WARN] remove failed for {job.sample_id}/{job.anchor_id}; trying SDXL fill fallback: {remove_exc}"
                    )
                    fallback = self._remove_fallback_candidate(image, mask, job, meta, stats)
                    if fallback is None:
                        result.accepted = False
                        result.reject_reason = f"remove_failed_no_fallback: {remove_exc}"
                        result.duration_s = time.time() - t0
                        return result
                    edited = fallback
                    result.effective_operation = "fill"
                    result.route_reason = f"{route.route_reason}|remove_failed_fallback_to_fill"
                    effective_operation = "fill"

            elif effective_operation in ("attribute_change", "fill", "insert", "replace"):
                schedule = self._attempt_schedule(effective_operation, job.severity, meta, stats)
                if not schedule:
                    raise ValueError(f"No SDXL attempt schedule built for {effective_operation}")

                edited = None
                last_mean_delta = 0.0
                last_changed_ratio = 0.0
                for attempt_idx, params in enumerate(schedule):
                    candidate = self._do_sdxl_and_paste(
                        image,
                        mask,
                        positive_prompt,
                        negative_prompt,
                        strength=params["strength"],
                        guidance_scale=params["guidance_scale"],
                        num_inference_steps=params["num_inference_steps"],
                        feather_radius=params["feather_radius"],
                        effective_operation=effective_operation,
                        meta=meta,
                        seed=self._job_seed(job, attempt=attempt_idx),
                    )
                    edited = candidate
                    last_mean_delta, last_changed_ratio = self._masked_change_stats(
                        image, candidate, mask
                    )
                    result.masked_mean_delta = last_mean_delta
                    result.masked_changed_ratio = last_changed_ratio
                    if self._edit_is_visible(image, candidate, mask, job.severity):
                        break
                assert edited is not None
                if not self._edit_is_visible(image, edited, mask, job.severity):
                    out_path = self._reject_path(job, effective_operation)
                    edited.save(out_path)
                    result.output_path = out_path
                    result.accepted = False
                    result.reject_reason = (
                        "edit_change_below_visibility_threshold "
                        f"(mean_delta={last_mean_delta:.3f}, changed_ratio={last_changed_ratio:.3f})"
                    )
                    result.duration_s = time.time() - t0
                    return result

            else:
                raise ValueError(f"Unknown effective operation: {effective_operation}")

            if result.masked_mean_delta is None or result.masked_changed_ratio is None:
                md, cr = self._masked_change_stats(image, edited, mask)
                result.masked_mean_delta = md
                result.masked_changed_ratio = cr

            realism_score = self.realism.score(edited)
            result.realism_score = realism_score

            if realism_score < self.realism_threshold:
                reject_path = self._reject_path(job, effective_operation)
                edited.save(reject_path)
                result.output_path = reject_path
                result.accepted = False
                result.reject_reason = (
                    f"realism_score={realism_score:.3f} < "
                    f"threshold={self.realism_threshold}"
                )
            else:
                out_path = self._output_path(job, effective_operation)
                edited.save(out_path)
                result.output_path = out_path
                result.accepted = True

        except Exception as e:
            result.error = str(e)
            result.accepted = False
            logger.error(
                f"[ERROR] {job.sample_id}/{job.anchor_id}: {e}\n{traceback.format_exc()}"
            )

        result.duration_s = time.time() - t0
        return result


# ---------------------------------------------------------------------------
# Result serialisation
# ---------------------------------------------------------------------------
def serialise_result(result: EditResult) -> Dict[str, Any]:
    job = result.job
    return {
        "sample_id":            job.sample_id,
        "anchor_id":            job.anchor_id,
        "requested_operation":  job.operation,
        "effective_operation":  result.effective_operation,
        "route_reason":         result.route_reason,
        "severity":             job.severity,
        "candidate_index":      job.candidate_index,
        "tier":                 job.tier,
        "planning_score":       job.planning_score,
        "edit_instruction":     job.edit_instruction,
        "edited_anchor":        job.edited_anchor,
        "fill_clause":          job.fill_clause,
        "visual_prompt":        job.visual_prompt,
        "image_path":           str(job.image_path),
        "union_mask_path":      str(job.union_mask_path) if job.union_mask_path else None,
        "bbox_xyxy":            job.bbox_xyxy,
        "output_path":          str(result.output_path) if result.output_path else None,
        "lama_intermediate":    (
            str(result.lama_intermediate_path)
            if result.lama_intermediate_path else None
        ),
        "realism_score":        result.realism_score,
        "masked_mean_delta":    result.masked_mean_delta,
        "masked_changed_ratio": result.masked_changed_ratio,
        "accepted":             result.accepted,
        "reject_reason":        result.reject_reason,
        "error":                result.error,
        "duration_s":           round(result.duration_s, 3),
        "rewritten_caption":    job.rewritten_caption,
        "nli_direction":        job.nli_direction,
        "text_edit_operation":  job.text_edit_operation,
        "modality":             (
            "image_text" if job.rewritten_caption else "image_only"
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Execute T-IMPACT image edits from Qwen edit plans "
            "(LaMa + SDXL + realism filter)"
        )
    )

    p.add_argument("--edit-plans", type=str, required=True)
    p.add_argument("--anchors-jsonl", type=str, required=True)
    p.add_argument("--output-jsonl", type=str, required=True)
    p.add_argument("--images-root", type=str, default=".")
    p.add_argument("--output-images-root", type=str, default="data/edited_images")

    p.add_argument("--lama-config", type=str, default=None)
    p.add_argument("--lama-checkpoint", type=str, default=None)

    p.add_argument(
        "--sdxl-model",
        type=str,
        # Must be an inpainting-specific checkpoint (9-channel UNet).
        # The base SDXL model (stable-diffusion-xl-base-1.0) is a text-to-image
        # model and will not work correctly with StableDiffusionXLInpaintPipeline.
        default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    )

    p.add_argument("--realism-model", type=str, default=None)
    p.add_argument("--realism-threshold", type=float, default=DEFAULT_REALISM_THRESHOLD)

    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max-jobs", type=int, default=None)
    p.add_argument("--print-every", type=int, default=50)
    p.add_argument(
        "--severity-filter", type=str, default=None, choices=["low", "medium", "high"]
    )
    p.add_argument("--max-candidates-per-severity", type=int, default=None)
    p.add_argument("--log-level", type=str, default="INFO")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Mutual exclusion guard
    if args.overwrite and args.skip_existing:
        raise ValueError(
            "--overwrite and --skip-existing are mutually exclusive. "
            "Use one or the other."
        )

    edit_plans_path = Path(args.edit_plans)
    anchors_path = Path(args.anchors_jsonl)
    output_jsonl = Path(args.output_jsonl)
    images_root = Path(args.images_root)
    output_images_root = Path(args.output_images_root)

    ensure_dir(output_jsonl.parent)
    ensure_dir(output_images_root)

    if output_jsonl.exists() and args.overwrite:
        output_jsonl.unlink()
    elif output_jsonl.exists() and not args.skip_existing:
        raise FileExistsError(
            f"Output exists: {output_jsonl}. Use --skip-existing or --overwrite."
        )

    logger.info("[INFO] Loading anchor index…")
    anchor_index = load_anchors_index(anchors_path)
    logger.info(f"[INFO] Loaded {len(anchor_index)} anchor rows")

    logger.info("[INFO] Loading edit plans…")
    plan_rows = load_jsonl(edit_plans_path)
    logger.info(f"[INFO] Loaded {len(plan_rows)} plan rows")

    all_jobs: List[EditJob] = []
    for row in plan_rows:
        all_jobs.extend(extract_jobs_from_plan_row(
            row,
            anchor_index,
            images_root,
            max_candidates_per_severity=args.max_candidates_per_severity,
        ))

    from collections import Counter
    sev_counts = Counter(j.severity for j in all_jobs)
    logger.info(
        f"[INFO] Jobs extracted — "
        f"low={sev_counts.get('low', 0)}  "
        f"medium={sev_counts.get('medium', 0)}  "
        f"high={sev_counts.get('high', 0)}  "
        f"total={len(all_jobs)}"
    )

    if args.severity_filter:
        all_jobs = [j for j in all_jobs if j.severity == args.severity_filter]
        logger.info(
            f"[INFO] After --severity-filter '{args.severity_filter}': {len(all_jobs)} jobs"
        )

    if args.skip_existing:
        done = already_done_keys(output_jsonl)
        all_jobs = [
            j for j in all_jobs
            if (
                j.sample_id,
                j.anchor_id,
                j.operation,
                j.severity,
                j.candidate_index,
            ) not in done
        ]

    if args.max_jobs is not None:
        all_jobs = all_jobs[:args.max_jobs]

    logger.info(f"[INFO] Jobs to process: {len(all_jobs)}")
    if not all_jobs:
        logger.info("[INFO] Nothing to do.")
        return

    logger.info("[INFO] Loading LaMa…")
    lama = LamaInpainter(args.lama_config, args.lama_checkpoint, args.device)

    logger.info("[INFO] Loading SDXL…")
    sdxl = SDXLInpainter(args.sdxl_model, args.device)

    logger.info("[INFO] Loading realism filter…")
    realism = RealismFilter(args.realism_model, args.device)

    executor = EditExecutor(
        lama=lama,
        sdxl=sdxl,
        realism=realism,
        realism_threshold=args.realism_threshold,
        output_images_root=output_images_root,
    )

    accepted = rejected = errors = 0
    with output_jsonl.open("a", encoding="utf-8") as fout:
        for idx, job in enumerate(all_jobs, 1):
            result = executor.execute(job)
            fout.write(json.dumps(serialise_result(result), ensure_ascii=False) + "\n")
            fout.flush()

            if result.error:
                errors += 1
            elif result.accepted:
                accepted += 1
            else:
                rejected += 1

            if idx % max(1, args.print_every) == 0:
                logger.info(
                    f"[INFO] {idx}/{len(all_jobs)}  "
                    f"accepted={accepted}  rejected={rejected}  errors={errors}"
                )

    logger.info(
        f"[DONE] accepted={accepted}  rejected={rejected}  errors={errors}  "
        f"→ {output_jsonl}"
    )


if __name__ == "__main__":
    main()
