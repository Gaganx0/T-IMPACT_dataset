#!/usr/bin/env python3
"""
T-IMPACT final release builder (patched).

Fixes over the initial builder:
- packages text_only examples directly from planner candidate headline rewrites
- uses collision-resistant pair_ids
- writes both metadata.csv and metadata_rich.csv
- surfaces explanation fields in CSV
- reports requested->final severity and requested->effective operation mappings
- optional visual defect QC with heuristic + VLM judge
- deduplicates examples deterministically

Ghosting / transparency patch (v2):
- adds detect_ghosting_transparency() with four independent signals
- heuristic_visual_qc raises confidence to 0.80 when ghosting is confirmed
- build_examples_from_edit_results quarantines whenever the heuristic backend
  itself (not just the manager mode) marks acceptable=False, so ghosting is
  caught in hybrid mode even when VLM is absent / silent
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageStat
import numpy as np
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

SEVERITY_WEIGHTS = {
    "tier_obj": 0.35,
    "ctx_incongruity": 0.25,
    "nli_contradiction": 0.20,
    "salience": 0.15,
    "visibility": 0.05,
}
DEFAULT_THRESHOLDS = {"low_max": 0.30, "medium_max": 0.60}
DEFAULT_TIER_PRIOR = {"tier_a": 0.15, "tier_b": 0.50, "tier_c": 0.85}
DEFAULT_NONE_MAX = None
DEFAULT_SPLITS = (0.80, 0.10, 0.10)
VALID_MODALITIES = {"pristine", "image_only", "text_only", "image_text"}


def clamp(x: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        x = float(x)
    except Exception:
        return lo
    return max(lo, min(hi, x))


def safe_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def safe_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []


def safe_str(x: Any) -> str:
    return x if isinstance(x, str) else ""


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def normalize_text(s: str) -> str:
    s = safe_str(s).lower()
    s = re.sub(r"[^\w\s']+", " ", s)
    return normalize_ws(s)


def token_set(s: str) -> set[str]:
    return set(t for t in normalize_text(s).split() if t)


def lexical_change_ratio(a: str, b: str) -> float:
    sa, sb = token_set(a), token_set(b)
    union = sa | sb
    if not union:
        return 0.0
    return 1.0 - (len(sa & sb) / len(union))


def bucket_from_score(score: float, low_max: float, medium_max: float) -> str:
    if score <= low_max:
        return "low"
    if score <= medium_max:
        return "medium"
    return "high"


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def split_from_sample_id(sample_id: str, train: float, val: float) -> str:
    h = int(stable_hash(sample_id)[:8], 16) / 0xFFFFFFFF
    if h < train:
        return "train"
    if h < train + val:
        return "val"
    return "test"


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def iter_jsonl(paths: Sequence[str]) -> Iterable[Dict[str, Any]]:
    for pat in paths:
        for fp in sorted(glob.glob(pat)):
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            yield obj
                    except Exception:
                        continue


def load_jsonl_map(paths: Sequence[str], key_fields: Tuple[str, ...]) -> Dict[Tuple[str, ...], Dict[str, Any]]:
    out: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    for row in iter_jsonl(paths):
        key = tuple(safe_str(row.get(k)) for k in key_fields)
        if all(key):
            out[key] = row
    return out


def load_jsonl_rows(paths: Sequence[str]) -> List[Dict[str, Any]]:
    return list(iter_jsonl(paths))


def image_qc(path: str, min_side: int = 96, min_std: float = 4.0) -> Tuple[bool, str, Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return False, "missing_image", {}
    try:
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            im = im.convert("RGB")
            w, h = im.size
            if min(w, h) < min_side:
                return False, "image_too_small", {"width": w, "height": h}
            if max(w / max(h, 1), h / max(w, 1)) > 6.0:
                return False, "extreme_aspect_ratio", {"width": w, "height": h}
            gray = im.convert("L")
            stat = ImageStat.Stat(gray)
            std = stat.stddev[0] if stat.stddev else 0.0
            if std < min_std:
                return False, "low_variance_image", {"width": w, "height": h, "gray_std": round(std, 4)}
            return True, "ok", {"width": w, "height": h, "gray_std": round(std, 4)}
    except Exception as e:
        return False, f"image_error:{type(e).__name__}", {}


def has_human_anchor(anchor_row: Optional[Dict[str, Any]], manifest_row: Optional[Dict[str, Any]]) -> bool:
    human_terms = {"person", "people", "man", "woman", "boy", "girl", "face", "hand", "body", "human", "adult", "child"}
    def _scan(obj: Any) -> bool:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in {"category", "anchor_norm", "anchor_text", "name", "label"} and normalize_text(safe_str(v)) in human_terms:
                    return True
                if _scan(v):
                    return True
        elif isinstance(obj, list):
            for v in obj:
                if _scan(v):
                    return True
        elif isinstance(obj, str):
            if normalize_text(obj) in human_terms:
                return True
        return False
    return _scan(anchor_row or {}) or _scan(manifest_row or {})


# ---------------------------------------------------------------------------
# NEW: ghosting / transparency artefact detector
# ---------------------------------------------------------------------------

def detect_ghosting_transparency(
    gray: "np.ndarray",
    arr_bgr: "np.ndarray",
) -> List[str]:
    """
    Detect inpainting-induced transparency and ghosting artefacts.

    Four independent signals are checked; each one that fires adds a reason
    string.  The caller (heuristic_visual_qc) treats *any* hit as a defect.

    Signal 1 – doubled-edge ratio
        Failed inpainting leaves the background edges visible *through* the
        transparent subject, creating a much higher ratio of fine edges to
        coarse (Gaussian-smoothed) edges than a clean photo ever would.

    Signal 2 – mid-tone pixel overload
        A ghost subject is an additive blend of the subject and the background.
        That blend pushes pixel values toward the mid-range [60, 196].  Clean
        photos concentrate energy at both ends (dark shadows, bright highlights).

    Signal 3 – centre-gradient suppression
        An opaque subject in the centre of frame normally has stronger local
        gradients than the plain sky / background at the top and bottom borders.
        A transparent subject inverts or flattens this ratio.

    Signal 4 – centre desaturation
        Blending a colourful subject with a neutral background de-saturates the
        blended region.  If the central patch is significantly less saturated
        than the frame average the subject is likely ghosted.
    """
    if cv2 is None:
        return []

    reasons: List[str] = []
    h, w = gray.shape

    # -- Signal 1: doubled-edge ratio -----------------------------------------
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges_sharp = cv2.Canny(gray, 40, 110)
    edges_smooth = cv2.Canny(blurred, 40, 110)
    sharp_density = float((edges_sharp > 0).mean())
    smooth_density = float((edges_smooth > 0).mean())
    if smooth_density > 1e-4 and sharp_density / smooth_density > 4.0:
        reasons.append("doubled_edge_ghosting")

    # -- Signal 2: mid-tone overload ------------------------------------------
    hist = cv2.calcHist([gray], [0], None, [256], [0.0, 256.0]).flatten()
    hist /= hist.sum() + 1e-9
    mid_mass = float(hist[60:196].sum())
    if mid_mass > 0.70:
        reasons.append("midtone_overload_ghosting")

    # -- Signal 3: centre-gradient suppression --------------------------------
    pad_y = max(h // 8, 16)
    pad_x = max(w // 8, 16)
    cy, cx = h // 2, w // 2
    sobel_mag = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3))
    centre_grad = float(sobel_mag[cy - pad_y : cy + pad_y, cx - pad_x : cx + pad_x].mean())
    border_grad = float(
        (sobel_mag[:pad_y, :].mean() + sobel_mag[-pad_y:, :].mean()) / 2.0
    )
    if border_grad > 1e-3 and centre_grad / border_grad < 0.55:
        reasons.append("centre_gradient_suppression_ghosting")

    # -- Signal 4: centre desaturation ----------------------------------------
    hsv = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat = hsv[:, :, 1]
    centre_sat = float(sat[cy - pad_y : cy + pad_y, cx - pad_x : cx + pad_x].mean())
    overall_sat = float(sat.mean())
    if overall_sat > 5.0 and centre_sat / overall_sat < 0.65:
        reasons.append("centre_desaturation_ghosting")

    return reasons


# ---------------------------------------------------------------------------

def heuristic_visual_qc(path: str, human_focus: bool = False, blur_min: float = 45.0) -> Dict[str, Any]:
    out = {
        "backend": "heuristic",
        "checked": False,
        "acceptable": True,
        "confidence": 0.0,
        "reasons": [],
        "metrics": {},
        "raw_response": None,
    }
    p = Path(path)
    if not p.exists():
        out.update({"checked": True, "acceptable": False, "confidence": 1.0, "reasons": ["missing_image"]})
        return out
    try:
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            im = im.convert("RGB")
            out["checked"] = True
            out["metrics"]["width"], out["metrics"]["height"] = im.size
            arr = None
            if cv2 is not None:
                arr = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

                # --- original blur / edge checks ----------------------------
                lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                out["metrics"]["laplacian_var"] = round(lap_var, 4)
                if lap_var < blur_min:
                    out["reasons"].append("blurred_or_smeared")
                edges = cv2.Canny(gray, 80, 160)
                edge_ratio = float((edges > 0).mean())
                out["metrics"]["edge_ratio"] = round(edge_ratio, 6)
                if human_focus and edge_ratio < 0.015:
                    out["reasons"].append("low_structural_detail_human")

                # --- NEW: ghosting / transparency check ---------------------
                ghost_reasons = detect_ghosting_transparency(gray, arr)
                out["reasons"].extend(ghost_reasons)
                out["metrics"]["ghost_signals"] = ghost_reasons

            stat = ImageStat.Stat(im)
            means = stat.mean if stat.mean else [0.0, 0.0, 0.0]
            out["metrics"]["rgb_mean"] = [round(x, 3) for x in means]
    except Exception as e:
        out.update({"checked": True, "acceptable": False, "confidence": 1.0, "reasons": [f"image_error:{type(e).__name__}"]})
        return out

    if out["reasons"]:
        out["acceptable"] = False
        # Use higher confidence when ghosting signals are present – they are
        # strong and specific, unlike the generic blur check.
        ghosting_hit = any("ghosting" in r or "midtone" in r for r in out["reasons"])
        if ghosting_hit:
            out["confidence"] = 0.80
        elif human_focus:
            out["confidence"] = 0.72
        else:
            out["confidence"] = 0.60
    return out


class VLMVisualDefectJudge:
    def __init__(self, model_name_or_path: str, device: str = "cuda"):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self._loaded = False
        self._processor = None
        self._model = None
        self._torch = None
        self.load_error = None

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            import torch
            from transformers import AutoProcessor
            self._torch = torch
            model_cls = None
            import transformers
            for name in ["Qwen3VLForConditionalGeneration", "Qwen2_5_VLForConditionalGeneration", "Qwen2VLForConditionalGeneration"]:
                model_cls = getattr(transformers, name, None)
                if model_cls is not None:
                    break
            if model_cls is None:
                try:
                    from transformers import AutoModelForVision2Seq as model_cls
                except Exception:
                    from transformers import AutoModelForImageTextToText as model_cls
            self._processor = AutoProcessor.from_pretrained(self.model_name_or_path, trust_remote_code=True)
            kwargs = {"trust_remote_code": True}
            if self.device == "cuda" and torch.cuda.is_available():
                kwargs["torch_dtype"] = getattr(torch, "bfloat16", torch.float16)
                kwargs["device_map"] = "auto"
            self._model = model_cls.from_pretrained(self.model_name_or_path, **kwargs)
        except Exception as e:
            self.load_error = f"{type(e).__name__}: {e}"

    def judge(self, image_path: str, human_focus: bool = False) -> Dict[str, Any]:
        self._load()
        if self.load_error:
            return {
                "backend": "vlm",
                "checked": False,
                "acceptable": None,
                "confidence": 0.0,
                "reasons": [f"vlm_unavailable:{self.load_error}"],
                "metrics": {},
                "raw_response": None,
            }
        prompt = (
            "You are checking whether this edited image is suitable for inclusion in a research dataset. "
            "Focus only on visual defects, not truthfulness. "
            "Return strict JSON with keys acceptable, defect_type, severity, confidence, reason. "
            "Reject the image if there are obvious generative defects, including distorted limbs, fused regions, "
            "broken proportions, unnatural facial structure, impossible object shapes, or corrupted edited regions. "
            "Also reject if the subject appears semi-transparent, ghosted, or blended with the background "
            "(i.e. you can see through the subject to the background behind it)."
        )
        if human_focus:
            prompt += " Be especially strict about anatomy, body boundaries, hands, face structure, and clothing-body blending."
        try:
            image = Image.open(image_path).convert("RGB")
            messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
            if hasattr(self._processor, "apply_chat_template"):
                text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self._processor(text=[text], images=[image], return_tensors="pt")
            else:
                inputs = self._processor(images=[image], text=[prompt], return_tensors="pt")
            if self._torch is not None and hasattr(self._model, "device"):
                inputs = {k: (v.to(self._model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
            generated = self._model.generate(**inputs, max_new_tokens=220)
            input_ids = inputs.get("input_ids")
            if input_ids is not None and getattr(generated, "shape", [0,0])[1] > input_ids.shape[1]:
                generated = generated[:, input_ids.shape[1]:]
            text_out = self._processor.batch_decode(generated, skip_special_tokens=True)[0]
            m = re.search(r"\{.*\}", text_out, flags=re.S)
            obj = json.loads(m.group(0)) if m else {}
            acceptable = obj.get("acceptable")
            defect_type = obj.get("defect_type")
            if isinstance(defect_type, str):
                reasons = [normalize_ws(defect_type)] if defect_type and defect_type.lower() != "none" else []
            else:
                reasons = [normalize_ws(str(x)) for x in defect_type or [] if str(x).strip() and str(x).lower() != "none"]
            reason = normalize_ws(str(obj.get("reason", "")))
            if reason:
                reasons.append(reason)
            conf = clamp(obj.get("confidence", 0.0))
            return {
                "backend": "vlm",
                "checked": True,
                "acceptable": bool(acceptable) if acceptable is not None else None,
                "confidence": conf,
                "reasons": list(dict.fromkeys(reasons)),
                "metrics": {"severity": normalize_ws(str(obj.get("severity", "")))},
                "raw_response": text_out,
            }
        except Exception as e:
            return {
                "backend": "vlm",
                "checked": True,
                "acceptable": None,
                "confidence": 0.0,
                "reasons": [f"vlm_inference_error:{type(e).__name__}"],
                "metrics": {},
                "raw_response": None,
            }


class VisualQCManager:
    def __init__(self, mode: str = "heuristic", vlm_model: str = "", confidence_min: float = 0.65, human_only: bool = False):
        self.mode = mode
        self.confidence_min = confidence_min
        self.human_only = human_only
        self.vlm = VLMVisualDefectJudge(vlm_model) if vlm_model else None

    def evaluate(self, image_path: str, human_focus: bool = False) -> Dict[str, Any]:
        if self.mode == "off":
            return {"mode": "off", "checked": False, "acceptable": None, "confidence": 0.0, "reasons": [], "subchecks": []}
        if self.human_only and not human_focus:
            return {"mode": self.mode, "checked": False, "acceptable": None, "confidence": 0.0, "reasons": [], "subchecks": []}
        subchecks = []
        h = heuristic_visual_qc(image_path, human_focus=human_focus)
        subchecks.append(h)
        final_ok = h.get("acceptable", True)
        final_conf = clamp(h.get("confidence", 0.0))
        final_reasons = list(h.get("reasons", []))
        checked = bool(h.get("checked"))
        if self.mode in {"vlm", "hybrid"} and self.vlm is not None:
            v = self.vlm.judge(image_path, human_focus=human_focus)
            subchecks.append(v)
            checked = checked or bool(v.get("checked"))
            if v.get("acceptable") is False and clamp(v.get("confidence", 0.0)) >= self.confidence_min:
                final_ok = False
                final_conf = max(final_conf, clamp(v.get("confidence", 0.0)))
                final_reasons.extend(v.get("reasons", []))
            elif self.mode == "vlm" and v.get("acceptable") is True:
                final_ok = True
                final_conf = clamp(v.get("confidence", 0.0))
                final_reasons = []
        final_reasons = list(dict.fromkeys([r for r in final_reasons if r]))
        return {
            "mode": self.mode,
            "checked": checked,
            "acceptable": final_ok,
            "confidence": round(final_conf, 4),
            "reasons": final_reasons,
            "subchecks": subchecks,
        }


def text_qc(
    text: str,
    original_text: Optional[str] = None,
    min_tokens: int = 3,
    max_tokens: int = 80,
    lexical_change_min: float = 0.01,
) -> Tuple[bool, str, Dict[str, Any]]:
    t = normalize_ws(text)
    if not t:
        return False, "empty_text", {}
    if any(tag in t.lower() for tag in ["[mask]", "[fill]", "[visual]"]):
        return False, "prompt_markup_left_in_text", {}
    tokens = t.split()
    if len(tokens) < min_tokens:
        return False, "too_short", {"token_count": len(tokens)}
    if len(tokens) > max_tokens:
        return False, "too_long", {"token_count": len(tokens)}
    if re.search(r"(.)\1{4,}", t):
        return False, "repeated_chars", {}
    meta = {"token_count": len(tokens)}
    if original_text is not None:
        diff = lexical_change_ratio(original_text, t)
        meta["lexical_change"] = round(diff, 4)
        if diff < lexical_change_min:
            return False, "too_similar_to_original", meta
    return True, "ok", meta



def parse_calibration_map(path: Optional[str]) -> Optional[List[Tuple[float, float]]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    obj = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(obj, dict) and "points" in obj:
        obj = obj["points"]
    if not isinstance(obj, list):
        raise ValueError("Calibration JSON must be a list of [raw, calibrated] points or {'points': ...}")
    pts = []
    for item in obj:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            pts.append((clamp(item[0]), clamp(item[1])))
        elif isinstance(item, dict):
            pts.append((clamp(item.get("raw")), clamp(item.get("calibrated"))))
    pts.sort(key=lambda x: x[0])
    if not pts:
        raise ValueError("Calibration map is empty")
    return pts


def parse_severity_model(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Severity model file not found: {path}")
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("Severity model JSON must be an object")

    weights = safe_dict(obj.get("weights"))
    tier_priors = safe_dict(obj.get("tier_priors"))
    thresholds = safe_dict(obj.get("thresholds"))

    calibration_obj = obj.get("calibration")
    calibration = None
    if isinstance(calibration_obj, dict) and "points" in calibration_obj:
        calibration = parse_inline_calibration_points(calibration_obj.get("points"))
    elif isinstance(calibration_obj, list):
        calibration = parse_inline_calibration_points(calibration_obj)
    elif isinstance(obj.get("calibration_points"), list):
        calibration = parse_inline_calibration_points(obj.get("calibration_points"))

    return {
        "model_version": safe_str(obj.get("model_version") or obj.get("version")) or p.stem,
        "weights": {
            k: clamp(weights.get(k))
            for k in ("tier_obj", "ctx_incongruity", "nli_contradiction", "salience", "visibility")
            if k in weights
        },
        "tier_priors": {
            k: clamp(tier_priors.get(k))
            for k in ("tier_a", "tier_b", "tier_c")
            if k in tier_priors
        },
        "thresholds": {
            "none_max": thresholds.get("none_max"),
            "low_max": thresholds.get("low_max"),
            "medium_max": thresholds.get("medium_max"),
        },
        "calibration": calibration,
    }


def parse_inline_calibration_points(obj: Any) -> Optional[List[Tuple[float, float]]]:
    if not isinstance(obj, list):
        return None
    pts = []
    for item in obj:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            pts.append((clamp(item[0]), clamp(item[1])))
        elif isinstance(item, dict):
            pts.append((clamp(item.get("raw")), clamp(item.get("calibrated"))))
    pts.sort(key=lambda x: x[0])
    return pts or None


def apply_calibration(raw: float, calibration: Optional[List[Tuple[float, float]]]) -> float:
    raw = clamp(raw)
    if not calibration:
        return raw
    if raw <= calibration[0][0]:
        return calibration[0][1]
    if raw >= calibration[-1][0]:
        return calibration[-1][1]
    for (x0, y0), (x1, y1) in zip(calibration, calibration[1:]):
        if x0 <= raw <= x1:
            if x1 == x0:
                return y1
            t = (raw - x0) / (x1 - x0)
            return clamp(y0 + t * (y1 - y0))
    return raw


def load_manifests(manifest_dir: Optional[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not manifest_dir:
        return out
    md = Path(manifest_dir)
    if not md.exists():
        return out
    for path in sorted(md.glob("*.json")):
        if path.name == "index.json":
            continue
        obj = read_json(path)
        if not obj:
            continue
        sid = safe_str(obj.get("sample_id"))
        if sid:
            out[sid] = obj
    return out


def collect_pristine_examples(manifests: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for sid, mf in manifests.items():
        image_path = safe_str(mf.get("image_path"))
        headline = safe_str(mf.get("headline"))
        summary = safe_str(mf.get("summary"))
        img_ok, _, img_meta = image_qc(image_path)
        txt_ok, _, txt_meta = text_qc(headline or summary or safe_str(safe_dict(mf.get("caption")).get("image_caption_literal")))
        if not img_ok or not txt_ok:
            continue
        pair_id = f"{sid}__pristine"
        out.append({
            "pair_id": pair_id,
            "sample_id": sid,
            "anchor_id": None,
            "authenticity_label": "pristine",
            "is_pristine": True,
            "modality": "pristine",
            "severity_model_version": None,
            "severity_label": None,
            "severity_score_raw": None,
            "severity_score": None,
            "severity_components": None,
            "requested_generation_severity": None,
            "edit_type_requested": None,
            "edit_type_effective": None,
            "headline_original": headline,
            "headline_edited": headline,
            "summary_original": summary,
            "summary_edited": summary,
            "image_original": image_path,
            "image_edited": image_path,
            "mask_path": None,
            "bbox_xyxy": None,
            "explanation": None,
            "quality": {
                "image_qc": img_meta,
                "text_qc": txt_meta,
                "filter_reason": "passed",
            },
            "provenance": {"manifest_present": True},
        })
    return out


def compute_delta_sem_plan(candidate: Dict[str, Any]) -> float:
    ss = clamp(candidate.get("semantic_shift", 0.0))
    rs = clamp(candidate.get("role_shift", 0.0))
    pi = clamp(candidate.get("public_impact", 0.0))
    return clamp(0.50 * ss + 0.30 * rs + 0.20 * pi)


def resolve_tier_prior(plan_row: Dict[str, Any], fallback_tier: str = "") -> float:
    planner_input = safe_dict(plan_row.get("planner_input"))
    planner_out = safe_dict(plan_row.get("planner_output"))
    proxies = safe_dict(planner_out.get("planning_proxies"))
    tier_prior = planner_input.get("tier_prior")
    if tier_prior is None:
        tier_prior = proxies.get("tier_prior")
    if tier_prior is None:
        tier_name = normalize_text(safe_str(fallback_tier) or safe_str(planner_input.get("tier_name")) or safe_str(proxies.get("tier_name")))
        tier_prior = DEFAULT_TIER_PRIOR.get(tier_name, 0.50)
    return clamp(tier_prior)


def get_candidate_from_plan(plan_row: Dict[str, Any], severity: str, candidate_index: int) -> Dict[str, Any]:
    planner_out = safe_dict(plan_row.get("planner_output"))
    sev_map = safe_dict(planner_out.get("severity_candidates"))
    cands = safe_list(sev_map.get(severity))
    if 0 <= candidate_index < len(cands) and isinstance(cands[candidate_index], dict):
        return cands[candidate_index]
    return {}


def resolve_severity_components_from_candidate(
    candidate: Dict[str, Any],
    plan_row: Dict[str, Any],
    fallback_tier: str = "",
    nli_direction: str = "",
) -> Dict[str, float]:
    planner_input = safe_dict(plan_row.get("planner_input"))
    planner_out = safe_dict(plan_row.get("planner_output"))
    proxies = safe_dict(planner_out.get("planning_proxies"))
    t_obj = resolve_tier_prior(plan_row, fallback_tier=fallback_tier)
    i_ctx = clamp(safe_dict(candidate.get("derived")).get("delta_sem_plan"))
    if i_ctx == 0.0:
        i_ctx = compute_delta_sem_plan(candidate)
    c_nli = clamp(candidate.get("contradiction_potential"))
    if c_nli == 0.0:
        c_nli = {"contradiction": 1.0, "neutral": 0.5, "entailment": 0.0}.get(normalize_text(nli_direction), 0.0)
    sal = clamp(proxies.get("blended_salience") if proxies else planner_input.get("salience_proxy"))
    vis = clamp(proxies.get("blended_visibility") if proxies else planner_input.get("visibility_proxy"))
    return {"T_obj": t_obj, "I_ctx": i_ctx, "C_nli": c_nli, "Sal": sal, "Vis": vis}


def compute_severity_score(components: Dict[str, float]) -> float:
    return clamp(
        SEVERITY_WEIGHTS["tier_obj"] * clamp(components.get("T_obj"))
        + SEVERITY_WEIGHTS["ctx_incongruity"] * clamp(components.get("I_ctx"))
        + SEVERITY_WEIGHTS["nli_contradiction"] * clamp(components.get("C_nli"))
        + SEVERITY_WEIGHTS["salience"] * clamp(components.get("Sal"))
        + SEVERITY_WEIGHTS["visibility"] * (1.0 - clamp(components.get("Vis")))
    )


@dataclass
class BuildConfig:
    repo_root: Path
    output_root: Path
    realism_min: float
    changed_ratio_min: float
    changed_ratio_max: float
    mean_delta_min: float
    low_max: float
    medium_max: float
    none_max: Optional[float]
    severity_model_version: str
    exclude_sensitive: bool
    calibration: Optional[List[Tuple[float, float]]]
    visual_qc_mode: str
    visual_qc_confidence_min: float
    require_edit_accepted: bool
    require_anchor_valid: bool
    allow_cross_mode_rewrite_fallback: bool
    include_joint_as_text_only: bool
    synthesize_missing_rewrites: bool
    text_min_tokens: int
    text_max_tokens: int
    text_lexical_change_min: float
    visual_heuristic_hard_reject: bool
    visual_judge: Optional[Any] = None


def parse_raw_rewrite_output(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    s = safe_str(raw).strip()
    if not s:
        return {}
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        m = re.search(r"\{.*\}", s, flags=re.S)
        if not m:
            return {}
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}


def merge_rewrite_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(parse_raw_rewrite_output(payload.get("raw_rewrite_output")))
    merged.update(payload)
    return merged


def get_rewrite_payload(candidate: Dict[str, Any], mode: str) -> Dict[str, Any]:
    hr = safe_dict(candidate.get("headline_rewrite"))
    payload = safe_dict(hr.get(mode))
    if not payload:
        return {}
    merged = merge_rewrite_payload(payload)
    merged["_rewrite_mode"] = mode
    return merged


def resolve_rewrite_payload(
    candidate: Dict[str, Any],
    preferred_mode: str,
    allow_cross_mode_rewrite_fallback: bool = True,
) -> Dict[str, Any]:
    modes = [preferred_mode]
    if allow_cross_mode_rewrite_fallback:
        if preferred_mode == "joint":
            modes.append("text_only")
        elif preferred_mode == "text_only":
            modes.append("joint")
    for mode in modes:
        payload = get_rewrite_payload(candidate, mode)
        if safe_str(payload.get("rewritten_headline")):
            if mode != preferred_mode:
                payload["_rewrite_source"] = f"{mode}_fallback"
            else:
                payload["_rewrite_source"] = mode
            return payload
    return {}


def synthesize_rewritten_headline(original_headline: str, candidate: Dict[str, Any]) -> str:
    base = normalize_ws(original_headline)
    if not base:
        return ""
    op = normalize_text(safe_str(candidate.get("operation")))
    anchor = normalize_ws(safe_str(candidate.get("edited_anchor")))
    instruction = normalize_ws(safe_str(candidate.get("edit_instruction")))
    if op == "remove" and anchor:
        return f"{base} without {anchor}"
    if op == "attribute_change" and anchor:
        return f"{base} after change to {anchor}"
    if op == "replace" and anchor:
        return f"{base} featuring {anchor}"
    if anchor:
        return f"{base} involving {anchor}"
    if instruction:
        return f"{base}: {instruction}"
    return ""


def collect_text_only_rewrite_variants(
    candidate: Dict[str, Any],
    original_headline: str,
    cfg: BuildConfig,
) -> List[Dict[str, Any]]:
    variants: List[Dict[str, Any]] = []
    seen: set[str] = set()

    primary = resolve_rewrite_payload(
        candidate,
        "text_only",
        allow_cross_mode_rewrite_fallback=cfg.allow_cross_mode_rewrite_fallback,
    )
    if primary:
        rewritten = normalize_ws(safe_str(primary.get("rewritten_headline")))
        key = normalize_text(rewritten)
        if rewritten and key not in seen:
            payload = dict(primary)
            payload["_rewrite_source"] = safe_str(payload.get("_rewrite_source")) or "text_only"
            variants.append(payload)
            seen.add(key)

    if cfg.include_joint_as_text_only:
        joint = get_rewrite_payload(candidate, "joint")
        rewritten = normalize_ws(safe_str(joint.get("rewritten_headline")))
        key = normalize_text(rewritten)
        if rewritten and key not in seen:
            payload = dict(joint)
            payload["_rewrite_source"] = "joint_as_text_only"
            variants.append(payload)
            seen.add(key)

    if not variants and cfg.synthesize_missing_rewrites:
        synth = synthesize_rewritten_headline(original_headline, candidate)
        key = normalize_text(synth)
        if synth and key not in seen:
            variants.append({
                "rewritten_headline": synth,
                "text_edit_operation": safe_str(candidate.get("operation")),
                "nli_direction": "",
                "rewrite_rationale": "synthetic_fallback_from_builder",
                "_rewrite_source": "synthetic",
                "_synthetic": True,
            })
    return variants


def resolve_original_text(manifest_row: Optional[Dict[str, Any]], plan_row: Dict[str, Any]) -> str:
    planner_input = safe_dict(plan_row.get("planner_input"))
    if manifest_row:
        return safe_str(manifest_row.get("headline")) or safe_str(planner_input.get("headline"))
    return safe_str(planner_input.get("headline"))


def resolve_original_text(manifest_row: Optional[Dict[str, Any]], plan_row: Dict[str, Any]) -> str:
    planner_input = safe_dict(plan_row.get("planner_input"))
    if manifest_row:
        return safe_str(manifest_row.get("headline")) or safe_str(planner_input.get("headline"))
    return safe_str(planner_input.get("headline"))


def resolve_summary(manifest_row: Optional[Dict[str, Any]], plan_row: Dict[str, Any]) -> str:
    planner_input = safe_dict(plan_row.get("planner_input"))
    if manifest_row:
        return safe_str(manifest_row.get("summary")) or safe_str(planner_input.get("summary"))
    return safe_str(planner_input.get("summary"))


def detect_modality(edit_row: Dict[str, Any]) -> str:
    m = normalize_text(safe_str(edit_row.get("modality")))
    if m in VALID_MODALITIES:
        return m
    if m == "imagetext":
        return "image_text"
    return "image_only"


def make_pair_id(
    sid: str,
    aid: str,
    modality: str,
    requested_op: str,
    effective_op: str,
    requested_severity: str,
    final_severity: str,
    candidate_index: int,
    headline_edited: str,
    image_edited: str,
) -> str:
    seed = "|".join([
        sid, aid, modality, requested_op, effective_op, requested_severity,
        final_severity, str(candidate_index), normalize_ws(headline_edited), image_edited
    ])
    suffix = stable_hash(seed)[:10]
    return f"{sid}__{aid}__{modality}__{effective_op or requested_op or 'unknown'}__{requested_severity or 'na'}__{final_severity or 'na'}__{candidate_index}__{suffix}"


def base_explanation(candidate: Dict[str, Any], rewrite_obj: Optional[Dict[str, Any]], row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    row = row or {}
    rewrite_obj = rewrite_obj or {}
    return {
        "edited_anchor": safe_str(row.get("edited_anchor")) or safe_str(candidate.get("edited_anchor")),
        "fill_clause": safe_str(row.get("fill_clause")),
        "edit_instruction": safe_str(row.get("edit_instruction")) or safe_str(candidate.get("edit_instruction")),
        "route_reason": safe_str(row.get("route_reason")),
        "rationale": safe_str(candidate.get("rationale")),
        "rewrite_rationale": safe_str(rewrite_obj.get("rewrite_rationale")),
        "text_edit_operation": safe_str(rewrite_obj.get("text_edit_operation")) or safe_str(row.get("text_edit_operation")),
        "nli_direction": safe_str(rewrite_obj.get("nli_direction")) or safe_str(row.get("nli_direction")),
    }


def dedupe_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        pid = r["pair_id"]
        if pid not in best:
            best[pid] = r
            continue
        cur = best[pid]
        cur_score = clamp(safe_dict(cur.get("quality")).get("realism_score"), 0.0, 1.0)
        new_score = clamp(safe_dict(r.get("quality")).get("realism_score"), 0.0, 1.0)
        if new_score > cur_score:
            best[pid] = r
    return list(best.values())


def append_or_quarantine(kept: List[Dict[str, Any]], quarantined: List[Dict[str, Any]], example: Dict[str, Any], reasons: List[str]) -> None:
    if reasons:
        example["quality"]["filter_reason"] = ";".join(dict.fromkeys(reasons))
        quarantined.append(example)
    else:
        example["quality"]["filter_reason"] = "passed"
        kept.append(example)


def _heuristic_flagged(visual_meta: Dict[str, Any]) -> bool:
    """
    Return True if the *heuristic sub-check* (inside any manager mode) flagged
    the image as unacceptable.  This is checked independently of the manager's
    final confidence so that ghosting detections are never silently swallowed in
    hybrid mode when a VLM is absent.
    """
    for sub in safe_list(visual_meta.get("subchecks")):
        if safe_str(sub.get("backend")) == "heuristic" and sub.get("acceptable") is False:
            return True
    # Fallback: manager itself is in heuristic mode
    if visual_meta.get("mode") == "heuristic" and visual_meta.get("acceptable") is False:
        return True
    return False


def build_examples_from_edit_results(
    manifests: Dict[str, Dict[str, Any]],
    anchors_map: Dict[Tuple[str, str], Dict[str, Any]],
    plans_map: Dict[Tuple[str, str], Dict[str, Any]],
    edit_result_rows: Sequence[Dict[str, Any]],
    cfg: BuildConfig,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    kept, quarantined = [], []
    for row in edit_result_rows:
        sid = safe_str(row.get("sample_id"))
        aid = safe_str(row.get("anchor_id"))
        key = (sid, aid)
        plan_row = plans_map.get(key, {})
        anchor_row = anchors_map.get(key, {})
        manifest_row = manifests.get(sid)
        modality = detect_modality(row)
        requested_severity = safe_str(row.get("severity"))
        candidate_index = int(row.get("candidate_index") or 0)
        candidate = get_candidate_from_plan(plan_row, requested_severity, candidate_index)
        reasons: List[str] = []

        if cfg.require_edit_accepted and not row.get("accepted", False):
            reasons.append("edit_not_accepted")
        if safe_str(row.get("error")):
            reasons.append("edit_error")
        if clamp(row.get("realism_score", 0.0)) < cfg.realism_min:
            reasons.append("low_realism")
        changed_ratio = float(row.get("masked_changed_ratio") or 0.0)
        if changed_ratio < cfg.changed_ratio_min:
            reasons.append("changed_ratio_too_low")
        if changed_ratio > cfg.changed_ratio_max:
            reasons.append("changed_ratio_too_high")
        mean_delta = float(row.get("masked_mean_delta") or 0.0)
        if mean_delta < cfg.mean_delta_min:
            reasons.append("masked_delta_too_low")
        if cfg.exclude_sensitive and safe_dict(plan_row.get("sensitive_content_flag")).get("flagged"):
            reasons.append("sensitive_flagged")
        if cfg.require_anchor_valid and anchor_row and not anchor_row.get("technical_valid", True):
            reasons.append(f"anchor_invalid:{safe_str(anchor_row.get('technical_reject_reason')) or 'unknown'}")

        original_headline = resolve_original_text(manifest_row, plan_row)
        original_summary = resolve_summary(manifest_row, plan_row)
        rewrite_obj = resolve_rewrite_payload(
            candidate,
            "joint",
            allow_cross_mode_rewrite_fallback=cfg.allow_cross_mode_rewrite_fallback,
        )
        rewritten_caption = (
            safe_str(row.get("rewritten_caption"))
            or safe_str(rewrite_obj.get("rewritten_headline"))
        )
        if not rewritten_caption and cfg.synthesize_missing_rewrites and modality in {"image_text"}:
            rewritten_caption = synthesize_rewritten_headline(original_headline, candidate)
            if rewritten_caption:
                rewrite_obj = {
                    "rewritten_headline": rewritten_caption,
                    "text_edit_operation": safe_str(candidate.get("operation")),
                    "nli_direction": "",
                    "rewrite_rationale": "synthetic_fallback_from_builder",
                    "_rewrite_source": "synthetic",
                    "_synthetic": True,
                }
        if not rewritten_caption:
            rewritten_caption = original_headline

        edited_image = safe_str(row.get("output_path")) or safe_str(row.get("image_path"))
        img_ok, img_reason, img_meta = image_qc(edited_image)
        if not img_ok:
            reasons.append(img_reason)
        visual_meta = {"mode": cfg.visual_qc_mode, "checked": False, "acceptable": None, "confidence": 0.0, "reasons": [], "subchecks": []}
        if modality in {"image_only", "image_text"} and cfg.visual_judge is not None:
            human_focus = has_human_anchor(anchor_row, manifest_row)
            visual_meta = cfg.visual_judge.evaluate(edited_image, human_focus=human_focus)
            manager_rejects = (
                visual_meta.get("checked")
                and visual_meta.get("acceptable") is False
                and clamp(visual_meta.get("confidence", 0.0)) >= cfg.visual_qc_confidence_min
            )
            heuristic_rejects = cfg.visual_heuristic_hard_reject and _heuristic_flagged(visual_meta)
            if manager_rejects or heuristic_rejects:
                reasons.append("visual_defect_detected")
                for rr in safe_list(visual_meta.get("reasons")):
                    reasons.append(f"visual:{normalize_text(rr)[:64]}")

        if modality in {"text_only", "image_text"}:
            txt_ok, txt_reason, txt_meta = text_qc(
                rewritten_caption,
                original_headline,
                min_tokens=cfg.text_min_tokens,
                max_tokens=cfg.text_max_tokens,
                lexical_change_min=cfg.text_lexical_change_min,
            )
            if not txt_ok:
                reasons.append(txt_reason)
        else:
            txt_ok, txt_reason, txt_meta = text_qc(
                original_headline,
                min_tokens=cfg.text_min_tokens,
                max_tokens=cfg.text_max_tokens,
                lexical_change_min=cfg.text_lexical_change_min,
            )
            if not txt_ok:
                reasons.append(f"orig_{txt_reason}")

        components = resolve_severity_components_from_candidate(
            candidate,
            plan_row,
            fallback_tier=safe_str(row.get("tier")),
            nli_direction=safe_str(row.get("nli_direction")) or safe_str(rewrite_obj.get("nli_direction")),
        )
        raw_score = compute_severity_score(components)
        sev_score = apply_calibration(raw_score, cfg.calibration)
        if cfg.none_max is not None and sev_score < cfg.none_max:
            reasons.append("severity_below_meaningful_impact")
        sev_label = bucket_from_score(sev_score, cfg.low_max, cfg.medium_max)

        explanation = base_explanation(candidate, rewrite_obj, row)
        pair_id = make_pair_id(
            sid=sid, aid=aid, modality=modality,
            requested_op=safe_str(row.get("requested_operation")),
            effective_op=safe_str(row.get("effective_operation")) or safe_str(row.get("requested_operation")),
            requested_severity=requested_severity,
            final_severity=sev_label,
            candidate_index=candidate_index,
            headline_edited=rewritten_caption if modality in {"text_only", "image_text"} else original_headline,
            image_edited=edited_image if modality in {"image_only", "image_text"} else safe_str(row.get("image_path")),
        )
        example = {
            "pair_id": pair_id,
            "sample_id": sid,
            "anchor_id": aid,
            "authenticity_label": "manipulated",
            "is_pristine": False,
            "modality": modality,
            "severity_model_version": cfg.severity_model_version,
            "severity_label": sev_label,
            "severity_score_raw": round(raw_score, 6),
            "severity_score": round(sev_score, 6),
            "severity_components": {k: round(v, 6) for k, v in components.items()},
            "requested_generation_severity": requested_severity,
            "edit_type_requested": safe_str(row.get("requested_operation")),
            "edit_type_effective": safe_str(row.get("effective_operation")) or safe_str(row.get("requested_operation")),
            "headline_original": original_headline,
            "headline_edited": rewritten_caption if modality in {"text_only", "image_text"} else original_headline,
            "summary_original": original_summary,
            "summary_edited": original_summary,
            "image_original": safe_str(row.get("image_path")),
            "image_edited": edited_image if modality in {"image_only", "image_text"} else safe_str(row.get("image_path")),
            "mask_path": safe_str(row.get("union_mask_path")) or safe_str(anchor_row.get("union_mask_path")),
            "bbox_xyxy": row.get("bbox_xyxy") or anchor_row.get("bbox_xyxy"),
            "explanation": explanation,
            "quality": {
                "realism_score": clamp(row.get("realism_score", 0.0)),
                "masked_changed_ratio": round(changed_ratio, 6),
                "masked_mean_delta": round(mean_delta, 6),
                "image_qc": img_meta,
                "text_qc": txt_meta,
                "visual_qc": visual_meta,
                "accepted": bool(row.get("accepted", False)),
                "reject_reason": safe_str(row.get("reject_reason")),
            },
            "provenance": {
                "manifest_present": manifest_row is not None,
                "anchor_row_present": bool(anchor_row),
                "planner_row_present": bool(plan_row),
                "edit_result_present": True,
                "sensitive_flagged": bool(safe_dict(plan_row.get("sensitive_content_flag")).get("flagged")),
                "candidate_index": candidate_index,
                "rewrite_source": safe_str(rewrite_obj.get("_rewrite_source")) or "row_or_original",
                "synthetic_rewrite": bool(rewrite_obj.get("_synthetic")),
            },
        }
        append_or_quarantine(kept, quarantined, example, reasons)
    return kept, quarantined


def build_text_only_examples(
    manifests: Dict[str, Dict[str, Any]],
    anchors_map: Dict[Tuple[str, str], Dict[str, Any]],
    plans_rows: Sequence[Dict[str, Any]],
    cfg: BuildConfig,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    kept, quarantined = [], []
    for plan_row in plans_rows:
        sid = safe_str(plan_row.get("sample_id"))
        aid = safe_str(plan_row.get("anchor_id"))
        anchor_row = anchors_map.get((sid, aid), {})
        manifest_row = manifests.get(sid)
        original_headline = resolve_original_text(manifest_row, plan_row)
        original_summary = resolve_summary(manifest_row, plan_row)
        base_image = safe_str(plan_row.get("image_path")) or safe_str(manifest_row.get("image_path") if manifest_row else "")
        img_ok, img_reason, img_meta = image_qc(base_image)
        planner_output = safe_dict(plan_row.get("planner_output"))
        sev_map = safe_dict(planner_output.get("severity_candidates"))

        for requested_severity, cands in sev_map.items():
            for candidate_index, candidate in enumerate(safe_list(cands)):
                if not isinstance(candidate, dict):
                    continue

                variants = collect_text_only_rewrite_variants(candidate, original_headline, cfg)
                if not variants:
                    variants = [{"rewritten_headline": "", "_rewrite_source": "missing"}]

                for variant_index, rewrite_obj in enumerate(variants):
                    rewritten = safe_str(rewrite_obj.get("rewritten_headline"))
                    reasons: List[str] = []
                    if not rewritten:
                        reasons.append("missing_text_only_rewrite")
                    if safe_str(rewrite_obj.get("error")):
                        reasons.append("rewrite_error")
                    if cfg.exclude_sensitive and safe_dict(plan_row.get("sensitive_content_flag")).get("flagged"):
                        reasons.append("sensitive_flagged")
                    if cfg.require_anchor_valid and anchor_row and not anchor_row.get("technical_valid", True):
                        reasons.append(f"anchor_invalid:{safe_str(anchor_row.get('technical_reject_reason')) or 'unknown'}")
                    if not img_ok:
                        reasons.append(img_reason)

                    txt_ok, txt_reason, txt_meta = text_qc(
                        rewritten,
                        original_headline,
                        min_tokens=cfg.text_min_tokens,
                        max_tokens=cfg.text_max_tokens,
                        lexical_change_min=cfg.text_lexical_change_min,
                    )
                    if not txt_ok:
                        reasons.append(txt_reason)

                    nli_direction = safe_str(rewrite_obj.get("nli_direction"))
                    components = resolve_severity_components_from_candidate(
                        candidate,
                        plan_row,
                        fallback_tier=safe_str(safe_dict(plan_row.get("planner_input")).get("tier_name")),
                        nli_direction=nli_direction,
                    )
                    raw_score = compute_severity_score(components)
                    sev_score = apply_calibration(raw_score, cfg.calibration)
                    if cfg.none_max is not None and sev_score < cfg.none_max:
                        reasons.append("severity_below_meaningful_impact")
                    sev_label = bucket_from_score(sev_score, cfg.low_max, cfg.medium_max)

                    explanation = base_explanation(candidate, rewrite_obj, None)
                    pair_id = make_pair_id(
                        sid=sid, aid=aid, modality="text_only",
                        requested_op=safe_str(candidate.get("operation")),
                        effective_op=safe_str(candidate.get("operation")),
                        requested_severity=requested_severity,
                        final_severity=sev_label,
                        candidate_index=(candidate_index * 10) + variant_index,
                        headline_edited=rewritten,
                        image_edited=base_image,
                    )
                    example = {
                        "pair_id": pair_id,
                        "sample_id": sid,
                        "anchor_id": aid,
                        "authenticity_label": "manipulated",
                        "is_pristine": False,
                        "modality": "text_only",
                        "severity_model_version": cfg.severity_model_version,
                        "severity_label": sev_label,
                        "severity_score_raw": round(raw_score, 6),
                        "severity_score": round(sev_score, 6),
                        "severity_components": {k: round(v, 6) for k, v in components.items()},
                        "requested_generation_severity": requested_severity,
                        "edit_type_requested": safe_str(candidate.get("operation")),
                        "edit_type_effective": safe_str(candidate.get("operation")),
                        "headline_original": original_headline,
                        "headline_edited": rewritten or original_headline,
                        "summary_original": original_summary,
                        "summary_edited": original_summary,
                        "image_original": base_image,
                        "image_edited": base_image,
                        "mask_path": safe_str(safe_dict(plan_row.get("planner_input")).get("union_mask_path")) or safe_str(anchor_row.get("union_mask_path")),
                        "bbox_xyxy": safe_dict(plan_row.get("planner_input")).get("bbox_xyxy") or anchor_row.get("bbox_xyxy"),
                        "explanation": explanation,
                        "quality": {
                            "realism_score": None,
                            "masked_changed_ratio": None,
                            "masked_mean_delta": None,
                            "image_qc": img_meta,
                            "text_qc": txt_meta,
                        },
                        "provenance": {
                            "manifest_present": manifest_row is not None,
                            "anchor_row_present": bool(anchor_row),
                            "planner_row_present": True,
                            "edit_result_present": False,
                            "sensitive_flagged": bool(safe_dict(plan_row.get("sensitive_content_flag")).get("flagged")),
                            "candidate_index": candidate_index,
                            "rewrite_variant_index": variant_index,
                            "rewrite_source": safe_str(rewrite_obj.get("_rewrite_source")) or "text_only",
                            "synthetic_rewrite": bool(rewrite_obj.get("_synthetic")),
                        },
                    }
                    append_or_quarantine(kept, quarantined, example, reasons)
    return kept, quarantined


def sample_items(items: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    if len(items) <= n:
        return list(items)
    rnd = random.Random(seed)
    items = list(items)
    rnd.shuffle(items)
    return items[:n]


def balance_manipulated(items: List[Dict[str, Any]], seed: int, balance_edit_type_within_cell: bool) -> List[Dict[str, Any]]:
    if not items:
        return items
    by_cell: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for ex in items:
        by_cell[(ex["modality"], ex["severity_label"])].append(ex)
    filtered_cells: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for cell, cell_items in by_cell.items():
        if balance_edit_type_within_cell:
            by_edit: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for ex in cell_items:
                by_edit[ex["edit_type_effective"] or "unknown"].append(ex)
            target_per_edit = min(len(v) for v in by_edit.values())
            cell_balanced: List[Dict[str, Any]] = []
            for i, (_, exs) in enumerate(sorted(by_edit.items())):
                cell_balanced.extend(sample_items(exs, target_per_edit, seed + i + 101))
            filtered_cells[cell] = cell_balanced
        else:
            filtered_cells[cell] = list(cell_items)
    target_per_cell = min(len(v) for v in filtered_cells.values())
    out: List[Dict[str, Any]] = []
    for i, (_, exs) in enumerate(sorted(filtered_cells.items())):
        out.extend(sample_items(exs, target_per_cell, seed + i + 137))
    return out


def maybe_cap_pristine(pristine: List[Dict[str, Any]], max_pristine: Optional[int], seed: int) -> List[Dict[str, Any]]:
    if max_pristine is None:
        return pristine
    return sample_items(pristine, max_pristine, seed)


def assign_splits(items: List[Dict[str, Any]], split_tuple: Tuple[float, float, float]) -> None:
    train, val, _ = split_tuple
    for ex in items:
        ex["split"] = split_from_sample_id(ex["sample_id"], train, val)


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _flat_row(r: Dict[str, Any], rich: bool = False) -> Dict[str, Any]:
    base = {
        "pair_id": r["pair_id"],
        "sample_id": r["sample_id"],
        "anchor_id": r.get("anchor_id"),
        "split": r.get("split"),
        "authenticity_label": r["authenticity_label"],
        "modality": r["modality"],
        "requested_generation_severity": r.get("requested_generation_severity"),
        "severity_label": r.get("severity_label"),
        "severity_score": r.get("severity_score"),
        "severity_score_raw": r.get("severity_score_raw"),
        "edit_type_requested": r.get("edit_type_requested"),
        "edit_type_effective": r.get("edit_type_effective"),
        "image_original": r.get("image_original"),
        "image_edited": r.get("image_edited"),
        "headline_original": r.get("headline_original"),
        "headline_edited": r.get("headline_edited"),
        "mask_path": r.get("mask_path"),
    }
    if not rich:
        return base
    ex = safe_dict(r.get("explanation"))
    q = safe_dict(r.get("quality"))
    prov = safe_dict(r.get("provenance"))
    comps = safe_dict(r.get("severity_components"))
    base.update({
        "bbox_xyxy": json.dumps(r.get("bbox_xyxy"), ensure_ascii=False),
        "edited_anchor": ex.get("edited_anchor"),
        "fill_clause": ex.get("fill_clause"),
        "edit_instruction": ex.get("edit_instruction"),
        "route_reason": ex.get("route_reason"),
        "rationale": ex.get("rationale"),
        "rewrite_rationale": ex.get("rewrite_rationale"),
        "text_edit_operation": ex.get("text_edit_operation"),
        "nli_direction": ex.get("nli_direction"),
        "T_obj": comps.get("T_obj"),
        "I_ctx": comps.get("I_ctx"),
        "C_nli": comps.get("C_nli"),
        "Sal": comps.get("Sal"),
        "Vis": comps.get("Vis"),
        "realism_score": q.get("realism_score"),
        "masked_changed_ratio": q.get("masked_changed_ratio"),
        "masked_mean_delta": q.get("masked_mean_delta"),
        "visual_qc_mode": safe_dict(q.get("visual_qc")).get("mode"),
        "visual_qc_checked": safe_dict(q.get("visual_qc")).get("checked"),
        "visual_qc_acceptable": safe_dict(q.get("visual_qc")).get("acceptable"),
        "visual_qc_confidence": safe_dict(q.get("visual_qc")).get("confidence"),
        "visual_qc_reasons": json.dumps(safe_dict(q.get("visual_qc")).get("reasons"), ensure_ascii=False),
        "filter_reason": q.get("filter_reason"),
        "candidate_index": prov.get("candidate_index"),
        "manifest_present": prov.get("manifest_present"),
        "planner_row_present": prov.get("planner_row_present"),
        "edit_result_present": prov.get("edit_result_present"),
        "sensitive_flagged": prov.get("sensitive_flagged"),
    })
    return base


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], rich: bool = False) -> None:
    flat_rows = [_flat_row(r, rich=rich) for r in rows]
    if not flat_rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)


def compute_stats(all_rows: Sequence[Dict[str, Any]], quarantined: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    modality_counts = Counter(r["modality"] for r in all_rows)
    severity_counts = Counter(r["severity_label"] for r in all_rows if r["severity_label"])
    requested_sev_counts = Counter(r.get("requested_generation_severity") for r in all_rows if r.get("requested_generation_severity"))
    edit_counts = Counter(r["edit_type_effective"] for r in all_rows if r.get("edit_type_effective"))
    split_counts = Counter(r.get("split") for r in all_rows)
    q_reasons = Counter()
    req_to_final = Counter()
    req_to_eff = Counter()
    for r in all_rows:
        req = r.get("requested_generation_severity")
        fin = r.get("severity_label")
        if req and fin:
            req_to_final[f"{req}->{fin}"] += 1
        rop = r.get("edit_type_requested")
        eop = r.get("edit_type_effective")
        if rop and eop:
            req_to_eff[f"{rop}->{eop}"] += 1
    for r in quarantined:
        for reason in safe_str(safe_dict(r.get("quality")).get("filter_reason")).split(";"):
            if reason:
                q_reasons[reason] += 1
    sev_scores = [r["severity_score"] for r in all_rows if r.get("severity_score") is not None]
    return {
        "num_examples": len(all_rows),
        "num_quarantined": len(quarantined),
        "modality_counts": dict(sorted(modality_counts.items())),
        "severity_counts": dict(sorted(severity_counts.items())),
        "requested_generation_severity_counts": dict(sorted(requested_sev_counts.items())),
        "requested_to_final_severity": dict(sorted(req_to_final.items())),
        "edit_type_counts": dict(sorted(edit_counts.items())),
        "requested_to_effective_operation": dict(sorted(req_to_eff.items())),
        "modality_severity_counts": dict(sorted(Counter(f"{r.get('modality')}|{r.get('severity_label')}" for r in all_rows if r.get("severity_label")).items())),
        "split_counts": dict(sorted(split_counts.items())),
        "quarantine_reason_counts": dict(sorted(q_reasons.items())),
        "severity_score_summary": {
            "mean": round(mean(sev_scores), 6) if sev_scores else None,
            "min": round(min(sev_scores), 6) if sev_scores else None,
            "max": round(max(sev_scores), 6) if sev_scores else None,
        },
    }


def write_dataset_card(path: Path, stats: Dict[str, Any], args: argparse.Namespace) -> None:
    text = f"""# T-IMPACT final release

This release was built with `timpact_final_builder_100k.py`.

## What is included
- pristine pairs
- manipulated pairs with modality labels
- final severity scores and severity tiers
- explanation metadata in JSONL and rich CSV
- grounding and mask references when available
- quarantine file for excluded items
- requested-vs-final severity summary
- requested-vs-effective operation summary

## QC / scale policy used in this build
- minimum realism score: {args.realism_min}
- minimum masked changed ratio: {args.changed_ratio_min}
- maximum masked changed ratio: {args.changed_ratio_max}
- minimum masked mean delta: {args.mean_delta_min}
- require accepted edit flag: {args.require_edit_accepted}
- require anchor technical validity: {args.require_anchor_valid}
- exclude sensitive flagged edits: {not args.include_sensitive}
- severity model version: {getattr(args, "severity_model_version", "default_v1")}
- severity none cutoff (quarantine): {getattr(args, "none_max", None)}
- severity thresholds: low <= {args.low_max}, medium <= {args.medium_max}, else high
- balance manipulated data: {args.balance}
- balance edit type within modality-severity cells: {args.balance_edit_type_within_cell}
- visual QC mode: {args.visual_qc_mode}
- visual QC model: {args.visual_vlm_model or "none"}
- visual QC confidence threshold: {args.visual_qc_confidence_min}
- heuristic visual hard reject: {args.visual_heuristic_hard_reject}
- cross-mode rewrite fallback: {args.allow_cross_mode_rewrite_fallback}
- include joint rewrites as text_only variants: {args.include_joint_as_text_only}
- synthesize missing rewrites: {args.synthesize_missing_rewrites}
- text min tokens: {args.text_min_tokens}
- text max tokens: {args.text_max_tokens}
- text lexical change min: {args.text_lexical_change_min}

## Notes
This scale-oriented build is designed to recover more valid examples from the
existing planner and edit-result outputs without fabricating duplicate pairs.
It relaxes several hard rejections, especially around accepted flags, text
rewrite availability, and heuristic visual QC.

## Counts
```json
{json.dumps(stats, indent=2)}
```
"""
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build final T-IMPACT release (100k-oriented)")
    p.add_argument("--repo-root", required=True)
    p.add_argument("--manifest-dir", default=None)
    p.add_argument("--anchors-jsonl", nargs="*", default=[])
    p.add_argument("--edit-plans-glob", nargs="*", default=[])
    p.add_argument("--edit-results-glob", nargs="+", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--realism-min", type=float, default=0.35)
    p.add_argument("--changed-ratio-min", type=float, default=0.0025)
    p.add_argument("--changed-ratio-max", type=float, default=0.95)
    p.add_argument("--mean-delta-min", type=float, default=1.5)
    p.add_argument("--low-max", type=float, default=DEFAULT_THRESHOLDS["low_max"])
    p.add_argument("--medium-max", type=float, default=DEFAULT_THRESHOLDS["medium_max"])
    p.add_argument("--severity-model-json", default=None)
    p.add_argument("--calibration-json", default=None)
    p.add_argument("--visual-qc-mode", choices=["off", "heuristic", "vlm", "hybrid"], default="heuristic")
    p.add_argument("--visual-vlm-model", default="")
    p.add_argument("--visual-qc-confidence-min", type=float, default=0.85)
    p.add_argument("--visual-qc-human-only", action="store_true")
    p.add_argument("--visual-heuristic-hard-reject", action="store_true")
    p.add_argument("--balance", action="store_true")
    p.add_argument("--balance-edit-type-within-cell", action="store_true")
    p.add_argument("--max-pristine", type=int, default=None)
    p.add_argument("--include-sensitive", action="store_true")
    p.add_argument("--require-edit-accepted", action="store_true")
    p.add_argument("--require-anchor-valid", action="store_true")
    p.add_argument("--allow-cross-mode-rewrite-fallback", action="store_true", default=True)
    p.add_argument("--no-allow-cross-mode-rewrite-fallback", dest="allow_cross_mode_rewrite_fallback", action="store_false")
    p.add_argument("--include-joint-as-text-only", action="store_true", default=True)
    p.add_argument("--no-include-joint-as-text-only", dest="include_joint_as_text_only", action="store_false")
    p.add_argument("--synthesize-missing-rewrites", action="store_true")
    p.add_argument("--text-min-tokens", type=int, default=3)
    p.add_argument("--text-max-tokens", type=int, default=80)
    p.add_argument("--text-lexical-change-min", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=13)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    ensure_dir(output_root)
    ensure_dir(output_root / "splits")
    calibration = parse_calibration_map(args.calibration_json)
    severity_model = parse_severity_model(args.severity_model_json)
    if severity_model:
        for k, v in safe_dict(severity_model.get("weights")).items():
            SEVERITY_WEIGHTS[k] = clamp(v)
        for k, v in safe_dict(severity_model.get("tier_priors")).items():
            DEFAULT_TIER_PRIOR[k] = clamp(v)
        thr = safe_dict(severity_model.get("thresholds"))
        if thr.get("low_max") is not None:
            args.low_max = clamp(thr.get("low_max"))
        if thr.get("medium_max") is not None:
            args.medium_max = clamp(thr.get("medium_max"))
        args.none_max = clamp(thr.get("none_max")) if thr.get("none_max") is not None else None
        if calibration is None and severity_model.get("calibration"):
            calibration = severity_model.get("calibration")
    else:
        args.none_max = None
    args.severity_model_version = safe_str(severity_model.get("model_version")) if severity_model else "default_v1"
    visual_judge = None
    if args.visual_qc_mode != "off":
        visual_judge = VisualQCManager(
            mode=args.visual_qc_mode,
            vlm_model=args.visual_vlm_model,
            confidence_min=args.visual_qc_confidence_min,
            human_only=args.visual_qc_human_only,
        )
    cfg = BuildConfig(
        repo_root=Path(args.repo_root),
        output_root=output_root,
        realism_min=args.realism_min,
        changed_ratio_min=args.changed_ratio_min,
        changed_ratio_max=args.changed_ratio_max,
        mean_delta_min=args.mean_delta_min,
        low_max=args.low_max,
        medium_max=args.medium_max,
        none_max=args.none_max,
        severity_model_version=args.severity_model_version,
        exclude_sensitive=not args.include_sensitive,
        calibration=calibration,
        visual_qc_mode=args.visual_qc_mode,
        visual_qc_confidence_min=args.visual_qc_confidence_min,
        require_edit_accepted=args.require_edit_accepted,
        require_anchor_valid=args.require_anchor_valid,
        allow_cross_mode_rewrite_fallback=args.allow_cross_mode_rewrite_fallback,
        include_joint_as_text_only=args.include_joint_as_text_only,
        synthesize_missing_rewrites=args.synthesize_missing_rewrites,
        text_min_tokens=args.text_min_tokens,
        text_max_tokens=args.text_max_tokens,
        text_lexical_change_min=args.text_lexical_change_min,
        visual_heuristic_hard_reject=args.visual_heuristic_hard_reject,
        visual_judge=visual_judge,
    )

    manifests = load_manifests(args.manifest_dir)
    anchors_map = load_jsonl_map(args.anchors_jsonl, ("sample_id", "anchor_id")) if args.anchors_jsonl else {}
    plans_rows = load_jsonl_rows(args.edit_plans_glob) if args.edit_plans_glob else []
    plans_map = {(safe_str(r.get("sample_id")), safe_str(r.get("anchor_id"))): r for r in plans_rows if safe_str(r.get("sample_id")) and safe_str(r.get("anchor_id"))}
    edit_rows = load_jsonl_rows(args.edit_results_glob)

    pristine = maybe_cap_pristine(collect_pristine_examples(manifests), args.max_pristine, args.seed)
    kept_img, q_img = build_examples_from_edit_results(manifests, anchors_map, plans_map, edit_rows, cfg)
    kept_txt, q_txt = build_text_only_examples(manifests, anchors_map, plans_rows, cfg)

    manipulated = dedupe_rows(kept_img + kept_txt)
    quarantined = dedupe_rows(q_img + q_txt)

    if args.balance:
        manipulated = balance_manipulated(manipulated, seed=args.seed, balance_edit_type_within_cell=args.balance_edit_type_within_cell)

    all_rows = pristine + manipulated
    assign_splits(all_rows, DEFAULT_SPLITS)
    assign_splits(quarantined, DEFAULT_SPLITS)

    write_jsonl(output_root / "all_examples.jsonl", all_rows)
    write_jsonl(output_root / "quarantine.jsonl", quarantined)
    write_csv(output_root / "metadata.csv", all_rows, rich=False)
    write_csv(output_root / "metadata_rich.csv", all_rows, rich=True)

    for split in ["train", "val", "test"]:
        rows = [r for r in all_rows if r.get("split") == split]
        write_jsonl(output_root / "splits" / f"{split}.jsonl", rows)

    stats = compute_stats(all_rows, quarantined)
    (output_root / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    write_dataset_card(output_root / "DATASET_CARD.md", stats, args)
    print("[DONE] Wrote final release to", output_root)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()