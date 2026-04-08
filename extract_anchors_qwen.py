#!/usr/bin/env python3
"""
extract_anchors_qwen.py

Step 2: Use Qwen3-VL to extract:
- image-only literal description (caption)
- text-only article context rewrite (from CSV title+summary)
- localized anchors: objects/regions + attributes + relations

Input:
  - CSV:  timpact/metadata/pristine/*.csv  (auto-picks newest unless --csv given)
    Expected columns (best-effort): title, summary, published, url, images
  - Images: timpact/data/pristine_images/images/images/{images_filename}

Output:
  - timpact/data/anchors_qwen3.json

Recommended install on Rangpur:
  pip install -U git+https://github.com/huggingface/transformers accelerate
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


# ----------------------------
# Prompts
# ----------------------------

CAPTION_PROMPT = """You are describing the image ONLY.

Return ONLY one valid JSON object with these keys:
{
  "image_caption_literal": "1-2 sentence literal description. No speculation.",
  "visible_text": "Concise string with clearly readable text from the image; else empty string.",
  "people_count": 0,
  "notable_objects": ["..."],
  "visible_text_items": ["..."],
  "visible_regions": ["..."],
  "candidate_objects_all": ["..."]
}

Rules:
- Focus only on what is visibly present.
- Do not guess identities, locations, dates, events, jobs, emotions, or intentions.
- Prefer literal visual description over interpretation.
- "notable_objects" should contain the main visible objects/entities.
- "candidate_objects_all" should be broader and include smaller but still visible/editable items such as signs, logos, badges, bags, papers, screens, plaques, handheld objects, clothing items, microphones, food items, flags, helmets, tools, stickers, etc., if present.
- "visible_regions" can include meaningful scene regions such as "brick wall", "glass window", "wooden podium front", "street background", "grass patch".
- "visible_text_items" should break out multiple readable text snippets when possible.
- If uncertain, omit details rather than speculate.
- Output JSON only. No markdown. No explanation.
"""

CONTEXT_PROMPT_TEMPLATE = """You are rewriting article context using ONLY the provided title and summary.

Return ONLY one valid JSON object with exactly this schema:
{{
  "article_context_rewrite": "2-3 neutral sentences based only on title and summary."
}}

Rules:
- Do not invent facts beyond the title and summary.
- Keep proper nouns if provided.
- Stay neutral and concise.
- Output JSON only. No markdown. No explanation.

TITLE:
{title}

SUMMARY:
{summary}
"""

ANCHORS_PROMPT_TEMPLATE = """You are extracting localized visual anchors from the image, using the provided image caption, visible text, and article context only as supporting context.

Return ONLY one valid JSON object with exactly these keys:
{{
  "grounded_context_caption": "1 sentence linking visible image content to article context WITHOUT adding new facts. If linkage is unclear, say 'No clear grounding.'",
  "anchors": [
    {{
      "anchor_id": "a1",
      "span": "specific short noun phrase",
      "semantic_tags_raw": ["person", "clothing"],
      "semantic_tags": ["person"],
      "functional_role": "literal visible role in scene",
      "localization": {{
        "level": "object",
        "region_hint": "left"
      }},
      "attributes": ["attribute1"],
      "relations": [
        {{"type": "next_to", "target_anchor_id": "a2"}}
      ],
      "visibility": "full",
      "salience": "high",
      "confidence": "high"
    }}
  ]
}}

Rules:
- Extract visually grounded anchors from the image even if article linkage is unclear.
- If article linkage is unclear, set grounded_context_caption to "No clear grounding." but STILL provide visual anchors.
- Prefer specific, editable, visually distinct anchors over abstract summaries.
- Use literal visible roles only, such as: "main person", "clothing item", "mounted sign", "background vegetation", "display device", "text element", "door hardware", "scene region", "furniture item", "device".
- Do NOT use interpretive roles like journalist, mail carrier, observer, politician, distressed person, company identifier, protester, worker, official, etc., unless those are directly and unambiguously visible in the image text itself.
- Include people, clothing, accessories, carried objects, logos, readable signs, labels, plaques, podiums, doors, windows, vehicles, plants, furniture, food items, screens, posters, flags, tools, microphones, banners, products, and small salient objects when visible.
- Include small but visible items if they could plausibly matter for later editing, but do not invent uncertain details.
- If visible text exists, include text anchors in modular form. Prefer "Departures", "Cancelled", "0609 Cambridge" instead of one overly long combined line when possible.
- Avoid vague spans like "scene", "background", "area", "object", "person" unless no more specific phrase is possible. Prefer phrases like "blue tie", "red sign", "brass plaque", "microphone", "green backpack", "front window", "white helmet".
- For people wearing visible items, decompose when possible:
  - include the person anchor
  - include separate anchors for visible clothing/accessories like tie, jacket, cap, helmet, bag, badge, glasses
  - connect them with relations like "wearing" or "on" when confident
- For text screens/signboards, include both:
  - the screen/board/device
  - smaller text units such as headers, labels, rows, or visible text blocks
- Do not invent items that are not visibly present.
- Do not anchor unseen claims from the article.
- Provide 6-15 anchors when justified by the image. If the image is sparse, provide fewer, but always provide at least 1 anchor whenever any visible object/person/text/region can be identified.
- Use coarse region hints only: left, right, center, top-left, top-right, bottom-left, bottom-right, foreground, background.
- Keep semantic_tags_raw as fine-grained as possible.
- semantic_tags must use only these normalized tags:
  ["person", "clothing", "accessory", "sign_text", "structure", "vehicle", "plant", "furniture", "device", "food", "animal", "object"]
- Use the most specific normalized tag possible. Do not add "object" if a more specific normalized tag already applies.
- Set visibility to one of: full, partial, small, blurry.
- Set salience to one of: high, medium, low.
- Set confidence to one of: high, medium, low.
- Relations must reference existing anchor_ids.
- If relations are uncertain, use an empty list.
- Output JSON only. No markdown. No explanation.

IMAGE_CAPTION_LITERAL:
{caption}

VISIBLE_TEXT_IN_IMAGE:
{visible_text}

ARTICLE_CONTEXT:
{context}
"""

MISSED_ANCHORS_PROMPT_TEMPLATE = """You are reviewing an image and a current anchor list.

Your task is to identify additional visible anchors that were missed in the current anchor list.

Return ONLY one valid JSON object with exactly this schema:
{{
  "missed_anchors": [
    {{
      "span": "specific short noun phrase",
      "semantic_tags_raw": ["logo"],
      "semantic_tags": ["sign_text"],
      "functional_role": "literal visible role in scene",
      "localization": {{
        "level": "object",
        "region_hint": "right"
      }},
      "attributes": ["attribute1"],
      "visibility": "small",
      "salience": "medium",
      "confidence": "medium"
    }}
  ]
}}

Rules:
- Only propose anchors that are visibly present in the image.
- Do NOT repeat or paraphrase anchors already present in the current list.
- Focus especially on smaller missed items, signs, logos, text regions, carried objects, clothing items, accessories, handheld objects, foreground objects, and meaningful scene regions.
- Prefer literal visible roles only.
- Avoid vague spans like "scene", "area", "object", "background", "surface".
- Avoid interpretive additions.
- Use only these normalized tags for semantic_tags:
  ["person", "clothing", "accessory", "sign_text", "structure", "vehicle", "plant", "furniture", "device", "food", "animal", "object"]
- Provide up to 8 missed anchors if justified; return an empty list if nothing useful is missing.
- Output JSON only. No markdown. No explanation.

CURRENT_ANCHORS:
{current_anchors_json}
"""

JSON_REPAIR_PROMPT_TEMPLATE = """You are repairing malformed model output into valid JSON.

Return ONLY one valid JSON object following this target schema exactly:

{schema}

Malformed output to repair:
{raw}
"""


# ----------------------------
# Utilities
# ----------------------------

NORMALIZED_TAGS = {
    "person",
    "clothing",
    "accessory",
    "sign_text",
    "structure",
    "vehicle",
    "plant",
    "furniture",
    "device",
    "food",
    "animal",
    "object",
}

GENERIC_SPANS = {
    "object",
    "objects",
    "thing",
    "things",
    "item",
    "items",
    "stuff",
    "scene",
    "area",
    "region",
    "surface",
    "background",
    "foreground",
    "person",
    "people",
    "human",
    "structure",
    "text",
    "sign",
    "building",
}

INTERPRETIVE_ROLE_HINTS = {
    "mail carrier",
    "observer",
    "company identifier",
    "action indicating distress",
    "distressed person",
    "politician",
    "official",
    "worker",
    "journalist",
    "protester",
}

RELATION_TYPES_ALLOWLIST = {
    "next_to",
    "behind",
    "in_front_of",
    "on",
    "under",
    "holding",
    "carrying",
    "wearing",
    "attached_to",
    "inside",
    "near",
    "beside",
    "above",
    "below",
    "overlapping",
}

REGION_HINTS_ALLOWLIST = {
    "left",
    "right",
    "center",
    "top-left",
    "top-right",
    "bottom-left",
    "bottom-right",
    "foreground",
    "background",
}

VISIBILITY_ALLOWLIST = {"full", "partial", "small", "blurry"}
THREE_LEVEL_ALLOWLIST = {"high", "medium", "low"}


def now_iso() -> str:
    return dt.datetime.now().replace(microsecond=0).isoformat()


def pick_newest_csv(folder: Path) -> Path:
    cands = sorted(folder.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(f"No CSV files found in: {folder}")
    return cands[0]


def write_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def load_existing(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def decode_weird_utf8(s: str) -> str:
    if not s:
        return s
    try:
        repaired = s.encode("latin1").decode("utf-8")
        return repaired
    except Exception:
        return s


def strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def safe_json_loads(s: str) -> Any:
    s = strip_fences(s)

    try:
        return json.loads(s)
    except Exception:
        pass

    m = re.search(r"(\{.*\}|\[.*\])", s, flags=re.DOTALL)
    if m:
        candidate = m.group(1)
        return json.loads(candidate)

    raise ValueError("json_parse_failed:JSONDecodeError")


def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def normalize_span_text(s: str) -> str:
    s = normalize_space(s).lower()
    s = re.sub(r"^[\-\–\—\:\;\,\.\s]+", "", s)
    s = re.sub(r"[\-\–\—\:\;\,\.\s]+$", "", s)
    s = re.sub(r"\b(the|a|an)\b\s+", "", s)
    s = normalize_space(s)
    return s


def is_generic_span(s: str) -> bool:
    t = normalize_span_text(s)
    if not t:
        return True
    if t in GENERIC_SPANS:
        return True
    if len(t) <= 2:
        return True
    return False


def clean_string_list(xs: Any, dedupe_lower: bool = True) -> List[str]:
    if not isinstance(xs, list):
        return []
    out = []
    seen = set()
    for x in xs:
        if not isinstance(x, str):
            continue
        v = normalize_space(x)
        if not v:
            continue
        key = v.lower() if dedupe_lower else v
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


def infer_broad_tag_from_span(span: str) -> str:
    lower = normalize_span_text(span)

    if any(word in lower for word in ["man", "woman", "boy", "girl", "person", "people", "human", "face", "hand", "arm", "head"]):
        return "person"
    if any(word in lower for word in ["shirt", "jacket", "coat", "dress", "trousers", "pants", "uniform", "hoodie", "sweater"]):
        return "clothing"
    if any(word in lower for word in ["tie", "cap", "hat", "helmet", "glasses", "bag", "backpack", "badge", "watch", "scarf"]):
        return "accessory"
    if any(word in lower for word in ["logo", "label", "sign", "text", "plaque", "banner", "poster", "board_text", "display_text"]):
        return "sign_text"
    if any(word in lower for word in ["screen", "display", "monitor", "phone", "laptop", "microphone", "speaker", "camera", "board"]):
        return "device"
    if any(word in lower for word in ["door", "wall", "window", "building", "doorway", "facade", "podium", "fence", "slot", "letterbox", "hardware"]):
        return "structure"
    if any(word in lower for word in ["car", "truck", "bike", "bicycle", "motorcycle", "bus", "van", "train"]):
        return "vehicle"
    if any(word in lower for word in ["tree", "bush", "bushes", "grass", "plant", "plants", "flower", "leaf", "foliage"]):
        return "plant"
    if any(word in lower for word in ["couch", "chair", "table", "bench", "bed", "sofa", "desk"]):
        return "furniture"
    if any(word in lower for word in ["food", "meal", "plate", "bread", "fruit", "drink", "cup", "bottle"]):
        return "food"
    if any(word in lower for word in ["dog", "cat", "bird", "horse", "animal"]):
        return "animal"

    return "object"


def normalize_semantic_tags(anchor_list: List[Dict[str, Any]]) -> None:
    def map_tag(tag: str) -> str:
        t = normalize_span_text(tag)

        if t in {"person", "people", "human", "man", "woman", "boy", "girl", "face", "body_part"}:
            return "person"
        if t in {"shirt", "jacket", "coat", "dress", "trousers", "pants", "uniform", "hoodie", "sweater", "clothing"}:
            return "clothing"
        if t in {"tie", "cap", "hat", "helmet", "glasses", "bag", "backpack", "badge", "watch", "scarf", "accessory", "headwear"}:
            return "accessory"
        if t in {"sign_text", "text", "logo", "label", "plaque", "sign", "banner", "poster", "board_text", "display_text"}:
            return "sign_text"
        if t in {"screen", "display", "monitor", "phone", "laptop", "microphone", "speaker", "camera", "board", "device"}:
            return "device"
        if t in {"structure", "door", "wall", "window", "building", "doorway", "facade", "podium", "fence", "slot", "letterbox", "hardware"}:
            return "structure"
        if t in {"vehicle", "car", "truck", "bike", "bicycle", "motorcycle", "bus", "van", "train"}:
            return "vehicle"
        if t in {"plant", "tree", "bush", "bushes", "grass", "vegetation", "flower", "leaf", "foliage"}:
            return "plant"
        if t in {"furniture", "couch", "chair", "table", "bench", "bed", "sofa", "desk"}:
            return "furniture"
        if t in {"food", "meal", "plate", "bread", "fruit", "drink", "cup", "bottle"}:
            return "food"
        if t in {"animal", "dog", "cat", "bird", "horse"}:
            return "animal"
        return "object"

    for anchor in anchor_list:
        raw_tags = clean_string_list(anchor.get("semantic_tags_raw", []))
        coarse_tags_in = clean_string_list(anchor.get("semantic_tags", []))

        if not raw_tags and coarse_tags_in:
            raw_tags = coarse_tags_in[:]

        if not raw_tags:
            raw_tags = [infer_broad_tag_from_span(anchor.get("span", ""))]

        mapped = []
        for tag in raw_tags + coarse_tags_in:
            mapped_tag = map_tag(tag)
            if mapped_tag in NORMALIZED_TAGS:
                mapped.append(mapped_tag)

        if not mapped:
            mapped = [infer_broad_tag_from_span(anchor.get("span", ""))]

        deduped_mapped = []
        seen = set()
        for tag in mapped:
            if tag not in seen:
                deduped_mapped.append(tag)
                seen.add(tag)

        if len(deduped_mapped) > 1 and "object" in deduped_mapped:
            deduped_mapped = [t for t in deduped_mapped if t != "object"]

        anchor["semantic_tags_raw"] = raw_tags
        anchor["semantic_tags"] = deduped_mapped or ["object"]


def validate_caption_payload(obj: Any) -> None:
    if not isinstance(obj, dict):
        raise ValueError("caption_not_dict")
    for key in ["image_caption_literal", "visible_text", "people_count", "notable_objects"]:
        if key not in obj:
            raise ValueError(f"caption_missing_key:{key}")
    if not isinstance(obj["image_caption_literal"], str):
        raise ValueError("caption_invalid_image_caption_literal")
    if not isinstance(obj["visible_text"], str):
        raise ValueError("caption_invalid_visible_text")
    if not isinstance(obj["people_count"], int):
        raise ValueError("caption_invalid_people_count")
    if not isinstance(obj["notable_objects"], list):
        raise ValueError("caption_invalid_notable_objects")
    if "visible_text_items" in obj and not isinstance(obj["visible_text_items"], list):
        raise ValueError("caption_invalid_visible_text_items")
    if "visible_regions" in obj and not isinstance(obj["visible_regions"], list):
        raise ValueError("caption_invalid_visible_regions")
    if "candidate_objects_all" in obj and not isinstance(obj["candidate_objects_all"], list):
        raise ValueError("caption_invalid_candidate_objects_all")


def validate_context_payload(obj: Any) -> None:
    if not isinstance(obj, dict):
        raise ValueError("context_not_dict")
    if "article_context_rewrite" not in obj:
        raise ValueError("context_missing_article_context_rewrite")
    if not isinstance(obj["article_context_rewrite"], str):
        raise ValueError("context_invalid_article_context_rewrite")


def normalize_relations(anchor_list: List[Dict[str, Any]]) -> None:
    valid_ids = {
        a.get("anchor_id")
        for a in anchor_list
        if isinstance(a, dict) and isinstance(a.get("anchor_id"), str) and a.get("anchor_id")
    }

    for anchor in anchor_list:
        rels = anchor.get("relations", [])
        if not isinstance(rels, list):
            anchor["relations"] = []
            continue

        cleaned = []
        seen = set()
        for rel in rels:
            if not isinstance(rel, dict):
                continue
            target = rel.get("target_anchor_id")
            rtype = rel.get("type")
            if not (isinstance(target, str) and target in valid_ids and isinstance(rtype, str) and rtype):
                continue

            rtype_norm = normalize_span_text(rtype).replace(" ", "_")
            if rtype_norm not in RELATION_TYPES_ALLOWLIST:
                continue

            key = (rtype_norm, target)
            if key in seen:
                continue
            seen.add(key)
            cleaned.append({"type": rtype_norm, "target_anchor_id": target})

        anchor["relations"] = cleaned


def normalize_functional_role(role: str, span: str, tags: List[str]) -> str:
    r = normalize_space(role).lower()
    s = normalize_span_text(span)
    tags_norm = {normalize_span_text(t) for t in tags}

    if not r or r in INTERPRETIVE_ROLE_HINTS:
        if "person" in tags_norm:
            if any(w in s for w in ["hand", "arm", "head", "face"]):
                return "person detail"
            return "main person"
        if "clothing" in tags_norm:
            return "clothing item"
        if "accessory" in tags_norm:
            return "accessory"
        if "sign_text" in tags_norm:
            return "text element"
        if "device" in tags_norm:
            return "device"
        if "structure" in tags_norm:
            return "scene structure"
        if "plant" in tags_norm:
            return "background vegetation"
        if "vehicle" in tags_norm:
            return "vehicle"
        if "furniture" in tags_norm:
            return "furniture item"
        if "food" in tags_norm:
            return "food item"
        if "animal" in tags_norm:
            return "animal"
        return "visible scene element"

    if "distress" in r:
        return "person detail"
    if "identifier" in r:
        return "text element"
    if "observer" in r:
        return "background person"
    if "mail carrier" in r:
        return "main person"

    return normalize_space(role)


def is_small_low_confidence_anchor(anchor: Dict[str, Any]) -> bool:
    visibility = normalize_space(anchor.get("visibility", "")).lower()
    confidence = normalize_space(anchor.get("confidence", "")).lower()
    return visibility == "small" and confidence != "high"


def sanitize_anchor(anchor: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
    if not isinstance(anchor, dict):
        return None

    span = normalize_space(anchor.get("span", ""))
    if is_generic_span(span):
        return None

    localization = anchor.get("localization", {})
    if not isinstance(localization, dict):
        localization = {}

    level = normalize_space(localization.get("level", "")) or "object"
    region_hint = normalize_space(localization.get("region_hint", "")).lower() or "center"
    if region_hint not in REGION_HINTS_ALLOWLIST:
        region_hint = "center"

    attributes = clean_string_list(anchor.get("attributes", []))
    semantic_tags_raw = clean_string_list(anchor.get("semantic_tags_raw", []))
    semantic_tags = clean_string_list(anchor.get("semantic_tags", []))
    all_tags_for_role = semantic_tags_raw + semantic_tags
    functional_role = normalize_functional_role(anchor.get("functional_role", ""), span, all_tags_for_role)

    relations = anchor.get("relations", [])
    if not isinstance(relations, list):
        relations = []

    visibility = normalize_space(anchor.get("visibility", "")).lower() or "partial"
    if visibility not in VISIBILITY_ALLOWLIST:
        visibility = "partial"

    salience = normalize_space(anchor.get("salience", "")).lower() or "medium"
    if salience not in THREE_LEVEL_ALLOWLIST:
        salience = "medium"

    confidence = normalize_space(anchor.get("confidence", "")).lower() or "medium"
    if confidence not in THREE_LEVEL_ALLOWLIST:
        confidence = "medium"

    return {
        "anchor_id": f"a{idx+1}",
        "span": span,
        "semantic_tags_raw": semantic_tags_raw,
        "semantic_tags": semantic_tags,
        "functional_role": functional_role,
        "localization": {
            "level": level,
            "region_hint": region_hint,
        },
        "attributes": attributes,
        "relations": relations,
        "visibility": visibility,
        "salience": salience,
        "confidence": confidence,
    }


def postprocess_anchors_payload(obj: Dict[str, Any]) -> Dict[str, Any]:
    grounded_context_caption = obj.get("grounded_context_caption", "")
    if not isinstance(grounded_context_caption, str):
        grounded_context_caption = ""

    raw_anchors = obj.get("anchors", [])
    if not isinstance(raw_anchors, list):
        raw_anchors = []

    cleaned = []
    seen_spans = set()

    for idx, a in enumerate(raw_anchors):
        aa = sanitize_anchor(a, idx)
        if aa is None:
            continue

        span_norm = normalize_span_text(aa["span"])
        if span_norm in seen_spans:
            continue

        seen_spans.add(span_norm)
        cleaned.append(aa)

    old_to_new = {}
    for i, a in enumerate(cleaned):
        old_to_new[a["anchor_id"]] = f"a{i+1}"
        a["anchor_id"] = f"a{i+1}"

    valid_ids = {a["anchor_id"] for a in cleaned}
    for a in cleaned:
        rels = a.get("relations", [])
        fixed = []
        seen = set()
        for r in rels:
            if not isinstance(r, dict):
                continue
            t = r.get("target_anchor_id")
            typ = r.get("type")
            if not isinstance(t, str) or not isinstance(typ, str):
                continue
            t = old_to_new.get(t, t)
            typ = normalize_span_text(typ).replace(" ", "_")
            if typ not in RELATION_TYPES_ALLOWLIST or t not in valid_ids:
                continue
            key = (typ, t)
            if key in seen:
                continue
            seen.add(key)
            fixed.append({"type": typ, "target_anchor_id": t})
        a["relations"] = fixed

    obj_out = {
        "grounded_context_caption": normalize_space(grounded_context_caption),
        "anchors": cleaned,
    }

    normalize_relations(obj_out["anchors"])
    normalize_semantic_tags(obj_out["anchors"])
    return obj_out


def validate_anchors_payload(obj: Any, min_anchors: int = 1) -> None:
    if not isinstance(obj, dict):
        raise ValueError("anchors_not_dict")

    if "grounded_context_caption" not in obj or "anchors" not in obj:
        raise ValueError("anchors_missing_top_keys")

    if not isinstance(obj["grounded_context_caption"], str):
        raise ValueError("anchors_invalid_grounded_context_caption")

    anchors = obj["anchors"]
    if not isinstance(anchors, list):
        raise ValueError("anchors_invalid_list")

    if len(anchors) < min_anchors:
        raise ValueError(f"too_few_anchors:{len(anchors)}")

    seen = set()
    for idx, a in enumerate(anchors):
        if not isinstance(a, dict):
            raise ValueError(f"anchor_not_dict:{idx}")

        aid = a.get("anchor_id")
        if not isinstance(aid, str) or not aid:
            raise ValueError(f"anchor_invalid_id:{idx}")
        if aid in seen:
            raise ValueError(f"duplicate_anchor_id:{aid}")
        seen.add(aid)

        required = ["span", "semantic_tags", "functional_role", "localization", "attributes", "relations"]
        for k in required:
            if k not in a:
                raise ValueError(f"anchor_missing_key:{aid}:{k}")

        if not isinstance(a["span"], str) or not a["span"].strip():
            raise ValueError(f"anchor_invalid_span:{aid}")
        if is_generic_span(a["span"]):
            raise ValueError(f"anchor_generic_span:{aid}:{a['span']}")
        if not isinstance(a["semantic_tags"], list):
            raise ValueError(f"anchor_invalid_semantic_tags:{aid}")
        if not isinstance(a["functional_role"], str):
            raise ValueError(f"anchor_invalid_functional_role:{aid}")
        if not isinstance(a["localization"], dict):
            raise ValueError(f"anchor_invalid_localization:{aid}")
        if not isinstance(a["attributes"], list):
            raise ValueError(f"anchor_invalid_attributes:{aid}")
        if not isinstance(a["relations"], list):
            raise ValueError(f"anchor_invalid_relations:{aid}")

        loc = a["localization"]
        if "level" not in loc or "region_hint" not in loc:
            raise ValueError(f"anchor_missing_localization_fields:{aid}")
        if not isinstance(loc["level"], str) or not isinstance(loc["region_hint"], str):
            raise ValueError(f"anchor_invalid_localization_fields:{aid}")

    normalize_relations(anchors)
    normalize_semantic_tags(anchors)


def validate_missed_anchors_payload(obj: Any) -> None:
    if not isinstance(obj, dict):
        raise ValueError("missed_not_dict")
    missed = obj.get("missed_anchors")
    if not isinstance(missed, list):
        raise ValueError("missed_invalid_list")


def split_visible_text_into_units(visible_text: str) -> List[str]:
    txt = normalize_space(visible_text)
    if not txt:
        return []

    parts = re.split(r"\s{2,}|[|/]+", txt)
    parts = [normalize_space(p) for p in parts if normalize_space(p)]

    out: List[str] = []
    seen = set()

    def add_piece(piece: str):
        p = normalize_space(piece)
        if not p:
            return
        if len(p) < 2:
            return
        key = p.lower()
        if key in seen:
            return
        seen.add(key)
        out.append(p)

    for part in parts:
        if len(part.split()) <= 3:
            add_piece(part)
            continue

        row_matches = re.findall(r"\b\d{3,4}\s+[A-Za-z0-9&\-\.\']+(?:\s+[A-Za-z0-9&\-\.\']+){0,4}", part)
        if row_matches:
            for m in row_matches:
                add_piece(m)

        for token in re.findall(r"\b[A-Z][a-zA-Z]+\b", part):
            if token.lower() in {"departures", "cancelled", "delayed", "platform"}:
                add_piece(token)

    if not out:
        add_piece(txt)

    return out[:10]


def enrich_text_like_anchors(
    anchors_payload: Dict[str, Any],
    caption_obj: Dict[str, Any],
) -> Dict[str, Any]:
    merged = {
        "grounded_context_caption": anchors_payload.get("grounded_context_caption", ""),
        "anchors": list(anchors_payload.get("anchors", [])),
    }

    existing_spans = {normalize_span_text(a.get("span", "")) for a in merged["anchors"]}

    visible_text_items = clean_string_list(caption_obj.get("visible_text_items", []))
    if not visible_text_items:
        visible_text_items = split_visible_text_into_units(caption_obj.get("visible_text", ""))

    for txt in visible_text_items:
        norm = normalize_span_text(txt)
        if not norm or norm in existing_spans or is_generic_span(txt):
            continue

        merged["anchors"].append({
            "anchor_id": f"a{len(merged['anchors'])+1}",
            "span": txt,
            "semantic_tags_raw": ["sign_text"],
            "semantic_tags": ["sign_text"],
            "functional_role": "text element",
            "localization": {"level": "object", "region_hint": "center"},
            "attributes": [],
            "relations": [],
            "visibility": "full",
            "salience": "medium",
            "confidence": "medium",
        })
        existing_spans.add(norm)

    return postprocess_anchors_payload(merged)


def filter_weak_anchors_with_caption_support(
    anchors_payload: Dict[str, Any],
    caption_obj: Dict[str, Any],
) -> Dict[str, Any]:
    supported_spans = set()

    for key in ["notable_objects", "candidate_objects_all", "visible_text_items", "visible_regions"]:
        for item in clean_string_list(caption_obj.get(key, [])):
            supported_spans.add(normalize_span_text(item))

    visible_text = normalize_space(caption_obj.get("visible_text", ""))
    for unit in split_visible_text_into_units(visible_text):
        supported_spans.add(normalize_span_text(unit))

    kept = []
    for anchor in anchors_payload.get("anchors", []):
        if not isinstance(anchor, dict):
            continue
        span_norm = normalize_span_text(anchor.get("span", ""))
        if is_small_low_confidence_anchor(anchor) and span_norm not in supported_spans:
            continue
        kept.append(anchor)

    return postprocess_anchors_payload({
        "grounded_context_caption": anchors_payload.get("grounded_context_caption", ""),
        "anchors": kept,
    })


def merge_missed_anchors(
    anchors_payload: Dict[str, Any],
    missed_payload: Dict[str, Any],
) -> Dict[str, Any]:
    merged = {
        "grounded_context_caption": anchors_payload.get("grounded_context_caption", ""),
        "anchors": list(anchors_payload.get("anchors", [])),
    }

    missed = missed_payload.get("missed_anchors", [])
    if not isinstance(missed, list):
        missed = []

    existing_spans = {normalize_span_text(a.get("span", "")) for a in merged["anchors"]}

    for m in missed:
        if not isinstance(m, dict):
            continue
        span = m.get("span", "")
        span_norm = normalize_span_text(span)
        if not span_norm or span_norm in existing_spans or is_generic_span(span):
            continue

        merged["anchors"].append({
            "anchor_id": f"a{len(merged['anchors'])+1}",
            "span": normalize_space(span),
            "semantic_tags_raw": clean_string_list(m.get("semantic_tags_raw", [])),
            "semantic_tags": clean_string_list(m.get("semantic_tags", [])),
            "functional_role": normalize_space(m.get("functional_role", "")) or "visible scene element",
            "localization": {
                "level": normalize_space(((m.get("localization") or {}).get("level", "")) or "object"),
                "region_hint": normalize_space(((m.get("localization") or {}).get("region_hint", "")) or "center").lower(),
            },
            "attributes": clean_string_list(m.get("attributes", [])),
            "relations": [],
            "visibility": normalize_space(m.get("visibility", "")).lower() or "partial",
            "salience": normalize_space(m.get("salience", "")).lower() or "medium",
            "confidence": normalize_space(m.get("confidence", "")).lower() or "medium",
        })
        existing_spans.add(span_norm)

    return postprocess_anchors_payload(merged)


def make_fallback_anchors_from_caption(caption_obj: Dict[str, Any]) -> Dict[str, Any]:
    caption_text = caption_obj.get("image_caption_literal", "") if isinstance(caption_obj, dict) else ""
    visible_text = caption_obj.get("visible_text", "") if isinstance(caption_obj, dict) else ""
    notable_objects = caption_obj.get("notable_objects", []) if isinstance(caption_obj, dict) else []
    candidate_objects_all = caption_obj.get("candidate_objects_all", []) if isinstance(caption_obj, dict) else []
    visible_text_items = caption_obj.get("visible_text_items", []) if isinstance(caption_obj, dict) else []
    visible_regions = caption_obj.get("visible_regions", []) if isinstance(caption_obj, dict) else []

    anchors: List[Dict[str, Any]] = []
    used = set()

    def try_add(span: str, functional_role: str = "visible scene element", hint: str = "center"):
        if not isinstance(span, str):
            return
        s = normalize_space(span)
        s_norm = normalize_span_text(s)
        if not s or s_norm in used or is_generic_span(s):
            return
        used.add(s_norm)

        broad_tag = infer_broad_tag_from_span(s)
        anchors.append({
            "anchor_id": f"a{len(anchors)+1}",
            "span": s,
            "semantic_tags_raw": [broad_tag],
            "semantic_tags": [broad_tag],
            "functional_role": functional_role,
            "localization": {
                "level": "object",
                "region_hint": hint,
            },
            "attributes": [],
            "relations": [],
            "visibility": "partial",
            "salience": "medium",
            "confidence": "medium",
        })

    for obj in notable_objects:
        try_add(obj)

    for obj in candidate_objects_all:
        if len(anchors) >= 8:
            break
        try_add(obj)

    for reg in visible_regions:
        if len(anchors) >= 10:
            break
        try_add(reg, functional_role="scene region")

    for txt in visible_text_items:
        if len(anchors) >= 12:
            break
        try_add(txt, functional_role="text element")

    if visible_text.strip():
        for txt in split_visible_text_into_units(visible_text):
            if len(anchors) >= 14:
                break
            try_add(txt, functional_role="text element")

    payload = {
        "grounded_context_caption": "No clear grounding." if caption_text else "",
        "anchors": anchors,
    }
    return postprocess_anchors_payload(payload)


# ----------------------------
# Qwen3-VL runner
# ----------------------------

class QwenVLRunner:
    def __init__(self, model_name: str, device: str = "cuda", dtype: str = "bfloat16", use_flash_attn: bool = False):
        self.device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"

        if dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "float16":
            torch_dtype = torch.float16
        else:
            raise ValueError("dtype must be 'bfloat16' or 'float16'")

        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": "auto" if self.device == "cuda" else None,
        }
        if use_flash_attn and self.device == "cuda":
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, **model_kwargs)

        if self.device != "cuda":
            self.model.to(self.device)

        self.model.eval()

    def _move_inputs(self, inputs):
        return inputs.to(self.model.device)

    @torch.inference_mode()
    def generate_text(self, prompt: str, max_new_tokens: int = 256) -> str:
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        inputs = self._move_inputs(inputs)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return output_text.strip()

    @torch.inference_mode()
    def generate_image_text(self, prompt: str, image_path: Path, max_new_tokens: int = 256) -> str:
        image_input = str(image_path.resolve())

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_input},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        inputs = self._move_inputs(inputs)

        image_token_index = getattr(self.model.config, "image_token_index", None)
        if image_token_index is not None:
            n_img_tokens = int((inputs["input_ids"] == image_token_index).sum().item())
            if n_img_tokens == 0:
                raise ValueError(
                    f"no_image_tokens_inserted_for:{image_path.name}. "
                    f"Check transformers version and message formatting."
                )

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return output_text.strip()


# ----------------------------
# Repair helper
# ----------------------------

def repair_json_with_model(
    runner: QwenVLRunner,
    schema_obj: Dict[str, Any],
    raw_text: str,
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    prompt = JSON_REPAIR_PROMPT_TEMPLATE.format(
        schema=json.dumps(schema_obj, ensure_ascii=False, indent=2),
        raw=raw_text[:6000],
    )
    repaired = runner.generate_text(prompt, max_new_tokens=max_new_tokens)
    return safe_json_loads(repaired)


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="", help="Path to CSV. If empty, picks newest in metadata/pristine/")
    ap.add_argument("--csv-dir", type=str, default="/home/Student/s4826850/timpact/metadata/pristine")
    ap.add_argument("--images-dir", type=str, default="/home/Student/s4826850/timpact/data/pristine_images/images/images")
    ap.add_argument("--output", type=str, default="/home/Student/s4826850/timpact/data/anchors_qwen3.json")
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--use-flash-attn", action="store_true")
    ap.add_argument("--max-samples", type=int, default=0, help="If >0, stop after N processed rows.")
    ap.add_argument("--start-row", type=int, default=0)
    ap.add_argument("--end-row", type=int, default=-1)
    ap.add_argument("--resume", action="store_true", help="Resume by skipping ids already in output.")
    ap.add_argument("--save-every", type=int, default=1, help="Write output every N rows.")
    args = ap.parse_args()

    csv_path = Path(args.csv) if args.csv else pick_newest_csv(Path(args.csv_dir))
    images_dir = Path(args.images_dir)
    out_path = Path(args.output)

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"

    print(f"[MODEL] device={device}")
    print(f"[INFO] Using CSV: {csv_path}")
    print(f"[INFO] Images dir: {images_dir}")
    print(f"[INFO] Output: {out_path}")
    sys.stdout.flush()

    runner = QwenVLRunner(
        model_name=args.model,
        device=device,
        dtype=args.dtype,
        use_flash_attn=args.use_flash_attn,
    )

    print(f"[MODEL] Loaded: {args.model}")
    sys.stdout.flush()

    existing = load_existing(out_path) if args.resume else None
    records: List[Dict[str, Any]] = []
    done_ids = set()

    if existing and isinstance(existing, dict) and "records" in existing:
        records = existing.get("records", [])
        for r in records:
            if "id" in r:
                done_ids.add(r["id"])

    output_obj: Dict[str, Any] = existing if (existing and isinstance(existing, dict)) else {
        "meta": {
            "created_at": now_iso(),
            "model": args.model,
            "csv": str(csv_path),
            "images_dir": str(images_dir),
        },
        "records": records,
    }

    with csv_path.open("r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    start = max(0, args.start_row)
    end = args.end_row if args.end_row >= 0 else len(rows)
    rows = rows[start:end]

    total = 0
    ok = 0
    missing_image = 0
    failed = 0

    for i, row in enumerate(rows):
        row_index = start + i
        rid = f"{row_index + 1:06d}"

        if args.resume and rid in done_ids:
            continue

        img_name = (row.get("images") or row.get("image") or row.get("img") or "").strip()
        title = decode_weird_utf8((row.get("title") or "").strip())
        summary = decode_weird_utf8((row.get("summary") or row.get("description") or "").strip())
        published = (row.get("published") or row.get("date") or "").strip()
        url = (row.get("url") or "").strip()

        print(f"[RUN] row={row_index} id={rid} img={img_name}")
        sys.stdout.flush()

        rec: Dict[str, Any] = {
            "id": rid,
            "row_index": row_index,
            "csv_fields": {
                "title": title,
                "summary": summary,
                "published": published,
                "url": url,
                "images": img_name,
            },
            "status": "pending",
            "errors": [],
            "outputs": {},
            "timing": {},
            "created_at": now_iso(),
        }

        img_path = images_dir / img_name if img_name else None
        if not img_path or not img_path.exists():
            rec["status"] = "missing_image"
            rec["errors"].append(f"missing_image:{img_name}")
            output_obj["records"].append(rec)
            missing_image += 1
            total += 1

            if total % args.save_every == 0:
                write_atomic(out_path, output_obj)
            if args.max_samples and total >= args.max_samples:
                print(f"[INFO] Reached max_samples={args.max_samples}, stopping.")
                break
            continue

        try:
            with Image.open(img_path) as im:
                im.verify()
        except Exception as e:
            rec["status"] = "failed"
            rec["errors"].append(f"image_open_failed:{repr(e)}")
            output_obj["records"].append(rec)
            failed += 1
            total += 1

            if total % args.save_every == 0:
                write_atomic(out_path, output_obj)
            if args.max_samples and total >= args.max_samples:
                print(f"[INFO] Reached max_samples={args.max_samples}, stopping.")
                break
            continue

        # 1) Caption
        t0 = time.time()
        caption_json = None
        raw_caption = ""
        try:
            raw_caption = runner.generate_image_text(CAPTION_PROMPT, img_path, max_new_tokens=384)
            try:
                caption_json = safe_json_loads(raw_caption)
                validate_caption_payload(caption_json)
            except Exception:
                caption_json = repair_json_with_model(
                    runner,
                    {
                        "image_caption_literal": "",
                        "visible_text": "",
                        "people_count": 0,
                        "notable_objects": [],
                        "visible_text_items": [],
                        "visible_regions": [],
                        "candidate_objects_all": [],
                    },
                    raw_caption,
                    max_new_tokens=384,
                )
                validate_caption_payload(caption_json)

            caption_json["notable_objects"] = clean_string_list(caption_json.get("notable_objects", []))
            caption_json["visible_text_items"] = clean_string_list(caption_json.get("visible_text_items", []))
            caption_json["visible_regions"] = clean_string_list(caption_json.get("visible_regions", []))
            caption_json["candidate_objects_all"] = clean_string_list(caption_json.get("candidate_objects_all", []))

            rec["outputs"]["caption"] = caption_json
            ok1 = True
        except Exception as e:
            ok1 = False
            rec["errors"].append(f"caption_failed:{repr(e)}")
            rec["outputs"]["caption_raw_snippet"] = raw_caption[:1200]

        rec["timing"]["caption_s"] = round(time.time() - t0, 2)
        print(f"[TIME] caption {rec['timing']['caption_s']}s ok={ok1}")
        sys.stdout.flush()

        # 2) Context rewrite
        t0 = time.time()
        context_json = None
        raw_context = ""
        try:
            ctx_prompt = CONTEXT_PROMPT_TEMPLATE.format(title=title, summary=summary)
            raw_context = runner.generate_text(ctx_prompt, max_new_tokens=256)

            try:
                context_json = safe_json_loads(raw_context)
                validate_context_payload(context_json)
            except Exception:
                context_json = repair_json_with_model(
                    runner,
                    {"article_context_rewrite": ""},
                    raw_context,
                    max_new_tokens=256,
                )
                validate_context_payload(context_json)

            rec["outputs"]["context"] = context_json
            ok2 = True
        except Exception as e:
            ok2 = False
            rec["errors"].append(f"context_failed:{repr(e)}")
            rec["outputs"]["context_raw_snippet"] = raw_context[:1200]

        rec["timing"]["context_s"] = round(time.time() - t0, 2)
        print(f"[TIME] context {rec['timing']['context_s']}s ok={ok2}")
        sys.stdout.flush()

        # 3) Anchors
        t0 = time.time()
        anchors_json = None
        raw_anchors = ""
        raw_anchors_repair = ""
        raw_missed = ""
        raw_missed_repair = ""

        try:
            cap_text = (caption_json or {}).get("image_caption_literal", "")
            visible_text = (caption_json or {}).get("visible_text", "")
            ctx_text = (context_json or {}).get("article_context_rewrite", "")

            a_prompt = ANCHORS_PROMPT_TEMPLATE.format(
                caption=cap_text,
                visible_text=visible_text,
                context=ctx_text,
            )

            raw_anchors = runner.generate_image_text(a_prompt, img_path, max_new_tokens=1300)

            try:
                anchors_json = safe_json_loads(raw_anchors)
            except Exception:
                raw_anchors_repair = runner.generate_text(
                    JSON_REPAIR_PROMPT_TEMPLATE.format(
                        schema=json.dumps(
                            {
                                "grounded_context_caption": "",
                                "anchors": [
                                    {
                                        "anchor_id": "a1",
                                        "span": "",
                                        "semantic_tags_raw": [],
                                        "semantic_tags": [],
                                        "functional_role": "",
                                        "localization": {
                                            "level": "object",
                                            "region_hint": ""
                                        },
                                        "attributes": [],
                                        "relations": [],
                                        "visibility": "partial",
                                        "salience": "medium",
                                        "confidence": "medium",
                                    }
                                ]
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        raw=raw_anchors[:6000],
                    ),
                    max_new_tokens=1300,
                )
                anchors_json = safe_json_loads(raw_anchors_repair)

            anchors_json = postprocess_anchors_payload(anchors_json)

            if len(anchors_json.get("anchors", [])) == 0:
                anchors_json = make_fallback_anchors_from_caption(caption_json or {})

            anchors_json = enrich_text_like_anchors(anchors_json, caption_json or {})

            # 3b) Missed-anchor recovery pass
            try:
                missed_prompt = MISSED_ANCHORS_PROMPT_TEMPLATE.format(
                    current_anchors_json=json.dumps(anchors_json.get("anchors", []), ensure_ascii=False, indent=2)
                )
                raw_missed = runner.generate_image_text(missed_prompt, img_path, max_new_tokens=700)
                try:
                    missed_json = safe_json_loads(raw_missed)
                    validate_missed_anchors_payload(missed_json)
                except Exception:
                    raw_missed_repair = runner.generate_text(
                        JSON_REPAIR_PROMPT_TEMPLATE.format(
                            schema=json.dumps(
                                {
                                    "missed_anchors": [
                                        {
                                            "span": "",
                                            "semantic_tags_raw": [],
                                            "semantic_tags": [],
                                            "functional_role": "",
                                            "localization": {
                                                "level": "object",
                                                "region_hint": "center"
                                            },
                                            "attributes": [],
                                            "visibility": "partial",
                                            "salience": "medium",
                                            "confidence": "medium",
                                        }
                                    ]
                                },
                                ensure_ascii=False,
                                indent=2,
                            ),
                            raw=raw_missed[:6000],
                        ),
                        max_new_tokens=700,
                    )
                    missed_json = safe_json_loads(raw_missed_repair)
                    validate_missed_anchors_payload(missed_json)

                anchors_json = merge_missed_anchors(anchors_json, missed_json)
            except Exception as e:
                rec["errors"].append(f"missed_anchors_recovery_failed:{repr(e)}")

            anchors_json = filter_weak_anchors_with_caption_support(anchors_json, caption_json or {})

            if len(anchors_json.get("anchors", [])) == 0:
                anchors_json = make_fallback_anchors_from_caption(caption_json or {})

            validate_anchors_payload(anchors_json, min_anchors=1)

            rec["outputs"]["anchors"] = anchors_json
            rec["status"] = "ok"
            ok3 = True
        except Exception as e:
            ok3 = False
            rec["status"] = "failed"
            rec["errors"].append(f"anchors_failed:{repr(e)}")
            rec["outputs"]["anchors_raw_snippet"] = raw_anchors[:1500]
            rec["outputs"]["anchors_repair_raw_snippet"] = raw_anchors_repair[:1500]
            rec["outputs"]["missed_anchors_raw_snippet"] = raw_missed[:1200]
            rec["outputs"]["missed_anchors_repair_raw_snippet"] = raw_missed_repair[:1200]

        rec["timing"]["anchors_s"] = round(time.time() - t0, 2)
        print(f"[TIME] anchors {rec['timing']['anchors_s']}s ok={ok3}")
        sys.stdout.flush()

        output_obj["records"].append(rec)

        total += 1
        if rec["status"] == "ok":
            ok += 1
        elif rec["status"] == "missing_image":
            missing_image += 1
        else:
            failed += 1

        if total % args.save_every == 0:
            write_atomic(out_path, output_obj)

        if args.max_samples and total >= args.max_samples:
            print(f"[INFO] Reached max_samples={args.max_samples}, stopping.")
            break

    write_atomic(out_path, output_obj)
    print(f"[DONE] Saved: {out_path}")
    print(f"[STATS] total={total} ok={ok} missing_image={missing_image} failed={failed}")


if __name__ == "__main__":
    main()