"""
Layer 2 — Parameterization & Sorting Algorithm

Pure deterministic processing — no AI calls.

Takes the ContextBundle from Layer 1 and the raw prompt, extracts all numeric
values with units, converts to millimetres, and sorts them into four categories:

  explicit_dims   — directly stated geometric dimensions (width, height, radius, …)
  implied_dims    — inferred from capacity/weight/volume targets
  functional_params — thread sizes, hole counts, wall thicknesses, tolerances
  aesthetic_params  — fillet radii, taper angles, surface finish hints

Also runs the Pareto 80/20 feature-priority sort so later layers know which
features define the object identity vs. which are supporting details.

Location: cadfactory-backend/services/param_sorter.py
"""

from __future__ import annotations

import re
import math
import logging
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field

from services.nlp_extractor import ContextBundle
from services.engineering_math import (
    pareto_critical_features,
    cylinder_volume,
    fill_height_cylinder,
    lookup_reference_dims,
    PHI,
)

logger = logging.getLogger(__name__)


# ── Output model ───────────────────────────────────────────────────────────────

class SortedParams(BaseModel):
    """Structured output of Layer 2 — sorted and unit-normalised parameters."""

    # Hard geometric dimensions in mm (name → mm value)
    explicit_dims: Dict[str, float] = Field(default_factory=dict)

    # Implied dimensions derived from capacity / weight targets (name → mm value)
    implied_dims: Dict[str, float] = Field(default_factory=dict)

    # Functional parameters: holes, threads, wall, clearances (name → value)
    functional_params: Dict[str, Any] = Field(default_factory=dict)

    # Aesthetic parameters: fillets, surface finish, taper angles (name → value)
    aesthetic_params: Dict[str, Any] = Field(default_factory=dict)

    # Pareto-ranked feature priority list (most identity-defining first)
    priority_order: List[str] = Field(default_factory=list)

    # Reference dims from industrial DB (if matched)
    reference_category: Optional[str] = None
    reference_dims_used: bool = False

    # Resolved manufacturing process
    manufacturing_process: str = "fdm"

    # Detected object category (forwarded from Layer 1)
    object_category: str = "other"

    # Confidence score (composite)
    confidence: float = 0.0


# ── Unit conversion ────────────────────────────────────────────────────────────

_UNIT_TO_MM: Dict[str, float] = {
    "mm": 1.0,
    "millimeter": 1.0,
    "millimetre": 1.0,
    "cm": 10.0,
    "centimeter": 10.0,
    "centimetre": 10.0,
    "m": 1000.0,
    "meter": 1000.0,
    "metre": 1000.0,
    "in": 25.4,
    "inch": 25.4,
    "inches": 25.4,
    '"': 25.4,
    "ft": 304.8,
    "foot": 304.8,
    "feet": 304.8,
}

# Regex: match things like "50mm", "3.5 cm", "1.5in", "2 inches", '3"'
_NUM_UNIT_RE = re.compile(
    r'(\d+(?:\.\d+)?)\s*'
    r'(mm|millimeters?|millimetres?|cm|centimeters?|centimetres?|m\b|meters?|metres?'
    r'|in\b|inches?|inch|"|ft\b|feet?|foot)',
    re.IGNORECASE
)

# Keyword → dimension key mappings
_DIM_KEYWORDS: List[Tuple[str, str]] = [
    # (search term, dimension key)
    ("wide",  "width"),  ("width",  "width"),
    ("tall",  "height"), ("height", "height"), ("high", "height"), ("deep", "depth"),
    ("depth", "depth"),  ("long",   "length"), ("length", "length"),
    ("thick", "wall_thickness"), ("thickness", "wall_thickness"), ("wall", "wall_thickness"),
    ("radius", "radius"), ("diameter", "diameter"), ("bore", "bore_diameter"),
    ("hole",  "hole_diameter"), ("inner", "inner_diameter"), ("outer", "outer_diameter"),
    ("neck",  "neck_diameter"), ("nozzle", "nozzle_diameter"),
    ("base",  "base_diameter"), ("top",  "top_diameter"),
]

# Capacity-related patterns (ml, l, cc, oz, pint)
_CAPACITY_RE = re.compile(
    r'(\d+(?:\.\d+)?)\s*(ml|milliliter|millilitre|l\b|liter|litre|cc|oz|fl\.? ?oz|pint)',
    re.IGNORECASE
)

# Weight patterns (g, kg, lb)
_WEIGHT_RE = re.compile(
    r'(\d+(?:\.\d+)?)\s*(g\b|gram|grams|kg|kilogram|lb|pound)',
    re.IGNORECASE
)

# Thread pattern e.g. M3, M6x1.0, 1/4-20
_THREAD_RE = re.compile(r'\bM(\d+(?:\.\d+)?)(?:×|x|×)?([\d.]+)?\b', re.IGNORECASE)

# Hole count pattern e.g. "4 holes", "6 mounting holes"
_HOLE_COUNT_RE = re.compile(r'(\d+)\s+(?:mounting\s+)?holes?', re.IGNORECASE)

# Fillet / chamfer / radius hints
_FILLET_RE = re.compile(r'(?:fillet|round|chamfer|radius|corner)\s+r?\s*=?\s*(\d+(?:\.\d+)?)', re.IGNORECASE)

# Draft / taper angle
_DRAFT_RE = re.compile(r'(\d+(?:\.\d+)?)\s*°?\s*(?:draft|taper)', re.IGNORECASE)


def _to_mm(value: float, unit_str: str) -> float:
    """Convert a numeric value in the given unit string to millimetres."""
    factor = _UNIT_TO_MM.get(unit_str.lower().strip(), 1.0)
    return value * factor


def _capacity_to_ml(value: float, unit_str: str) -> float:
    """Normalise capacity to millilitres."""
    u = unit_str.lower().strip()
    if u in ("ml", "milliliter", "millilitre", "cc"):
        return value
    if u in ("l", "liter", "litre"):
        return value * 1000.0
    if u in ("oz", "fl oz", "fl. oz"):
        return value * 29.5735
    if u in ("pint",):
        return value * 473.176
    return value


# ── Dim keyword extractor ──────────────────────────────────────────────────────

def _extract_dims_from_text(text: str) -> Dict[str, float]:
    """
    Find all 'NUMBER UNIT' pairs in text and associate them with the nearest
    dimension keyword (width, height, radius, …).
    Returns {dim_key: mm_value}.
    """
    dims: Dict[str, float] = {}

    # Find all value+unit spans
    for m in _NUM_UNIT_RE.finditer(text):
        val_str, unit = m.group(1), m.group(2)
        val_mm = _to_mm(float(val_str), unit)
        span_start = m.start()

        # Search for the closest dimension keyword within 40 chars before the number
        context_before = text[max(0, span_start - 40): span_start].lower()
        matched_key = None
        best_dist = 999
        for keyword, key in _DIM_KEYWORDS:
            idx = context_before.rfind(keyword)
            if idx >= 0:
                dist = len(context_before) - idx
                if dist < best_dist:
                    best_dist = dist
                    matched_key = key

        if matched_key and matched_key not in dims:
            dims[matched_key] = val_mm
        elif matched_key is None:
            # Heuristic: assign to generic positional dim based on existing keys
            for generic_key in ("width", "height", "length", "depth", "radius"):
                if generic_key not in dims:
                    dims[generic_key] = val_mm
                    break

    return dims


def _extract_from_raw_constraints(raw: Dict[str, str]) -> Dict[str, float]:
    """Parse the raw_constraints dict from ContextBundle into mm float values."""
    dims: Dict[str, float] = {}
    for key, val_str in raw.items():
        # Try to extract value+unit from the string
        m = _NUM_UNIT_RE.search(val_str)
        if m:
            val_mm = _to_mm(float(m.group(1)), m.group(2))
            # Map key name to standard dimension key
            key_lower = key.lower().replace(" ", "_")
            dims[key_lower] = val_mm
        else:
            # Try bare numeric (assume mm)
            try:
                dims[key.lower().replace(" ", "_")] = float(re.search(r'[\d.]+', val_str).group())
            except (AttributeError, ValueError):
                pass
    return dims


# ── Implied dimension derivation ───────────────────────────────────────────────

def _derive_dims_from_capacity(capacity_ml: float, existing: Dict[str, float]) -> Dict[str, float]:
    """
    When only a capacity is stated (e.g. 350ml), derive cylinder dimensions.
    Uses standard aspect ratio φ:1 (height:width) to solve r and h.
    """
    implied: Dict[str, float] = {}

    if "radius" in existing or "width" in existing or "height" in existing:
        return implied  # already have geometry, don't override

    # Solve: V = π·r²·h, h = φ·2r  →  V = π·r²·(2φ·r) = 2πφ·r³
    # r = (V / (2π·φ))^(1/3)
    V_mm3 = capacity_ml * 1000.0  # ml → mm³ (add wall factor 1.15 for shell)
    r_inner = (V_mm3 / (2 * math.pi * PHI)) ** (1.0 / 3.0)
    h_inner = 2 * PHI * r_inner

    # Round to practical values
    implied["inner_radius"] = round(r_inner, 1)
    implied["inner_height"] = round(h_inner, 1)
    implied["capacity_ml"]  = capacity_ml

    return implied


# ── Functional parameter extraction ───────────────────────────────────────────

def _extract_functional(text: str) -> Dict[str, Any]:
    """Extract functional parameters: threads, hole counts, tolerances, materials."""
    func: Dict[str, Any] = {}

    # Threads
    for m in _THREAD_RE.finditer(text):
        nominal = float(m.group(1))
        pitch   = float(m.group(2)) if m.group(2) else None
        func["thread_nominal_mm"] = nominal
        if pitch:
            func["thread_pitch_mm"] = pitch

    # Hole counts
    for m in _HOLE_COUNT_RE.finditer(text):
        func["hole_count"] = int(m.group(1))

    # Wall thickness (if not already in explicit dims — avoid double extraction)
    wm = re.search(r'(\d+(?:\.\d+)?)\s*mm\s*wall', text, re.IGNORECASE)
    if wm:
        func["wall_thickness_mm"] = float(wm.group(1))

    # Clearance / tolerance hints
    if re.search(r'press.?fit|interference\s+fit', text, re.IGNORECASE):
        func["fit_type"] = "press"
    elif re.search(r'slip.?fit|sliding\s+fit|clearance\s+fit', text, re.IGNORECASE):
        func["fit_type"] = "sliding"
    elif re.search(r'snap.?fit', text, re.IGNORECASE):
        func["fit_type"] = "snap"

    return func


def _extract_aesthetic(text: str) -> Dict[str, Any]:
    """Extract aesthetic parameters: fillets, draft angles, surface finish."""
    aes: Dict[str, Any] = {}

    # Fillet radius
    for m in _FILLET_RE.finditer(text):
        aes["fillet_r_mm"] = float(m.group(1))
        break  # take first

    # Draft angle
    for m in _DRAFT_RE.finditer(text):
        aes["draft_angle_deg"] = float(m.group(1))
        break

    # Surface finish hints
    if re.search(r'smooth|glossy|polished|mirror', text, re.IGNORECASE):
        aes["surface_finish"] = "smooth"
    elif re.search(r'rough|textured|matte|knurl', text, re.IGNORECASE):
        aes["surface_finish"] = "textured"

    # Symmetry hints
    if re.search(r'symmetric|symmetrical|mirrored', text, re.IGNORECASE):
        aes["symmetry"] = "bilateral"
    elif re.search(r'radial|circular|polar|rotational', text, re.IGNORECASE):
        aes["symmetry"] = "rotational"

    return aes


# ── Pareto priority sorter ─────────────────────────────────────────────────────

_FEATURE_WEIGHTS: Dict[str, float] = {
    # Higher = more identity-defining
    "body": 10.0, "main_body": 10.0, "hull": 10.0,
    "head": 8.0, "cap": 8.0, "pump_head": 8.0,
    "handle": 7.0, "grip": 7.0, "arm": 6.0,
    "flange": 5.0, "collar": 5.0, "neck": 5.0,
    "base": 4.0, "foot": 4.0, "rim": 4.0,
    "hole": 3.0, "slot": 3.0, "groove": 3.0,
    "chamfer": 2.0, "fillet": 2.0,
    "thread": 2.0, "pin": 2.0,
    "boss": 1.5, "rib": 1.5, "gusset": 1.5,
    "label_area": 1.0, "text": 1.0,
}


def _build_priority_order(explicit_dims: Dict[str, float], category: str) -> List[str]:
    """
    Build a Pareto-sorted feature priority list from dimension keys + category.
    Returns keys sorted by visual/structural importance.
    """
    feature_dicts = []
    for key in list(explicit_dims.keys()) + _infer_feature_names(category):
        clean = key.lower().replace("_mm", "").replace("_diameter", "_dia")
        weight = _FEATURE_WEIGHTS.get(clean, 1.0)

        # Boost weight for keys that match industrial DB fields
        if any(k in clean for k in ("body", "main", "hull", "head", "grip")):
            weight = max(weight, 6.0)

        feature_dicts.append({"name": key, "visual_weight": weight})

    result = pareto_critical_features(feature_dicts)
    priority = [f["name"] for f in result.get("critical", [])]
    priority += [f["name"] for f in result.get("supporting", [])]
    return priority


def _infer_feature_names(category: str) -> List[str]:
    """Return likely feature names for a product category."""
    cats = {
        "container": ["main_body", "neck", "cap", "base"],
        "fastener":  ["head", "shank", "thread", "socket"],
        "bracket":   ["horizontal_plate", "vertical_plate", "hole"],
        "enclosure": ["shell", "lid", "boss", "rib"],
        "mount":     ["plate", "arm", "hole", "fillet"],
        "connector": ["body", "pin", "socket", "flange"],
    }
    return cats.get(category.lower(), ["main_body", "hole"])


# ── Main sort function ─────────────────────────────────────────────────────────

def sort_parameters(context: ContextBundle, raw_prompt: str) -> SortedParams:
    """
    Layer 2: Convert a ContextBundle + raw prompt into a SortedParams structure.

    Pure algorithmic — no AI calls. Always returns a SortedParams.
    """
    full_text = raw_prompt + " " + " ".join(
        f"{k} {v}" for k, v in context.raw_constraints.items()
    )

    # ── Extract explicit dimensions ───────────────────────────────────────────
    explicit_from_text  = _extract_dims_from_text(full_text)
    explicit_from_ctx   = _extract_from_raw_constraints(context.raw_constraints)
    # Merge — ctx values (AI-extracted) take precedence
    explicit = {**explicit_from_text, **explicit_from_ctx}

    # ── Capacity-derived implied dimensions ───────────────────────────────────
    implied: Dict[str, float] = {}
    cap_match = _CAPACITY_RE.search(full_text)
    if cap_match:
        capacity_ml = _capacity_to_ml(float(cap_match.group(1)), cap_match.group(2))
        implied = _derive_dims_from_capacity(capacity_ml, explicit)
        explicit["capacity_ml"] = capacity_ml  # store for reference

    # ── Functional params ─────────────────────────────────────────────────────
    functional = _extract_functional(full_text)

    # ── Aesthetic params ──────────────────────────────────────────────────────
    aesthetic = _extract_aesthetic(full_text)
    # Apply geometric modifiers from Layer 1
    if "rounded" in context.geometric_modifiers:
        if "fillet_r_mm" not in aesthetic:
            aesthetic["fillet_r_mm"] = 3.0  # default soft radius for "rounded"
    if "tapered" in context.geometric_modifiers or "slim" in context.geometric_modifiers:
        if "draft_angle_deg" not in aesthetic:
            aesthetic["draft_angle_deg"] = 2.0

    # ── Industrial DB reference ────────────────────────────────────────────────
    ref_dims = lookup_reference_dims(context.object_category)
    ref_category = None
    ref_used = False
    if ref_dims and not explicit:
        # Fall back to industrial DB if we extracted nothing
        implied["width"] = ref_dims.body_width_mm
        implied["height"] = ref_dims.body_height_mm
        implied["depth"] = ref_dims.body_depth_mm
        implied["wall_thickness"] = ref_dims.wall_thickness_mm
        ref_category = ref_dims.category
        ref_used = True

    # ── Pareto priority sort ──────────────────────────────────────────────────
    all_dim_keys = list(explicit.keys()) + list(implied.keys())
    priority = _build_priority_order(
        {k: 1.0 for k in all_dim_keys},
        context.object_category
    )

    # ── Manufacturing process ─────────────────────────────────────────────────
    process = context.manufacturing_hint or "fdm"

    # ── Confidence score ──────────────────────────────────────────────────────
    n_params = len(explicit) + len(implied) + len(functional)
    confidence = min(1.0, 0.4 + 0.1 * n_params) * context.confidence

    return SortedParams(
        explicit_dims=explicit,
        implied_dims=implied,
        functional_params=functional,
        aesthetic_params=aesthetic,
        priority_order=priority,
        reference_category=ref_category,
        reference_dims_used=ref_used,
        manufacturing_process=process,
        object_category=context.object_category,
        confidence=round(confidence, 3),
    )
