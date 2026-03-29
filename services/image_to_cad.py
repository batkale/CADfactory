"""
Image-to-CAD generation service.

Inspired by GenCAD (https://github.com/ferdous-alam/GenCAD) — uses their
structured geometric extraction approach (base geometry, sketch profiles,
extrusion ops, key dimensions) but adapted for Gemini Vision instead of
a trained ResNet+diffusion pipeline.

Key difference from GenCAD: Gemini Vision is a general vision LLM that works
best with full-resolution colour images. GenCAD's edge-detection preprocessing
(designed for a ResNet trained on edge-processed images) is NOT applied here,
as it destroys the colour/texture information Gemini relies on.
"""

import io
import os
import base64
import logging
import mimetypes
from typing import Optional
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

_MAX_SIDE = 1024   # max dimension sent to Gemini — keeps payload reasonable

# ---------------------------------------------------------------------------
# Image preparation — resize only, preserve colour for Gemini Vision
# ---------------------------------------------------------------------------

def _prepare_image_bytes(image_bytes: bytes, filename: str = "image.png") -> tuple[bytes, str]:
    """
    Prepare an image for Gemini Vision.

    Resizes to at most _MAX_SIDE px on the longest side to keep the payload
    reasonable, but preserves full colour and resolution — Gemini Vision works
    best with high-quality images, not edge-processed thumbnails.
    """
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = img.size
        if max(w, h) > _MAX_SIDE:
            scale = _MAX_SIDE / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        out = io.BytesIO()
        img.save(out, format="PNG")
        return out.getvalue(), "image/png"

    except Exception as e:
        logger.warning(f"Image preparation failed (using original): {e}")
        mime_type, _ = mimetypes.guess_type(filename)
        return image_bytes, mime_type or "image/png"


# ---------------------------------------------------------------------------
# Gemini Vision prompts
# ---------------------------------------------------------------------------

# Outputs the exact same JSON schema as config/prompts.py DESIGN_PLAN_PROMPT
# so the result can be injected directly into DESIGN_PLAN_TO_CODE_TEMPLATE.
IMAGE_TO_CAD_PLAN_PROMPT = """You are a senior mechanical CAD design planner analyzing an image of a physical part, sketch, or technical drawing.

Analyze the image and output a concise design plan in JSON — no code, no markdown.

Think through methodically:
1. What is the primary base geometry? (box, cylinder, polygon_extrusion, revolved_profile)
2. Classify the aesthetic intent: mechanical (brackets, mounts), consumer (spinners, cases, handles), or decorative (ornaments, display pieces).
3. Exact key dimensions in millimetres — estimate from proportions and visual cues. Use realistic defaults (a coffee mug is ~80 mm tall, a phone is ~150×75 mm).
4. Structural features (max 4): the core geometry that defines the part shape.
5. Detail features (max 3): bearing seats, recesses, grooves, pockets, holes.
6. Finish features (max 3): fillets, chamfers, grooves — consumer/decorative parts MUST have at least 2.
7. Safe CadQuery selectors to use (only: >Z <Z >X <X >Y <Y |Z |X |Y).
8. Clearance or fit requirements if visible.

Also output a plain text_description field with a single paragraph suitable for a text-to-CAD fallback.

NEVER plan: .text(), filter_by(), StringSelector, Perimeter(), or complex edge chains.

Return ONLY valid JSON, no backticks:
{
  "part_name": "short name",
  "aesthetic_class": "mechanical|consumer|decorative",
  "base": "box|cylinder|polygon_extrusion|revolved_profile",
  "dims": {"key": value_mm},
  "features": [
    {"op": "shell|hole|cboreHole|pushPoints_hole|rect_extrude|rect_cut|circle_extrude|bearing_pocket|radial_arms|polar_array", "category": "structural|detail|finish", "desc": "...", "params": {}}
  ],
  "finish_features": [
    {"type": "fillet|chamfer|groove|recess", "location": "all_vertical_edges|bottom_edges|top_face|union_seams", "radius_mm": 2.0}
  ],
  "sequence": ["step 1 description", "step 2 description"],
  "notes": "any special CadQuery approach or selector notes",
  "text_description": "single paragraph describing the part for text-to-CAD use"
}"""

IMAGE_ANALYSIS_PROMPT = """You are an expert mechanical engineer analyzing a CAD part image.

Analyze this image and provide a structured assessment:
1. Part identification: What type of part is this?
2. Geometric complexity: Simple (1-3) / Moderate (4-6) / Complex (7-10)
3. Key features: List the main geometric features visible
4. Symmetry: Is the part symmetric? Along which axes?
5. Estimated dimensions: Rough mm estimates for L x W x H
6. Suggested manufacturing: Best manufacturing method
7. Material suggestion: Most likely material

Respond in JSON format:
{
    "part_type": "...",
    "complexity": 5,
    "features": ["feature1", "feature2"],
    "symmetry": {"x": true, "y": false, "z": false},
    "dimensions_mm": {"length": 100, "width": 50, "height": 30},
    "manufacturing": "cnc",
    "material": "aluminum"
}"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ImageAnalysisResult:
    """Result from analyzing an image for CAD generation."""
    description: str                          # plain text description for pipeline
    structured_plan: Optional[dict] = None   # same schema as DESIGN_PLAN_PROMPT JSON
    part_type: Optional[str] = None
    complexity: Optional[int] = None
    features: list[str] = field(default_factory=list)
    dimensions_mm: Optional[dict] = None
    manufacturing: Optional[str] = None
    material: Optional[str] = None
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _call_gemini_vision(b64_data: str, mime_type: str, prompt: str) -> str:
    """Send an image + prompt to Gemini Vision and return the text response."""
    import httpx

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not configured. Set it in .env")

    model = "gemini-2.5-flash"
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )
    payload = {
        "contents": [{
            "parts": [
                {"inline_data": {"mime_type": mime_type, "data": b64_data}},
                {"text": prompt},
            ]
        }],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 2048},
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(url, params={"key": GEMINI_API_KEY}, json=payload)

    if resp.status_code != 200:
        logger.error(f"Gemini Vision error: {resp.status_code} {resp.text[:200]}")
        raise ValueError(f"Gemini Vision API error: {resp.status_code}")

    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise ValueError("No text response from Gemini Vision")


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    return text.strip()


# ---------------------------------------------------------------------------
# Core analysis function
# ---------------------------------------------------------------------------

async def analyze_image_for_cad(
    image_path: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    filename: str = "image.png",
    extract_description: bool = True,
) -> ImageAnalysisResult:
    """
    Analyze an image using Gemini Vision and extract a CAD-ready plan.

    The structured_plan uses the same JSON schema as DESIGN_PLAN_PROMPT so it
    can be injected directly into the CadQuery generation pipeline.
    """
    import json

    if image_path:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        filename = filename or os.path.basename(image_path)

    if not image_bytes:
        raise ValueError("Either image_path or image_bytes must be provided")

    processed_bytes, mime_type = _prepare_image_bytes(image_bytes, filename)
    b64_data = base64.b64encode(processed_bytes).decode("utf-8")

    if extract_description:
        structured_plan = None
        description = ""
        confidence = 0.7

        try:
            raw = await _call_gemini_vision(b64_data, mime_type, IMAGE_TO_CAD_PLAN_PROMPT)
            cleaned = _strip_json_fences(raw)
            structured_plan = json.loads(cleaned)

            # Extract the plain-text description from the plan
            description = structured_plan.pop("text_description", "").strip()
            if not description:
                # Build one from the plan fields as fallback
                dims = structured_plan.get("dims", {})
                dims_str = ", ".join(f"{k}={v}mm" for k, v in dims.items())
                feats = "; ".join(
                    f.get("desc", "") for f in structured_plan.get("features", [])
                )
                description = (
                    f"{structured_plan.get('part_name', 'Part')}: "
                    f"{structured_plan.get('base', '')} base. "
                    f"Dimensions: {dims_str}. Features: {feats}. "
                    f"{structured_plan.get('notes', '')}"
                ).strip()
            confidence = 0.85
        except Exception as e:
            logger.warning(f"Structured extraction failed, falling back to prose: {e}")
            structured_plan = None
            try:
                # Fall back to plain prose description
                description = (await _call_gemini_vision(
                    b64_data, mime_type,
                    "Describe this mechanical part in one detailed paragraph for a CAD system. "
                    "Include shape, dimensions in mm (estimate if needed), holes, features, material."
                )).strip()
                confidence = 0.65
            except Exception as e2:
                raise ValueError(f"Image analysis failed: {e2}") from e2

        result = ImageAnalysisResult(
            description=description,
            structured_plan=structured_plan,
            confidence=confidence,
        )
        if structured_plan:
            result.part_type = structured_plan.get("part_name")
            result.dimensions_mm = structured_plan.get("dims")
            result.features = [f.get("desc", "") for f in structured_plan.get("features", [])]

    else:
        raw = await _call_gemini_vision(b64_data, mime_type, IMAGE_ANALYSIS_PROMPT)
        result = ImageAnalysisResult(description="", confidence=0.7)
        try:
            analysis = json.loads(_strip_json_fences(raw))
            result.part_type = analysis.get("part_type")
            result.complexity = analysis.get("complexity")
            result.features = analysis.get("features", [])
            result.dimensions_mm = analysis.get("dimensions_mm")
            result.manufacturing = analysis.get("manufacturing")
            result.material = analysis.get("material")
            result.confidence = 0.8
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse structured analysis: {e}")

    logger.info(f"Image analysis complete: {result.description[:80]}...")
    return result


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

async def image_to_cad_description(
    image_path: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    filename: str = "image.png",
    additional_context: str = "",
) -> str:
    """
    Convert an image to a text description suitable for the text-to-CAD pipeline.
    """
    result = await analyze_image_for_cad(
        image_path=image_path,
        image_bytes=image_bytes,
        filename=filename,
        extract_description=True,
    )
    description = result.description
    if additional_context:
        description += f"\n\nAdditional requirements: {additional_context}"
    return description


async def image_to_cad_plan(
    image_path: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    filename: str = "image.png",
    additional_context: str = "",
) -> ImageAnalysisResult:
    """
    Convert an image to a full structured CAD plan.

    Returns ImageAnalysisResult with both `description` (text) and
    `structured_plan` (dict — same schema as DESIGN_PLAN_PROMPT).
    """
    result = await analyze_image_for_cad(
        image_path=image_path,
        image_bytes=image_bytes,
        filename=filename,
        extract_description=True,
    )
    if additional_context and result.description:
        result.description += f"\n\nAdditional requirements: {additional_context}"
    return result
