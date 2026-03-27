"""
Image-to-CAD generation service.

Accepts an image (photo, sketch, napkin drawing) and uses Gemini Vision
to extract geometric descriptions, then feeds them into the existing
semantic decomposer -> CSG builder pipeline.

Inspired by GenCAD's image-conditioned CAD generation approach, but
leveraging the existing Gemini API integration and text-to-CAD pipeline.
"""

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

IMAGE_TO_CAD_PROMPT = """You are an expert mechanical engineer and CAD designer.

Analyze this image of a physical part, sketch, or technical drawing.

Extract a detailed natural language description that can be used to generate
a 3D CAD model. Be specific about:

1. **Overall shape**: What is the primary geometry? (box, cylinder, L-bracket, etc.)
2. **Dimensions**: Estimate dimensions in mm based on visual cues, proportions,
   and any visible measurements. If no scale reference, estimate reasonable
   dimensions for the part type.
3. **Features**: List all visible features:
   - Holes (through, counterbore, countersink) with estimated diameters
   - Fillets and chamfers with estimated radii
   - Slots, grooves, pockets
   - Bosses, ribs, walls
   - Threads (M-size if identifiable)
4. **Material hint**: If identifiable (metal, plastic, wood, etc.)
5. **Manufacturing method**: Best guess (CNC, 3D printed, injection molded, etc.)

Output a SINGLE paragraph description suitable for a text-to-CAD system.
Start directly with the part description, no preamble.

Example output:
"A rectangular mounting bracket 80mm x 40mm x 3mm thick with two M4 through
holes spaced 60mm apart centered along the length, 10mm from each edge.
The bracket has a 90-degree bend 20mm from one end creating an L-shape.
All edges have 1mm chamfers. Material: aluminum sheet metal."
"""

IMAGE_ANALYSIS_PROMPT = """You are an expert mechanical engineer analyzing a CAD part image.

Analyze this image and provide a structured assessment:

1. **Part identification**: What type of part is this?
2. **Geometric complexity**: Simple (1-3) / Moderate (4-6) / Complex (7-10)
3. **Key features**: List the main geometric features visible
4. **Symmetry**: Is the part symmetric? Along which axes?
5. **Estimated dimensions**: Rough mm estimates for L x W x H
6. **Suggested manufacturing**: Best manufacturing method
7. **Material suggestion**: Most likely material

Respond in JSON format:
{
    "part_type": "...",
    "complexity": 5,
    "features": ["feature1", "feature2"],
    "symmetry": {"x": true, "y": false, "z": false},
    "dimensions_mm": {"length": 100, "width": 50, "height": 30},
    "manufacturing": "cnc",
    "material": "aluminum"
}
"""


@dataclass
class ImageAnalysisResult:
    """Result from analyzing an image for CAD generation."""
    description: str
    part_type: Optional[str] = None
    complexity: Optional[int] = None
    features: list[str] = field(default_factory=list)
    dimensions_mm: Optional[dict] = None
    manufacturing: Optional[str] = None
    material: Optional[str] = None
    confidence: float = 0.0


def _encode_image_base64(image_path: str) -> tuple[str, str]:
    """Read and base64-encode an image file. Returns (b64_data, mime_type)."""
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/png"

    with open(image_path, "rb") as f:
        data = f.read()

    return base64.b64encode(data).decode("utf-8"), mime_type


def _encode_image_bytes_base64(image_bytes: bytes, filename: str = "image.png") -> tuple[str, str]:
    """Base64-encode image bytes. Returns (b64_data, mime_type)."""
    mime_type, _ = mimetypes.guess_type(filename)
    if mime_type is None:
        mime_type = "image/png"

    return base64.b64encode(image_bytes).decode("utf-8"), mime_type


async def analyze_image_for_cad(
    image_path: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    filename: str = "image.png",
    extract_description: bool = True,
) -> ImageAnalysisResult:
    """
    Analyze an image using Gemini Vision and extract a CAD-suitable description.

    Args:
        image_path: Path to an image file (PNG, JPG, etc.)
        image_bytes: Raw image bytes (alternative to image_path)
        filename: Original filename (for MIME type detection)
        extract_description: If True, extract a text-to-CAD description.
                           If False, extract structured analysis only.

    Returns:
        ImageAnalysisResult with description and optional metadata.
    """
    import httpx
    import json

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not configured. Set it in .env")

    # Encode image
    if image_path:
        b64_data, mime_type = _encode_image_base64(image_path)
    elif image_bytes:
        b64_data, mime_type = _encode_image_bytes_base64(image_bytes, filename)
    else:
        raise ValueError("Either image_path or image_bytes must be provided")

    prompt = IMAGE_TO_CAD_PROMPT if extract_description else IMAGE_ANALYSIS_PROMPT

    # Build Gemini Vision request
    model = "gemini-2.5-flash"
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )

    payload = {
        "contents": [{
            "parts": [
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": b64_data,
                    }
                },
                {"text": prompt},
            ]
        }],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 1024,
        },
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            url,
            params={"key": GEMINI_API_KEY},
            json=payload,
        )

    if resp.status_code != 200:
        logger.error(f"Gemini Vision API error: {resp.status_code} {resp.text[:200]}")
        raise ValueError(f"Gemini Vision API error: {resp.status_code}")

    data = resp.json()
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise ValueError("No text response from Gemini Vision")

    result = ImageAnalysisResult(description=text.strip(), confidence=0.7)

    # If structured analysis, parse JSON
    if not extract_description:
        try:
            # Extract JSON from response (may be wrapped in markdown)
            json_str = text
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]

            analysis = json.loads(json_str.strip())
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


async def image_to_cad_description(
    image_path: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    filename: str = "image.png",
    additional_context: str = "",
) -> str:
    """
    Convert an image to a text description suitable for the text-to-CAD pipeline.

    This is the main entry point for image-to-CAD generation. The returned
    description can be passed directly to the existing generate endpoint.

    Args:
        image_path: Path to image file.
        image_bytes: Raw image bytes.
        filename: Original filename.
        additional_context: Extra context from the user (e.g., "make it 50mm tall").

    Returns:
        A natural language description suitable for text-to-CAD generation.
    """
    result = await analyze_image_for_cad(
        image_path=image_path,
        image_bytes=image_bytes,
        filename=filename,
        extract_description=True,
    )

    description = result.description

    # Append user context if provided
    if additional_context:
        description += f"\n\nAdditional requirements: {additional_context}"

    return description
