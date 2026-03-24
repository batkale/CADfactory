"""
Layer 1.5 — Reference Matcher
Categorizes the user's intent into a generalized mechanical pattern/template.
Used to inject standard-specific DNA into the generation layers.
"""

import logging
from typing import Optional, List
from pydantic import BaseModel
from services.mechanical_registry import MECHANICAL_REGISTRY

logger = logging.getLogger(__name__)

class MatchResult(BaseModel):
    template_key: Optional[str]  # e.g. "RADIAL_ASSEMBLY", "ENCLOSURE"
    confidence: float
    reasoning: str
    specific_guidance: str = ""  # Object-specific hints (e.g. "Soap dispenser requires 28mm neck")
    suggested_standards: List[str] = []

_MATCHER_SYSTEM_PROMPT = """
You are Layer 1.5 of a precision CAD pipeline: the Mechanical Reference Matcher.

Your task is to take a user's prompt and a Context Bundle (from Layer 1) and match it to
one of the following GENERAL MECHANICAL PATTERNS:

1. RADIAL_ASSEMBLY: Symmetrical repetition around a hub (fans, cogs, fidget spinners, wheels).
2. ENCLOSURE: Boxes, cases, shells, hollow containers, and HIGH-FIDELITY CONSUMER ASSEMBLIES (dispensers, sprayers).
3. STRUCTURAL_BRACKET: L-brackets, joints, mounting plates, shelves.
4. FLUID_CONNECTOR: Pipes, nozzles, fittings, adapters.

Return "null" for the key if no specific pattern matches well.

In addition to the key, provide "specific_guidance" for the object. 
For CONSUMER PRODUCTS (e.g. soap dispenser), you MUST mandate a multi-part assembly decomposition:
- "Soap Dispenser": guidance="Decompose into 5 overlapping parts: Body (rounded), 28mm Neck, cylindrical Pump Hub, curved Spout, and top Plunger Button."
- "Fidget Spinner": guidance="Decompose into Hub + 3 Radial Arms. Hub MUST have 22.1mm bearing bore. Arms MUST overlap hub by 2mm."
- "Cooling Fan": guidance="Decompose into Central Hub + 7 Aerodynamic Blades. Blades MUST be curved_prisms with 2mm hub overlap."

Output ONLY valid JSON.
{
  "template_key": "RADIAL_ASSEMBLY" | "ENCLOSURE" | "STRUCTURAL_BRACKET" | "FLUID_CONNECTOR" | null,
  "confidence": 0.95,
  "reasoning": "...",
  "specific_guidance": "...",
  "suggested_standards": ["608_bearing", "soap_pump_neck", ...]
}
"""

def match_reference(prompt: str, object_category: str) -> MatchResult:
    """
    Match the prompt against the mechanical registry.
    """
    from services.claude_cad import _generate_content, MODEL_FLASH
    from services.script_utils import parse_json_response
    
    user_input = f"Prompt: {prompt}\nCategory: {object_category}"
    
    try:
        raw = _generate_content(MODEL_FLASH, _MATCHER_SYSTEM_PROMPT, user_input)
        data = parse_json_response(raw)
        
        if not data or not isinstance(data, dict):
            return MatchResult(template_key=None, confidence=0.0, reasoning="API error")
            
        key = data.get("template_key")
        if key not in MECHANICAL_REGISTRY:
            key = None
            
        return MatchResult(
            template_key=key,
            confidence=float(data.get("confidence", 0.0)),
            reasoning=str(data.get("reasoning", "")),
            specific_guidance=str(data.get("specific_guidance", "")),
            suggested_standards=list(data.get("suggested_standards") or [])
        )
    except Exception as exc:
        logger.error(f"Reference matcher failed: {exc}")
        return MatchResult(template_key=None, confidence=0.0, reasoning=f"Error: {exc}", specific_guidance="")
