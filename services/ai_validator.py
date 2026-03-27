"""
AI Validator — Post-generation validation that checks whether the
generated CAD model actually matches what the user requested.

Uses AI to:
1. Research what the object SHOULD look like (web-grounded)
2. Analyze the generated script's geometry against expectations
3. Produce a confidence score and specific corrections
4. Auto-correct the script if confidence is too low

Location: cadfactory-backend/services/ai_validator.py
"""

from __future__ import annotations

import logging
import math
import re
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── Models ────────────────────────────────────────────────────────────────────

class GeometricExpectation(BaseModel):
    """What the object SHOULD have based on AI research."""
    object_name: str = ""
    description: str = ""
    expected_features: List[str] = Field(default_factory=list)
    expected_holes: List[dict] = Field(default_factory=list)  # [{purpose, count, diameter_mm}]
    expected_cavities: List[dict] = Field(default_factory=list)  # [{purpose, shape}]
    expected_symmetry: Optional[str] = None
    expected_dimensions: dict = Field(default_factory=dict)
    critical_features: List[str] = Field(default_factory=list)
    reference_url: Optional[str] = None


class ValidationIssue(BaseModel):
    severity: str  # "critical", "warning", "info"
    feature: str
    expected: str
    actual: str
    fix_suggestion: str


class ValidationResult(BaseModel):
    """Result of validating a generated script against expectations."""
    confidence: float = 0.0  # 0-1, how well the script matches expectations
    issues: List[ValidationIssue] = Field(default_factory=list)
    missing_holes: List[dict] = Field(default_factory=list)
    missing_features: List[str] = Field(default_factory=list)
    corrections_applied: int = 0
    corrected_script: Optional[str] = None
    validation_notes: str = ""


# ── Research: What should this object look like? ──────────────────────────────

_RESEARCH_PROMPT = """You are an expert industrial designer and product researcher.
Given an object description, research and describe the PRECISE geometric features
that this object MUST have to be functional and recognizable.

Focus on:
1. HOLES: Count, purpose, size (diameter in mm), and position of every hole/bore
2. CAVITIES: Internal hollows, channels, slots that are functionally required
3. SYMMETRY: Radial, bilateral, or none
4. CRITICAL FEATURES: Features without which the object would not function
5. DIMENSIONS: Real-world standard dimensions in mm

Examples of critical features:
- Fidget spinner: MUST have 3 bearing holes (22mm OD bore) + 1 center bearing hole
- Extension cable with 3 plugs: MUST have 3 rectangular socket openings (each ~23x14mm for Type G, ~8x18mm for Europlug)
- Soap dispenser: MUST have hollow body cavity, pump mechanism tube channel, nozzle opening
- Gear: MUST have center bore hole + involute teeth

Object: {description}

Return ONLY valid JSON:
{{
  "object_name": "canonical name",
  "description": "one-line geometric description",
  "expected_features": ["feature 1", "feature 2"],
  "expected_holes": [
    {{"purpose": "center bearing", "count": 1, "diameter_mm": 22.0, "position": "center"}},
    {{"purpose": "arm bearings", "count": 3, "diameter_mm": 22.0, "position": "at each arm tip"}}
  ],
  "expected_cavities": [
    {{"purpose": "hollow body for liquid", "shape": "cylinder matching body minus wall"}}
  ],
  "expected_symmetry": "3-fold radial | bilateral | none",
  "expected_dimensions": {{"overall_diameter": 76, "thickness": 8}},
  "critical_features": ["center bearing hole", "3 arm bearing holes", "3 arms"]
}}
"""


_VALIDATE_PROMPT = """You are a CAD quality engineer. Analyze this CadQuery script and
determine if it correctly implements the expected geometric features.

EXPECTED OBJECT: {object_name}
EXPECTED FEATURES:
{expected_features}

EXPECTED HOLES:
{expected_holes}

EXPECTED CAVITIES:
{expected_cavities}

CRITICAL FEATURES THAT MUST EXIST:
{critical_features}

CADQUERY SCRIPT TO VALIDATE:
```python
{script}
```

Analyze the script carefully:
1. Count every .cut(), .cboreHole(), .hole(), .cskHole() call — these create holes
2. Check if subtract operations in CSG match expected holes/cavities
3. Verify dimensions match expectations (within 20% tolerance)
4. Check that ALL critical features are present

Return ONLY valid JSON:
{{
  "confidence": 0.85,
  "issues": [
    {{
      "severity": "critical",
      "feature": "arm bearing holes",
      "expected": "3 holes of 22mm diameter at arm tips",
      "actual": "no bearing holes found in script",
      "fix_suggestion": "Add .cut() operations for 3x cylinder(r=11, h=8) at arm tip positions"
    }}
  ],
  "missing_holes": [
    {{"purpose": "arm bearings", "count": 3, "diameter_mm": 22.0, "position_hint": "at arm tips"}}
  ],
  "missing_features": ["arm bearing holes"],
  "validation_notes": "Script creates hub and arms but lacks bearing seat cuts"
}}
"""


_CORRECTION_PROMPT = """You are a senior CadQuery engineer. Fix this script to add the missing features.

ORIGINAL SCRIPT:
```python
{script}
```

MISSING FEATURES TO ADD:
{missing_features}

MISSING HOLES TO ADD:
{missing_holes}

VALIDATION ISSUES:
{issues}

RULES:
1. Keep ALL existing geometry — do NOT remove or reshape existing parts
2. ADD the missing holes/features using .cut() operations on `result`
3. Holes must use cylinder primitives with correct dimensions
4. Position holes correctly based on existing geometry
5. For bearing holes: use exact clearance fit (e.g., r=11.05 for 608 bearing)
6. For socket openings: use box cuts with correct dimensions
7. Maintain show_object(result) at the end
8. Only import cadquery and math — no other libraries

Return ONLY the corrected Python script, no explanation, no markdown fences.
"""


def research_object_expectations(description: str) -> GeometricExpectation:
    """
    Use AI to research what geometric features the described object should have.
    This is the 'ground truth' that we validate against.
    """
    from services.claude_cad import _generate_content, MODEL_FLASH
    from services.script_utils import parse_json_response

    prompt = _RESEARCH_PROMPT.format(description=description)

    try:
        raw = _generate_content(MODEL_FLASH, "", prompt)
        data = parse_json_response(raw)
        if not data or not isinstance(data, dict):
            logger.warning("Object research returned no valid data")
            return GeometricExpectation()

        return GeometricExpectation(
            object_name=str(data.get("object_name", "")),
            description=str(data.get("description", "")),
            expected_features=data.get("expected_features", []),
            expected_holes=data.get("expected_holes", []),
            expected_cavities=data.get("expected_cavities", []),
            expected_symmetry=data.get("expected_symmetry"),
            expected_dimensions=data.get("expected_dimensions", {}),
            critical_features=data.get("critical_features", []),
        )
    except Exception as e:
        logger.error(f"Object research failed: {e}")
        return GeometricExpectation()


def validate_script_against_expectations(
    script: str,
    expectations: GeometricExpectation,
) -> ValidationResult:
    """
    Use AI to validate whether a generated script matches the expected features.
    Also performs static analysis to count holes/cuts.
    """
    from services.claude_cad import _generate_content, MODEL_FLASH
    from services.script_utils import parse_json_response

    # Static analysis: count cut operations in script
    cut_count = len(re.findall(r'\.cut\(', script))
    hole_count = len(re.findall(r'\.(hole|cboreHole|cskHole)\(', script))
    subtract_count = len(re.findall(r'op.*=.*"subtract"', script))
    total_subtractive = cut_count + hole_count + subtract_count

    expected_total_holes = sum(h.get("count", 1) for h in expectations.expected_holes)
    expected_cavities = len(expectations.expected_cavities)

    # Quick static check: if we expect holes but have none, that's already bad
    static_confidence = 1.0
    static_issues = []

    if expected_total_holes > 0 and total_subtractive == 0:
        static_confidence = 0.2
        static_issues.append(
            f"Expected {expected_total_holes} holes but script has 0 subtractive operations"
        )
    elif expected_total_holes > total_subtractive:
        static_confidence = max(0.3, total_subtractive / max(expected_total_holes, 1))
        static_issues.append(
            f"Expected {expected_total_holes} holes but only found ~{total_subtractive} subtractive ops"
        )

    # AI validation for deeper analysis
    holes_text = "\n".join(
        f"- {h.get('count', 1)}x {h.get('purpose', '?')}: "
        f"Ø{h.get('diameter_mm', '?')}mm at {h.get('position', '?')}"
        for h in expectations.expected_holes
    ) or "None expected"

    cavities_text = "\n".join(
        f"- {c.get('purpose', '?')}: {c.get('shape', '?')}"
        for c in expectations.expected_cavities
    ) or "None expected"

    prompt = _VALIDATE_PROMPT.format(
        object_name=expectations.object_name,
        expected_features="\n".join(f"- {f}" for f in expectations.expected_features),
        expected_holes=holes_text,
        expected_cavities=cavities_text,
        critical_features="\n".join(f"- {f}" for f in expectations.critical_features),
        script=script,
    )

    try:
        raw = _generate_content(MODEL_FLASH, "", prompt)
        data = parse_json_response(raw)
        if not data or not isinstance(data, dict):
            # Fall back to static analysis only
            return ValidationResult(
                confidence=static_confidence,
                validation_notes="AI validation failed; static analysis only. " + "; ".join(static_issues),
            )

        ai_confidence = float(data.get("confidence", 0.5))
        # Blend static and AI confidence (static is more reliable for hole counting)
        blended_confidence = min(static_confidence, ai_confidence)

        issues = []
        for iss in data.get("issues", []):
            try:
                issues.append(ValidationIssue(
                    severity=str(iss.get("severity", "warning")),
                    feature=str(iss.get("feature", "")),
                    expected=str(iss.get("expected", "")),
                    actual=str(iss.get("actual", "")),
                    fix_suggestion=str(iss.get("fix_suggestion", "")),
                ))
            except Exception:
                pass

        return ValidationResult(
            confidence=blended_confidence,
            issues=issues,
            missing_holes=data.get("missing_holes", []),
            missing_features=data.get("missing_features", []),
            validation_notes=data.get("validation_notes", ""),
        )

    except Exception as e:
        logger.error(f"AI validation failed: {e}")
        return ValidationResult(
            confidence=static_confidence,
            validation_notes=f"AI validation error: {e}. Static: {'; '.join(static_issues)}",
        )


def correct_script(
    script: str,
    validation: ValidationResult,
    expectations: GeometricExpectation,
) -> Optional[str]:
    """
    Use AI to fix a script that failed validation.
    Returns the corrected script, or None if correction fails.
    """
    from services.claude_cad import _generate_content, MODEL_FLASH
    from services.script_utils import extract_python_code

    if not validation.missing_holes and not validation.missing_features:
        return None  # Nothing to fix

    issues_text = "\n".join(
        f"- [{iss.severity}] {iss.feature}: expected {iss.expected}, "
        f"got {iss.actual}. Fix: {iss.fix_suggestion}"
        for iss in validation.issues
    ) or "No specific issues"

    holes_text = "\n".join(
        f"- {h.get('count', 1)}x {h.get('purpose', '?')}: "
        f"Ø{h.get('diameter_mm', '?')}mm at {h.get('position_hint', h.get('position', '?'))}"
        for h in validation.missing_holes
    ) or "None"

    features_text = "\n".join(f"- {f}" for f in validation.missing_features) or "None"

    prompt = _CORRECTION_PROMPT.format(
        script=script,
        missing_features=features_text,
        missing_holes=holes_text,
        issues=issues_text,
    )

    try:
        raw = _generate_content(MODEL_FLASH, "", prompt)
        corrected = extract_python_code(raw)
        if not corrected:
            # The model might have returned raw code without fences
            corrected = raw.strip()
            if not corrected.startswith("import"):
                return None

        logger.info(f"Script corrected: added {len(validation.missing_features)} features, "
                     f"{len(validation.missing_holes)} hole groups")
        return corrected
    except Exception as e:
        logger.error(f"Script correction failed: {e}")
        return None


async def validate_and_correct(
    script: str,
    description: str,
    max_correction_rounds: int = 2,
    expectations=None,
) -> tuple[str, ValidationResult]:
    """
    Full validation + correction loop.

    1. Research what the object should look like
    2. Validate the script against expectations
    3. If confidence < threshold, correct and re-validate
    4. Return (final_script, final_validation)
    """
    CONFIDENCE_THRESHOLD = 0.7

    # Step 1: Research expectations (reuse pre-computed if available)
    if expectations is None:
        expectations = research_object_expectations(description)
    if not expectations.critical_features and not expectations.expected_holes:
        # No specific expectations — skip validation
        return script, ValidationResult(
            confidence=0.8,
            validation_notes="No specific geometric expectations identified; skipping validation",
        )

    logger.info(
        f"[Validator] Expectations for '{expectations.object_name}': "
        f"{len(expectations.expected_holes)} hole groups, "
        f"{len(expectations.critical_features)} critical features"
    )

    # Step 2: Validate
    validation = validate_script_against_expectations(script, expectations)
    logger.info(
        f"[Validator] Initial confidence: {validation.confidence:.2f}, "
        f"issues: {len(validation.issues)}, "
        f"missing holes: {len(validation.missing_holes)}"
    )

    # Step 3: Correction loop
    current_script = script
    for round_num in range(max_correction_rounds):
        if validation.confidence >= CONFIDENCE_THRESHOLD:
            break

        logger.info(f"[Validator] Correction round {round_num + 1} (conf={validation.confidence:.2f})")

        corrected = correct_script(current_script, validation, expectations)
        if corrected is None:
            logger.warning("[Validator] Correction returned None — stopping")
            break

        # Re-validate the corrected script
        current_script = corrected
        validation = validate_script_against_expectations(current_script, expectations)
        validation.corrections_applied = round_num + 1
        validation.corrected_script = current_script

        logger.info(
            f"[Validator] After round {round_num + 1}: confidence={validation.confidence:.2f}"
        )

    return current_script, validation
