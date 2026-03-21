"""
Utility functions for extracting Python code from Claude responses
and parsing BOM information from generated CadQuery scripts.

Location: cadfactory-backend/services/script_utils.py
"""

import re
import json
from typing import Optional


def extract_python_code(response_text: str) -> str:
    """
    Extract Python code from Claude's response.
    
    Handles three formats:
    1. Raw Python code (no markdown)
    2. ```python ... ``` blocks
    3. ``` ... ``` blocks
    
    Returns clean Python code string.
    """
    # Try to extract from ```python ... ``` block first
    pattern = r"```python\s*\n(.*?)```"
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Try generic ``` ... ``` block
    pattern = r"```\s*\n(.*?)```"
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Assume the entire response is code (we asked for no markdown)
    # Strip any leading/trailing whitespace and potential markdown artifacts
    code = response_text.strip()
    
    # Remove any leading "Here is..." or similar preamble
    lines = code.split("\n")
    start_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from ") or stripped.startswith("#"):
            start_idx = i
            break
    
    return "\n".join(lines[start_idx:]).strip()


def validate_script(script: str) -> tuple[bool, list[str]]:
    """
    Basic validation of a CadQuery script before execution.
    
    Returns (is_valid, list_of_warnings).
    """
    warnings = []
    
    # Must contain cadquery import
    if "import cadquery" not in script and "from cadquery" not in script:
        warnings.append("Missing cadquery import — will be injected automatically")
    
    # Must define 'result' variable
    if "result" not in script:
        warnings.append("No 'result' variable found — script may not export correctly")
    
    # Check for dangerous imports/operations
    dangerous_patterns = [
        (r"\bos\.system\b", "os.system call detected"),
        (r"\bsubprocess\b", "subprocess module detected"),
        (r"\b__import__\b", "__import__ call detected"),
        (r"\beval\b", "eval() call detected"),
        (r"\bexec\b", "exec() call detected — only allowed for CadQuery execution"),
        (r"\bopen\s*\(", "file open() detected"),
        (r"\brequests\b", "requests module detected"),
        (r"\burllib\b", "urllib module detected"),
        (r"\bsocket\b", "socket module detected"),
        (r"\bshutil\b", "shutil module detected"),
    ]

    has_dangerous = False
    for pattern, msg in dangerous_patterns:
        if re.search(pattern, script):
            if "exec" in pattern:
                continue  # exec is used by some CadQuery patterns
            warnings.append(f"BLOCKED: {msg}")
            has_dangerous = True

    # Check for zero/negative literal dimensions in .box() calls — makeBox crashes on these
    for m in re.finditer(r'\.box\s*\(([^)]+)\)', script):
        args_str = m.group(1)
        args = [a.strip() for a in args_str.split(',')]
        for arg in args[:3]:  # only first 3 positional args (l, w, h)
            try:
                v = float(arg)
                if v <= 0:
                    warnings.append(
                        f"INVALID: .box() dimension {arg} ≤ 0 — makeBox().Shape() will crash. "
                        "All box dimensions must be strictly positive (> 0)."
                    )
                    has_dangerous = True
            except ValueError:
                pass  # computed value, can't check statically

    # Check for result used before assignment (result.union/cut before result = <shape>)
    lines = script.splitlines()
    result_initialized = False
    for line in lines:
        stripped = line.strip()
        if re.match(r'^result\s*=\s*result\b', stripped):
            if not result_initialized:
                warnings.append(
                    "INVALID: result.union/cut/intersect called before result is initialized. "
                    "First shape must be assigned as: result = <shape>"
                )
                has_dangerous = True
                break
        elif re.match(r'^result\s*=\s*(?!result\b)', stripped):
            result_initialized = True

    # Check for known non-existent CadQuery methods — fail fast before execution
    nonexistent_methods = [
        (r"\.\bthread\s*\(",        "INVALID: .thread() does not exist — use a plain cylinder for the shaft"),
        (r"\.\baddThread\s*\(",     "INVALID: .addThread() does not exist — use a plain cylinder"),
        (r"\.\bcutExtrude\s*\(",    "INVALID: .cutExtrude() does not exist — use .cutBlind(-depth)"),
        (r"\.\bextrudeCut\s*\(",    "INVALID: .extrudeCut() does not exist — use .cutBlind(-depth)"),
        (r"\.\bmakeHelix\s*\(",     "INVALID: .makeHelix() does not exist in CadQuery"),
        (r"\.\bhelix\s*\(",         "INVALID: .helix() does not exist — use Wire.makePolygon for curves"),
        (r"\bfrom cq_warehouse\b",  "INVALID: cq_warehouse is not installed — model screws manually (Example 12)"),
        (r"\bimport cq_warehouse\b","INVALID: cq_warehouse is not installed — model screws manually (Example 12)"),
    ]
    for pattern, msg in nonexistent_methods:
        if re.search(pattern, script):
            warnings.append(msg)
            has_dangerous = True

    is_valid = not has_dangerous
    return is_valid, warnings


def extract_bom_from_script(script: str) -> list[dict]:
    """
    Parse a CadQuery script to extract BOM-relevant information.
    
    Looks for:
    - Parameter comments indicating part names/descriptions
    - Dimensions that imply material volume
    - Bolt hole sizes that imply fastener requirements
    """
    bom_items = []
    
    # Extract the main part (always item 1)
    part_name = "Generated Part"
    
    # Try to find a descriptive comment at the top
    lines = script.split("\n")
    for line in lines[:10]:
        stripped = line.strip()
        if stripped.startswith("# ") and len(stripped) > 5:
            candidate = stripped[2:].strip()
            # Skip generic comments
            if candidate.lower() not in ["imports", "parameters", "cadquery script"]:
                part_name = candidate
                break
    
    bom_items.append({
        "name": part_name,
        "qty": 1,
        "unit_cost": None,  # To be filled by cost analysis
        "manufacturing_method": None,  # Set by user's selection
        "notes": "AI-generated part"
    })
    
    # Detect fasteners from hole patterns
    bolt_patterns = {
        r"\.hole\s*\(\s*3\.2": ("M3 Bolt", "M3"),
        r"\.hole\s*\(\s*2\.2": ("M2 Bolt", "M2"),
        r"\.hole\s*\(\s*2\.7": ("M2.5 Bolt", "M2.5"),
        r"\.hole\s*\(\s*4\.2": ("M4 Bolt", "M4"),
        r"\.hole\s*\(\s*5\.2": ("M5 Bolt", "M5"),
        r"\.hole\s*\(\s*6\.2": ("M6 Bolt", "M6"),
        r"\.cboreHole\s*\(\s*3\.2": ("M3 Socket Head Cap Screw", "M3"),
        r"\.cboreHole\s*\(\s*4\.2": ("M4 Socket Head Cap Screw", "M4"),
        r"\.cboreHole\s*\(\s*5\.2": ("M5 Socket Head Cap Screw", "M5"),
    }
    
    fastener_counts = {}
    for pattern, (name, size) in bolt_patterns.items():
        matches = re.findall(pattern, script)
        if matches:
            key = name
            fastener_counts[key] = fastener_counts.get(key, 0) + len(matches)
    
    for name, qty in fastener_counts.items():
        bom_items.append({
            "name": name,
            "qty": qty,
            "unit_cost": _estimate_fastener_cost(name),
            "manufacturing_method": "Off-the-Shelf",
            "notes": "Detected from hole pattern"
        })
    
    # Detect heat-set inserts
    insert_patterns = {
        r"\.hole\s*\(\s*4\.0": ("M3 Heat-Set Insert", 0.15),
        r"\.hole\s*\(\s*5\.2[^,]*": ("M4 Heat-Set Insert", 0.20),
    }
    
    for pattern, (name, cost) in insert_patterns.items():
        matches = re.findall(pattern, script)
        if matches:
            bom_items.append({
                "name": name,
                "qty": len(matches),
                "unit_cost": cost,
                "manufacturing_method": "Off-the-Shelf",
                "notes": "Detected from insert hole size"
            })
    
    return bom_items


def _estimate_fastener_cost(name: str) -> float:
    """Rough cost estimates for common fasteners (GBP)."""
    costs = {
        "M2 Bolt": 0.03,
        "M2.5 Bolt": 0.04,
        "M3 Bolt": 0.04,
        "M4 Bolt": 0.05,
        "M5 Bolt": 0.07,
        "M6 Bolt": 0.09,
        "M3 Socket Head Cap Screw": 0.06,
        "M4 Socket Head Cap Screw": 0.08,
        "M5 Socket Head Cap Screw": 0.10,
    }
    return costs.get(name, 0.05)


def parse_json_response(response_text: str) -> Optional[dict]:
    """
    Safely parse JSON from Claude's response.
    Handles markdown-wrapped JSON and common formatting issues.
    """
    text = response_text.strip()
    
    # Remove markdown code blocks if present
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    
    text = text.strip()
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None
    return None