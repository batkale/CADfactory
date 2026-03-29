"""
Parametric Template Library — core "Golden Path" for %100 reliable CAD.
Each function returns a string of CadQuery code.

Location: cadfactory-backend/services/templates.py
"""
from typing import List


def make_bolt(params: dict) -> str:
    """Standard bolt/screw template."""
    d = params.get("diameter", 6.0) or 6.0
    length = params.get("length", 30.0) or 30.0
    head_h = d * 1.0  # Approx proportion
    head_d = d * 1.6
    hex_size = d * 0.8

    return f"""import cadquery as cq
# Generated via make_bolt template
d = {d:.2f}
length = {length:.2f}
head_h = {head_h:.2f}
head_d = {head_d:.2f}
socket = {hex_size:.2f}

# Head
result = cq.Workplane("XY").circle(head_d/2).extrude(head_h)
# Shank
result = result.faces("<Z").workplane().circle(d/2).extrude(length)
# Socket
result = result.faces(">Z").workplane().polygon(6, socket).cutBlind(-head_h*0.6)

show_object(result)
"""

def make_enclosure(params: dict) -> str:
    """Simple project box/enclosure template."""
    length = params.get("length", 100.0) or 100.0
    w = params.get("width", 50.0) or 50.0
    h = params.get("height", 30.0) or 30.0
    t = params.get("wall_thickness", 2.0) or 2.0

    return f"""import cadquery as cq
# Generated via make_enclosure template
result = (
    cq.Workplane("XY")
    .box({length:.2f}, {w:.2f}, {h:.2f})
    .faces(">Z")
    .shell(-{t:.2f})
)

show_object(result)
"""

def make_bracket(params: dict) -> str:
    """L-shaped mounting bracket template."""
    base_l = params.get("base_length", 50.0) or 50.0
    side_h = params.get("side_height", 40.0) or 40.0
    width = params.get("width", 30.0) or 30.0
    thickness = params.get("thickness", 3.0) or 3.0
    hole_d = params.get("hole_diameter", 4.0) or 4.0

    return f"""import cadquery as cq
# Generated via make_bracket template
result = (
    cq.Workplane("XY")
    .box({base_l:.2f}, {width:.2f}, {thickness:.2f})
    .faces(">Z").workplane().center(0, {width/2 - thickness/2:.2f})
    .box({base_l:.2f}, {thickness:.2f}, {side_h:.2f}, combine=True)
)
# Add mounting holes
result = (
    result.faces("<Z").workplane()
    .rect({base_l*0.6:.2f}, {width*0.6:.2f}, forConstruction=True)
    .vertices()
    .hole({hole_d:.2f})
)

show_object(result)
"""

def make_plate_with_holes(params: dict) -> str:
    """Flat plate with a rectangular pattern of holes."""
    length = params.get("length", 100.0) or 100.0
    w = params.get("width", 100.0) or 100.0
    t = params.get("thickness", 5.0) or 5.0
    hole_d = params.get("hole_diameter", 5.0) or 5.0
    spacing_x = params.get("hole_spacing_x", 80.0) or 80.0
    spacing_y = params.get("hole_spacing_y", 80.0) or 80.0

    return f"""import cadquery as cq
# Generated via make_plate_with_holes template
result = (
    cq.Workplane("XY")
    .box({length:.2f}, {w:.2f}, {t:.2f})
    .faces(">Z").workplane()
    .rect({spacing_x:.2f}, {spacing_y:.2f}, forConstruction=True)
    .vertices()
    .hole({hole_d:.2f})
)

show_object(result)
"""

def make_nema_motor_mount(params: dict) -> str:
    """NEMA-standard motor mounting plate (NEMA 17, 23, etc.)."""
    size = params.get("nema_size", 17) # 17 or 23
    plate_size = 42.3 if size == 17 else 56.4
    bolt_circle = 31.0 if size == 17 else 47.1
    hole_d = 3.2 if size == 17 else 5.2
    thickness = params.get("thickness", 5.0)
    central_bore = 23.0 if size == 17 else 38.2

    return f"""import cadquery as cq
# Generated via make_nema_motor_mount template (NEMA {size})
plate = {plate_size:.2f}
thickness = {thickness:.2f}
bolt_circle = {bolt_circle:.2f}
central_bore = {central_bore:.2f}

result = cq.Workplane("XY").box(plate, plate, thickness)
# Central bore
result = result.faces(">Z").workplane().hole(central_bore)
# 4x mounting holes
result = (
    result.faces(">Z").workplane()
    .rect(bolt_circle, bolt_circle, forConstruction=True)
    .vertices()
    .hole({hole_d:.2f})
)

show_object(result)
"""

def make_standoff(params: dict) -> str:
    """Hex or round standoff / spacer template."""
    h = params.get("height", 10.0)
    d = params.get("diameter", 6.0)
    hole_d = params.get("hole_diameter", 3.2)
    is_hex = params.get("is_hex", True)

    shape_expr = f"polygon(6, {d:.2f})" if is_hex else f"circle({d/2:.2f})"

    return f"""import cadquery as cq
# Generated via make_standoff template
result = (
    cq.Workplane("XY")
    .{shape_expr}
    .extrude({h:.2f})
    .faces(">Z").workplane()
    .hole({hole_d:.2f})
)

show_object(result)
"""

def validate_template_params(name: str, params: dict) -> List[str]:
    """Check for physically impossible template parameters."""
    errors = []
    if name == "bolt":
        d = params.get("diameter", 6.0) or 6.0
        bolt_length = params.get("length", 30.0) or 30.0
        if d > bolt_length:
            errors.append(f"Bolt diameter ({d}mm) cannot be larger than length ({bolt_length}mm)")

    elif name == "enclosure":
        enc_length = params.get("length", 100.0) or 100.0
        w = params.get("width", 50.0) or 50.0
        h = params.get("height", 30.0) or 30.0
        t = params.get("wall_thickness", 2.0) or 2.0
        if t * 2 >= min(enc_length, w, h):
            errors.append(f"Wall thickness ({t}mm) is too large for enclosure dimensions")

    elif name == "bracket":
        base = params.get("base_length", 50.0) or 50.0
        side = params.get("side_height", 40.0) or 40.0
        thick = params.get("thickness", 3.0) or 3.0
        if thick >= min(base, side):
            errors.append(f"Bracket thickness ({thick}mm) exceeds its dimensions")

    elif name == "nema_mount":
        size = params.get("nema_size", 17) or 17
        if size not in (14, 17, 23, 34):
            errors.append(f"Unsupported NEMA size: {size}")

    return errors

TEMPLATE_MAP = {
    "bolt": make_bolt,
    "enclosure": make_enclosure,
    "bracket": make_bracket,
    "plate_with_holes": make_plate_with_holes,
    "nema_mount": make_nema_motor_mount,
    "standoff": make_standoff,
}
