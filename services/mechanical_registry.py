"""
Mechanical DNA Registry — General industrial design patterns and their constraints.
Used to inject dynamic engineering 'priors' into the AI generation layers.
"""

from typing import Dict, List

from pydantic import BaseModel


class MechanicalDNA(BaseModel):
    name: str
    description: str
    key_geometric_priors: List[str]
    math_rules: List[str]
    standard_lookups: Dict[str, str]
    example_csg: str

# ── Registry Definition ────────────────────────────────────────────────────────

MECHANICAL_REGISTRY: Dict[str, MechanicalDNA] = {
    "RADIAL_ASSEMBLY": MechanicalDNA(
        name="Radial Assembly",
        description="A rotating or symmetrical core with repeating functional arms/lobes.",
        key_geometric_priors=[
            "1x Central Hub (cylinder) - MUST hold the central axis/bearing.",
            "1x Functional Axis (cylinder, op=subtract) - ALWAYS required for rotation.",
            "Nx Radial Arms (cylinder or curve_prism) - MUST overlap hub by ≥2.0mm.",
            "Lobe Weights (op=subtract) - hollow pockets at the end of each arm."
        ],
        math_rules=[
            "Hub Radius: ≥ Bearing OD + 4.0mm structural wall.",
            "Step Angle: 360 / n_copies.",
            "Core Bore: Default to 22.1mm (r=11.05) if no axis specified."
        ],
        standard_lookups={
            "608_bearing": "OD=22mm, h=7mm, Bore_Pocket=22.1mm",
            "6201_bearing": "OD=32mm, h=10mm, Bore_Pocket=32.1mm"
        },
        example_csg=(
            "hub: cylinder r=16, h=7; "
            "bore: cylinder r=11.05, h=10, op=subtract; "
            "lobe: cylinder r=14, h=7, pos=[28,0,0], copies=3, angle=120; "
            "weight: cylinder r=11.05, h=10, pos=[28,0,0], op=subtract, copies=3, angle=120"
        )
    ),

    "ENCLOSURE": MechanicalDNA(
        name="Consumer Storage & Assembly",
        description="A hollow shell with integrated mechanical interfaces (pumps, nozzles, lids).",
        key_geometric_priors=[
            "1x Main Body (chamfered_box or rounded_cylinder) - The storage volume.",
            "1x Neck/Aperture (cylinder) - The primary mechanical interface.",
            "1x Pump/Hub Base (cylinder) - Sits on the neck, holds the mechanism.",
            "1x Plunger/Button (cylinder) - The user interaction point.",
            "1x Spout/Nozzle (curved_prism or cylinder) - The delivery path.",
            "Structural Overlap: ALL joined parts MUST overlap by ≥2.0mm."
        ],
        math_rules=[
            "Capacity Check: V_outer - V_inner (shell_t ≥ 2.5mm)",
            "Stack Height: Body_H + Neck_H + Pump_H + Button_H",
            "Clearance: Plunger radius = Pump radius - 1.5mm"
        ],
        standard_lookups={
            "soap_pump_neck": "OD=28mm, ID=18mm, h=25mm",
            "dispenser_nozzle": "OD=6mm, ID=3mm, curve_r=15mm"
        },
        example_csg=(
            "body: cylinder r=35, h=100; "
            "neck: cylinder r=14, h=25, pos=[0,0,60.5]; "  # 2mm overlap with body top at 50
            "pump: cylinder r=16, h=15, pos=[0,0,78.5]; "  # 2mm overlap with neck top at 50+25+overlap? No, calc carefully.
            "button: cylinder r=12, h=10, pos=[0,0,89.0]; "
            "spout: cylinder r=4, h=30, pos=[15,0,88.5], angle=[0,90,0]"
        )
    ),

    "STRUCTURAL_BRACKET": MechanicalDNA(
        name="Structural Bracket",
        description="A component designed to hold two planes or parts at a fixed angle.",
        key_geometric_priors=[
            "Ribbing: 45-degree gussets for L-bend reinforcement",
            "Mounting Holes: standard clearance diameters",
            "Fillets: large internal fillets on load-bearing joints"
        ],
        math_rules=[
            "Hole Clearance (ISO 273): M4=4.5mm, M5=5.5mm, M6=6.6mm",
            "Rib thickness ≈ 0.8 * wall_thickness"
        ],
        standard_lookups={
            "clevis_joint": "standard pin OD + 0.2mm clearance"
        },
        example_csg="base: box 50x30x5; wall: box 5x30x50, pos=[0,0,25]; hole: cylinder r=2.25, op=subtract"
    ),

    "FLUID_CONNECTOR": MechanicalDNA(
        name="Fluid/Pipe Connector",
        description="Tubular components for flow passage, usually with tapered or threaded ends.",
        key_geometric_priors=[
            "Lumen/Bore: central hollow passage",
            "Barbs or Flanges: for secure fitting",
            "Wall: minimum structural safety for pressure"
        ],
        math_rules=[
            "Tapered NPT: 1 in 16 slope (1.78 degrees)",
            "Flow area check: A = pi * r^2"
        ],
        standard_lookups={
            "1/2_inch_pipe": "OD=21.3mm, ID=15.8mm",
            "garden_hose": "ID=12.7mm or 19.0mm"
        },
        example_csg="pipe: cylinder r=10.65, h=50; bore: cylinder r=7.9, h=60, op=subtract"
    )
}

def get_dna_prompt_injection(dna_key: str) -> str:
    """Generate a text instruction block for a specific DNA type."""
    dna = MECHANICAL_REGISTRY.get(dna_key)
    if not dna:
        return ""

    instr = f"\nAPPLY {dna.name.upper()} REFERENCE STANDARDS:\n"
    instr += f"Description: {dna.description}\n"
    instr += "Rules to Enforce:\n"
    for r in dna.math_rules + dna.key_geometric_priors:
        instr += f"  • {r}\n"
    instr += "Standards:\n"
    for k, v in dna.standard_lookups.items():
        instr += f"  • {k}: {v}\n"
    instr += f"Pattern Example: {dna.example_csg}\n"
    return instr
