"""
Object Knowledge Database — Standard geometric specifications for common objects.

Inspired by GenCAD's CAD sequence database approach, this maps object categories
to their expected geometric features, required holes, and standard dimensions.

This serves as a ground-truth reference that doesn't require AI calls,
making the decomposer more accurate for known object types.

Location: cadfactory-backend/services/object_knowledge.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class HoleSpec:
    """Specification for a required hole/bore."""
    purpose: str
    count: int
    diameter_mm: float
    depth_mm: Optional[float] = None  # None = through-hole
    position_hint: str = ""
    clearance_fit: bool = False  # Add 0.1mm clearance


@dataclass
class CavitySpec:
    """Specification for a required cavity/hollow."""
    purpose: str
    shape: str  # "cylinder", "box", "conformal"
    wall_thickness_mm: float = 2.5


@dataclass
class ObjectKnowledge:
    """Complete geometric knowledge about an object type."""
    name: str
    aliases: List[str]  # alternative names that match this object
    description: str
    symmetry: Optional[str] = None
    holes: List[HoleSpec] = field(default_factory=list)
    cavities: List[CavitySpec] = field(default_factory=list)
    critical_features: List[str] = field(default_factory=list)
    dimensions: Dict[str, float] = field(default_factory=dict)
    n_sub_components: int = 1  # how many distinct additive parts
    aesthetic_class: str = "mechanical"


# ── Knowledge Database ────────────────────────────────────────────────────────

OBJECT_DB: Dict[str, ObjectKnowledge] = {
    "fidget_spinner": ObjectKnowledge(
        name="Fidget Spinner",
        aliases=["fidget spinner", "hand spinner", "finger spinner", "tri spinner"],
        description="3-arm bar spinner with 608 bearings",
        symmetry="3-fold rotational",
        holes=[
            HoleSpec("center bearing bore", 1, 22.1, 8.0, "center", True),
            HoleSpec("arm bearing bore", 3, 22.1, 8.0, "at each arm tip", True),
        ],
        critical_features=[
            "central hub cylinder (r=16mm, h=8mm)",
            "3 radial arms overlapping hub by 2mm",
            "1 center 608 bearing bore (r=11.05mm through-hole)",
            "3 arm 608 bearing bores (r=11.05mm through-hole at arm tips)",
        ],
        dimensions={"overall_diameter": 76, "thickness": 8, "hub_radius": 16, "arm_radius": 14, "arm_distance": 28},
        n_sub_components=2,  # hub + arms
        aesthetic_class="consumer",
    ),

    "extension_cable": ObjectKnowledge(
        name="Extension Cable / Power Strip",
        aliases=["extension cable", "power strip", "extension cord", "power bar",
                 "extension lead", "surge protector", "multi plug", "multi socket"],
        description="Rectangular housing with socket openings for plugs",
        symmetry="bilateral",
        holes=[
            HoleSpec("cable entry hole", 1, 12.0, None, "one end"),
        ],
        cavities=[
            CavitySpec("main internal cavity", "box", 2.5),
        ],
        critical_features=[
            "rectangular body (rounded_box)",
            "N socket openings on top face (subtract box cuts)",
            "cable entry hole at one end (subtract cylinder)",
            "hollow interior for wiring (subtract box cavity)",
        ],
        dimensions={"length": 280, "width": 55, "height": 45, "socket_spacing": 55},
        aesthetic_class="consumer",
    ),

    "soap_dispenser": ObjectKnowledge(
        name="Soap Dispenser",
        aliases=["soap dispenser", "hand soap dispenser", "liquid soap dispenser",
                 "pump dispenser", "lotion dispenser"],
        description="Bottle with pump mechanism on top, nozzle for dispensing",
        symmetry="bilateral",
        holes=[
            HoleSpec("nozzle opening", 1, 6.0, None, "at spout tip"),
            HoleSpec("pump tube channel", 1, 8.0, None, "through neck into body"),
        ],
        cavities=[
            CavitySpec("body cavity for liquid", "cylinder", 2.5),
        ],
        critical_features=[
            "cylindrical/tapered body (main container)",
            "hollow body cavity for soap (subtract)",
            "neck cylinder on top",
            "pump hub on neck",
            "spout/nozzle extending from pump",
            "plunger button on top",
            "pump tube channel through body (subtract)",
        ],
        dimensions={"body_height": 130, "body_radius": 35, "neck_radius": 14, "total_height": 185},
        n_sub_components=5,
        aesthetic_class="consumer",
    ),

    "gear": ObjectKnowledge(
        name="Spur Gear",
        aliases=["gear", "spur gear", "cog", "cogwheel", "gear wheel"],
        description="Toothed wheel for power transmission",
        symmetry="radial",
        holes=[
            HoleSpec("center axle bore", 1, 8.0, None, "center", True),
            HoleSpec("keyway slot", 1, 3.0, None, "center bore wall"),
        ],
        critical_features=[
            "gear teeth around circumference (involute profile)",
            "center axle bore (through-hole)",
            "hub with bore",
        ],
        dimensions={"pitch_diameter": 40, "thickness": 10, "bore_diameter": 8},
        aesthetic_class="mechanical",
    ),

    "bracket": ObjectKnowledge(
        name="Mounting Bracket",
        aliases=["bracket", "l bracket", "l-bracket", "mounting bracket", "angle bracket",
                 "wall bracket", "shelf bracket"],
        description="L-shaped or flat bracket with mounting holes",
        symmetry="bilateral",
        holes=[
            HoleSpec("mounting hole", 4, 4.5, None, "at corners of each face"),
        ],
        critical_features=[
            "L-shaped or flat body",
            "mounting holes through each face",
            "reinforcement rib or gusset",
        ],
        dimensions={"base_length": 60, "side_height": 40, "width": 30, "thickness": 3},
        aesthetic_class="mechanical",
    ),

    "enclosure": ObjectKnowledge(
        name="Electronic Enclosure",
        aliases=["enclosure", "project box", "electronics enclosure", "case",
                 "housing", "electronics case", "control box"],
        description="Rectangular box with lid, cable openings, and mounting posts",
        symmetry="bilateral",
        holes=[
            HoleSpec("cable opening", 2, 10.0, None, "side walls"),
            HoleSpec("screw bosses for lid", 4, 3.4, None, "top corners"),
            HoleSpec("PCB mounting holes", 4, 3.0, None, "bottom face"),
        ],
        cavities=[
            CavitySpec("main internal cavity", "box", 2.5),
        ],
        critical_features=[
            "rectangular box body",
            "hollow interior cavity (subtract)",
            "lid mounting screw bosses",
            "cable/connector openings on sides (subtract)",
            "PCB standoffs on bottom",
        ],
        dimensions={"length": 120, "width": 80, "height": 40, "wall_thickness": 2.5},
        aesthetic_class="mechanical",
    ),

    "phone_case": ObjectKnowledge(
        name="Phone Case",
        aliases=["phone case", "phone cover", "phone shell", "iphone case",
                 "mobile case", "smartphone case"],
        description="Protective shell that wraps around a phone",
        symmetry="bilateral",
        holes=[
            HoleSpec("camera cutout", 1, 30.0, None, "back upper area"),
            HoleSpec("charging port cutout", 1, 12.0, None, "bottom center"),
            HoleSpec("speaker holes", 2, 3.0, None, "bottom"),
            HoleSpec("button cutouts", 3, 4.0, None, "sides"),
        ],
        cavities=[
            CavitySpec("phone cavity", "box", 1.5),
        ],
        critical_features=[
            "outer shell body",
            "phone-shaped cavity (subtract)",
            "camera cutout opening (subtract)",
            "charging port cutout (subtract)",
            "button access cutouts on sides (subtract)",
        ],
        dimensions={"length": 155, "width": 78, "height": 12, "wall_thickness": 1.5},
        aesthetic_class="consumer",
    ),

    "wheel": ObjectKnowledge(
        name="Wheel / Pulley",
        aliases=["wheel", "pulley", "roller", "idler wheel", "drive wheel"],
        description="Circular disc with center bore for axle",
        symmetry="radial",
        holes=[
            HoleSpec("center axle bore", 1, 8.0, None, "center", True),
        ],
        critical_features=[
            "disc body",
            "center axle bore (through-hole)",
            "optional tire groove on circumference",
        ],
        dimensions={"diameter": 50, "width": 15, "bore_diameter": 8},
        aesthetic_class="mechanical",
    ),

    "bottle_opener": ObjectKnowledge(
        name="Bottle Opener",
        aliases=["bottle opener", "beer opener", "cap opener"],
        description="Hand-held tool with a hook/lever for prying bottle caps",
        symmetry="bilateral",
        holes=[
            HoleSpec("keyring hole", 1, 8.0, None, "handle end"),
            HoleSpec("cap hook opening", 1, 26.0, None, "business end"),
        ],
        critical_features=[
            "handle body",
            "cap-catching hook opening (subtract)",
            "leverage lip/edge for prying",
            "optional keyring hole",
        ],
        dimensions={"length": 100, "width": 35, "thickness": 4},
        aesthetic_class="consumer",
    ),
}


def find_object_knowledge(description: str) -> Optional[ObjectKnowledge]:
    """
    Try to match a user description to a known object type.
    Returns the ObjectKnowledge if found, None otherwise.
    """
    desc_lower = description.lower().strip()

    # Direct alias matching
    for key, obj in OBJECT_DB.items():
        for alias in obj.aliases:
            if alias in desc_lower:
                return obj

    return None


def get_knowledge_injection(description: str) -> str:
    """
    Get a text injection for the decomposer system prompt based on object knowledge.
    Returns empty string if no match found.
    """
    obj = find_object_knowledge(description)
    if obj is None:
        return ""

    holes_text = "\n".join(
        f"  - {h.count}x {h.purpose}: diameter={h.diameter_mm}mm"
        f"{f' depth={h.depth_mm}mm' if h.depth_mm else ' (through-hole)'}"
        f" at {h.position_hint}"
        for h in obj.holes
    )

    cavities_text = "\n".join(
        f"  - {c.purpose}: {c.shape} (wall={c.wall_thickness_mm}mm)"
        for c in obj.cavities
    )

    features_text = "\n".join(f"  - {f}" for f in obj.critical_features)

    dims_text = ", ".join(f"{k}={v}mm" for k, v in obj.dimensions.items())

    return f"""
═══════════════════════════════════════════════════════════════════════════════
KNOWN OBJECT: {obj.name}
═══════════════════════════════════════════════════════════════════════════════
{obj.description}
Symmetry: {obj.symmetry or 'none'}
Aesthetic class: {obj.aesthetic_class}
Standard dimensions: {dims_text}

MANDATORY HOLES (op="subtract"):
{holes_text or '  None'}

MANDATORY CAVITIES (op="subtract"):
{cavities_text or '  None'}

CRITICAL FEATURES (MUST ALL be present in your CSG tree):
{features_text}

Number of sub-components: {obj.n_sub_components}
"""
