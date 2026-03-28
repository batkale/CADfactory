import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
Geometry parsing service.
- STL binary & ASCII → exact volume via divergence theorem
- STEP → bounding box × fill factor, B-Rep face count, rich assembly BOM
"""

import math  # noqa: E402
import re  # noqa: E402
import struct  # noqa: E402
from collections import Counter  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from typing import Optional  # noqa: E402


@dataclass
class BoundingBox:
    """Bounding box with __slots__ for reduced memory on hot paths."""
    __slots__ = ('x', 'y', 'z', 'min_x', 'min_y', 'min_z')
    x: float
    y: float
    z: float
    min_x: float
    min_y: float
    min_z: float

    def __init__(self, x: float, y: float, z: float,
                 min_x: float = 0.0, min_y: float = 0.0, min_z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z
        self.min_x = min_x
        self.min_y = min_y
        self.min_z = min_z


@dataclass
class BomItem:
    """A single BOM line item extracted from STEP."""
    __slots__ = ('idx', 'item', 'description', 'qty', 'unit_cost', 'total_cost',
                 'material', 'part_type', 'source', 'aliexpress_price',
                 'amazon_price', 'aliexpress_url', 'amazon_url')
    idx: int
    item: str
    description: str
    qty: int
    unit_cost: float
    total_cost: float
    material: str
    part_type: str      # 'machined' | 'fastener' | 'bearing' | 'motor' | 'gear' | 'electronic'
    source: str
    aliexpress_price: Optional[float]
    amazon_price: Optional[float]
    aliexpress_url: Optional[str]
    amazon_url: Optional[str]

    def __init__(self, idx: int, item: str, description: str, qty: int,
                 unit_cost: float, total_cost: float, material: str,
                 part_type: str, source: str,
                 aliexpress_price: Optional[float] = None,
                 amazon_price: Optional[float] = None,
                 aliexpress_url: Optional[str] = None,
                 amazon_url: Optional[str] = None):
        self.idx = idx
        self.item = item
        self.description = description
        self.qty = qty
        self.unit_cost = unit_cost
        self.total_cost = total_cost
        self.material = material
        self.part_type = part_type
        self.source = source
        self.aliexpress_price = aliexpress_price
        self.amazon_price = amazon_price
        self.aliexpress_url = aliexpress_url
        self.amazon_url = amazon_url


@dataclass
class GeometryResult:
    file_format: str
    format_detail: str
    volume_cm3: float
    surface_area_cm2: float
    volume_method: str
    confidence_interval: float
    triangle_count: Optional[int]
    face_count: Optional[int]
    bounding_box_mm: BoundingBox
    complexity_score: float
    is_assembly: bool
    components: Optional[list] = field(default=None)
    units: Optional[str] = None
    bom_items: Optional[list] = field(default=None)  # List[BomItem]


# ── Assembly BOM Cache ───────────────────────────────────────────────────────
# Caches parsed BOM data by file content hash to avoid re-parsing
# the same STEP assembly multiple times.
import hashlib  # noqa: E402

_bom_cache: dict[str, list] = {}


def _content_hash(data: bytes) -> str:
    """Fast hash of file content for cache key."""
    return hashlib.md5(data).hexdigest()


def get_cached_bom(data: bytes, volume_cm3: float, is_assembly: bool) -> list:
    """Return cached BOM or parse and cache it."""
    key = _content_hash(data)
    if key in _bom_cache:
        return _bom_cache[key]
    text = data.decode("utf-8", errors="replace")
    bom = _extract_step_bom(text, volume_cm3, is_assembly)
    _bom_cache[key] = bom
    return bom


def invalidate_bom_cache(data: Optional[bytes] = None) -> None:
    """Clear BOM cache, optionally for a specific file."""
    if data is not None:
        key = _content_hash(data)
        _bom_cache.pop(key, None)
    else:
        _bom_cache.clear()


def _r2(n: float) -> float:
    return round(n, 2)


# ── MARKET-BASED PRICE DATABASE (AliExpress / Misumi / KHK research 2024) ────
#
# Fasteners (AliExpress bulk, per piece):
#   M3 screw: $0.03 | M4: $0.05 | M6: $0.09
#   M3 nut: $0.01   | M4: $0.02 | M6: $0.03
# Bearings (AliExpress):
#   4mm bore: $0.45 | 6mm: $0.75 | 8mm: $1.20 | 10mm+: $1.80
# Gears (AliExpress / KHK):
#   Small plastic spur (<20T): $0.80 | metal spur 20T: $2.50 | rack (100mm): $3.50
# Motors: $8–45 depending on type
# Shafts: $1–8 depending on diameter/length

# Pattern → (part_type, unit_cost_usd, source)
# Cost=None means we calculate dynamically from name
STANDARD_PART_PATTERNS = [
    # Screws / bolts  — M-size parsed dynamically
    (r'\bM(\d+)\b.*\b(screw|bolt|cap|socket|hex)\b', 'fastener', None, 'AliExpress'),
    (r'\b(M\d+)\s*\d+\s*mm\b',                        'fastener', None, 'AliExpress'),
    (r'\bM(\d+)\s+\d+mm\b',                            'fastener', None, 'AliExpress'),
    (r'\b(screw|cap screw|hex bolt|hex screw)\b',      'fastener', 0.05, 'AliExpress'),
    # Nuts
    (r'\b(nut|hex nut|lock nut|nyloc|nylock)\b',       'fastener', 0.02, 'AliExpress'),
    # Washers
    (r'\b(washer|flat washer|spring washer)\b',        'fastener', 0.01, 'AliExpress'),
    # Bare M-size (e.g. "M4 Screw v1", "M6 Nut v1")
    (r'\bM(\d+)\b',                                    'fastener', None, 'AliExpress'),
    # Threaded rod
    (r'\b(threaded rod|threaded bar|allthread)\b',     'fastener', 1.20, 'AliExpress'),
    # Bearings — bore parsed dynamically
    (r'\b(flanged bearing|flange bearing)\b',          'bearing',  None, 'AliExpress'),
    (r'\bbearing\s+(\d+)\s*mm\b',                      'bearing',  None, 'AliExpress'),
    (r'\b(bearing|ball bearing|roller bearing)\b',     'bearing',  None, 'AliExpress'),
    # Motors & actuators
    (r'\b(dc motor|brushless|bldc|stepper motor)\b',   'motor',    12.0, 'AliExpress'),
    (r'\b(servo|servo motor)\b',                       'motor',    8.0,  'AliExpress'),
    (r'\b(motor)\b',                                   'motor',    15.0, 'AliExpress'),
    (r'\b(actuator|linear actuator|solenoid)\b',       'motor',    18.0, 'Misumi'),
    # Gears & drive
    (r'\b(spur gear|helical gear|bevel gear)\b',       'gear',     None, 'AliExpress/KHK'),
    (r'\b(rack)\b',                                    'gear',     3.50, 'AliExpress'),
    (r'\bgear\b',                                      'gear',     None, 'AliExpress/KHK'),
    (r'\b(sprocket|pulley|timing pulley)\b',           'gear',     2.50, 'AliExpress'),
    (r'\b(shaft|axle|spindle|driven shaft)\b',         'gear',     None, 'AliExpress'),
    # Electronics
    (r'\b(pcb|controller|sensor|encoder|arduino)\b',  'electronic', None, 'Mouser/AliExpress'),
    (r'\b(led|battery|cable|connector|wire)\b',        'electronic', 1.50, 'AliExpress'),
    # Springs & seals
    (r'\b(spring|compression spring|tension spring)\b','fastener', 0.35, 'AliExpress'),
    (r'\b(seal|o-ring|gasket|lip seal)\b',             'fastener', 0.25, 'AliExpress'),
    (r'\b(pin|dowel pin|roll pin|split pin)\b',        'fastener', 0.20, 'AliExpress'),
]

MATERIAL_PATTERNS = [
    (r'\b(al(umini?um)?|6061|7075|5052|2024)\b', 'Al 6061-T6', 2.70,  5.50),
    (r'\b(steel|stainless|ss304|ss316|inox)\b',  'SS 304',      7.90, 12.00),
    (r'\b(titanium|ti-6al-4v|grade 5)\b',        'Ti-6Al-4V',   4.43, 80.00),
    (r'\b(peek)\b',                               'PEEK',        1.32, 95.00),
    (r'\b(nylon|pa12|pa6|polyamide)\b',           'Nylon PA12',  1.01,  8.00),
    (r'\b(abs|pla|petg|resin)\b',                 'ABS/PLA',     1.05,  3.00),
    (r'\b(brass|bronze)\b',                       'Brass',       8.50, 10.00),
    (r'\b(copper|cu)\b',                          'Copper',      8.96, 10.00),
    (r'\b(carbon|cfrp|composite)\b',              'CFRP',        1.60, 120.00),
]


def _detect_material(name: str):
    """Returns (material_name, density, price_per_kg) or None."""
    name_lower = name.lower()
    for pattern, mat, density, price in MATERIAL_PATTERNS:
        if re.search(pattern, name_lower, re.IGNORECASE):
            return mat, density, price
    return None


def _clean_part_name(name: str) -> str:
    """
    Clean STEP product names:
    - Remove McMaster catalog numbers (e.g. '7804K142_', '91290A-')
    - Remove version suffixes (v1, v2, Updated v1)
    - Remove STEP encoding artifacts
    - Truncate overly long names intelligently
    """
    # Remove STEP encoding artifacts
    name = re.sub(r'\\X[02]\\[^\\]*\\X[02]\\', '', name).strip()

    # Remove McMaster/catalog part numbers at start: "7804K142_", "91290A142-"
    name = re.sub(r'^[A-Z0-9]{4,12}[_\-]\s*', '', name, flags=re.IGNORECASE)

    # Remove version suffixes
    name = re.sub(r'\s+v\s*\d+(\.\d+)?\s*$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s+(updated|rough\s*draw\w*|rough|final|rev\w*)\s*\d*$', '', name, flags=re.IGNORECASE)

    # Remove redundant material adjectives that clutter the name
    name = re.sub(r'corrosion[- ]resistant\s+', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\b(grade\s+\d+|class\s+\d+)\b', '', name, flags=re.IGNORECASE)

    name = re.sub(r'\s{2,}', ' ', name).strip()

    # If still too long, try to extract meaningful part type
    if len(name) > 40:
        # Bearing
        m = re.search(r'ball bearing|bearing', name, re.IGNORECASE)
        if m:
            d = re.search(r'(\d+)\s*mm', name)
            return f'Ball Bearing{" Ø"+d.group(1)+"mm" if d else ""}'
        # Screw
        m = re.search(r'socket head|cap screw|hex screw|screw|bolt', name, re.IGNORECASE)
        if m:
            msize = re.search(r'M(\d+)', name, re.IGNORECASE)
            length = re.search(r'(\d{2,3})\s*mm', name)
            return f'{"M"+msize.group(1) if msize else ""} {"x"+length.group(1)+"mm " if length else ""}Screw'.strip()
        # Nut
        if re.search(r'\bnut\b', name, re.IGNORECASE):
            msize = re.search(r'M(\d+)', name, re.IGNORECASE)
            return f'{"M"+msize.group(1)+" " if msize else ""}Hex Nut'.strip()
        # Washer
        if re.search(r'\bwasher\b', name, re.IGNORECASE):
            msize = re.search(r'M(\d+)', name, re.IGNORECASE)
            return f'{"M"+msize.group(1)+" " if msize else ""}Washer'.strip()
        # Gear
        if re.search(r'\bgear\b', name, re.IGNORECASE):
            t = re.search(r'(\d+)\s*teeth', name, re.IGNORECASE)
            return f'Spur Gear{" "+t.group(1)+"T" if t else ""}'
        # Motor
        if re.search(r'\bmotor\b', name, re.IGNORECASE):
            return 'DC Motor'
        # Fallback: truncate at word boundary
        name = name[:35].rsplit(' ', 1)[0] + '…'

    return name.strip()


    name_lower = name.lower()
    for pattern, mat, density, price in MATERIAL_PATTERNS:
        if re.search(pattern, name_lower, re.IGNORECASE):
            return mat, density, price
    return None


def _infer_fastener_cost(name: str) -> float:
    """AliExpress market-based fastener cost from M-size."""
    mnum = re.search(r'M(\d+)', name, re.IGNORECASE)
    msize = int(mnum.group(1)) if mnum else 4
    # is it a nut?
    if re.search(r'\bnut\b', name, re.IGNORECASE):
        return {2: 0.01, 3: 0.01, 4: 0.02, 5: 0.03, 6: 0.03, 8: 0.05}.get(msize, round(msize * 0.007, 3))
    # washer
    if re.search(r'\bwasher\b', name, re.IGNORECASE):
        return {2: 0.01, 3: 0.01, 4: 0.01, 6: 0.02, 8: 0.03}.get(msize, 0.02)
    # default: screw / bolt
    return {2: 0.02, 3: 0.03, 4: 0.05, 5: 0.07, 6: 0.09, 8: 0.15, 10: 0.22, 12: 0.35}.get(msize, round(msize * 0.03, 2))


def _infer_bearing_cost(name: str) -> float:
    """AliExpress market-based bearing cost from bore diameter."""
    dnum = re.search(r'(\d+)\s*mm', name)
    bore = int(dnum.group(1)) if dnum else 8
    # AliExpress flanged/deep groove pricing
    table = {4: 0.45, 5: 0.55, 6: 0.75, 7: 0.90, 8: 1.20, 10: 1.50, 12: 1.80, 15: 2.20, 20: 3.00}
    for b, price in sorted(table.items()):
        if bore <= b:
            return price
    return round(bore * 0.18, 2)


def _infer_gear_cost(name: str) -> float:
    """AliExpress / KHK market-based gear cost from teeth count."""
    teeth_m = re.search(r'(\d+)\s*teeth', name, re.IGNORECASE)
    teeth = int(teeth_m.group(1)) if teeth_m else 20
    # Small plastic gear (AliExpress): $0.80–3.50; metal KHK: $3–15
    if re.search(r'\b(rack)\b', name, re.IGNORECASE):
        return 3.50
    if teeth <= 12:
        return 1.20
    elif teeth <= 20:
        return 2.50
    elif teeth <= 40:
        return 4.50
    else:
        return 8.00


def _infer_shaft_cost(name: str) -> float:
    """AliExpress shaft cost from diameter hint."""
    dnum = re.search(r'(\d+)\s*mm', name)
    d = int(dnum.group(1)) if dnum else 6
    if d <= 4:
        return 0.80
    if d <= 6:
        return 1.20
    if d <= 8:
        return 1.80
    if d <= 10:
        return 2.50
    return round(d * 0.30, 2)


def _classify_part(name: str):
    """Returns (part_type, description, unit_cost, source)."""
    name_lower = name.lower()

    for pattern, ptype, cost, source in STANDARD_PART_PATTERNS:
        if re.search(pattern, name_lower, re.IGNORECASE):
            if cost is None:
                if ptype == 'fastener':
                    cost = _infer_fastener_cost(name)
                elif ptype == 'bearing':
                    cost = _infer_bearing_cost(name)
                elif ptype == 'gear':
                    if re.search(r'\b(shaft|axle|spindle|driven shaft)\b', name_lower):
                        cost = _infer_shaft_cost(name)
                    else:
                        cost = _infer_gear_cost(name)
                elif ptype == 'electronic':
                    cost = 8.0
                else:
                    cost = 2.0
            return ptype, name, cost, source

    return 'machined', name, None, 'CNC Supplier'


def _extract_step_bom(text: str, volume_cm3: float, is_assembly: bool) -> list:
    """Extract rich BOM from STEP PRODUCT entities. Returns list of BomItem."""
    prod_re = re.compile(
        r"PRODUCT\s*\(\s*'([^']*)'\s*,\s*'([^']*)'\s*,\s*'([^']*)'",
        re.IGNORECASE,
    )

    raw_products = []
    for m in prod_re.finditer(text):
        short = m.group(1).strip()
        long  = m.group(2).strip()
        # prefer longer/more descriptive name, but strip encoding garbage
        raw = long if (long and len(long) > len(short) and len(long) < 120) else short
        name = _clean_part_name(raw)
        if name and len(name) > 2 and not re.match(r'^[\s\d.]*$', name):
            raw_products.append(name)

    name_counts = Counter(raw_products)
    # Timestamp/date patterns from Fusion 360: "2026-02-08-11-30-05-289", "v1 2025-01-01" etc.
    TIMESTAMP_RE = re.compile(r'^\d{4}[-_]\d{2}[-_]\d{2}', re.IGNORECASE)
    # Generic/useless names to skip
    SKIP = {
        '', ' ', 'part', 'assembly', 'component', 'body', 'solid', 'shape',
        'default', 'object', 'untitled', 'new part', 'new component',
        'dfm', 'dfm review', 'nre', 'service',
    }

    items = []
    for n, c in name_counts.items():
        n_clean = n.lower().strip()
        # Skip timestamps
        if TIMESTAMP_RE.match(n.strip()):
            continue
        # Skip generic names
        if n_clean in SKIP:
            continue
        # Skip very short or purely numeric
        if len(n) < 3 or re.match(r'^[\s\d.\-_/]+$', n):
            continue
        items.append((n, c))

    if not items:
        return []

    bom = []
    idx = 1
    DEFAULT_MAT, DEFAULT_DENSITY, DEFAULT_PRICE = 'Al 6061-T6', 2.70, 5.50

    for name, qty in sorted(items, key=lambda x: (-x[1], x[0])):
        ptype, desc, hint_cost, source = _classify_part(name)
        mat_result = _detect_material(name)

        if ptype == 'machined':
            mat_name, density, price_kg = mat_result if mat_result else (DEFAULT_MAT, DEFAULT_DENSITY, DEFAULT_PRICE)
            name_lower = name.lower()
            # Name-based CNC cost tiers (prototype, nearshore)
            if any(k in name_lower for k in ['chassis', 'frame', 'housing', 'body', 'base', 'plate', 'bracket']):
                unit_cost = 85.0
            elif any(k in name_lower for k in ['mechanism', 'drive', 'grip', 'locker', 'mirror']):
                unit_cost = 55.0
            elif any(k in name_lower for k in ['gear', 'axle', 'pulley', 'sprocket']):
                unit_cost = 45.0
            elif any(k in name_lower for k in ['holder', 'mount', 'spacer', 'clamp', 'clip', 'flange']):
                unit_cost = 18.0
            elif any(k in name_lower for k in ['lead', 'diaphragm', 'diaphram', 'cover', 'cap']):
                unit_cost = 22.0
            elif any(k in name_lower for k in ['shaft', 'rod', 'axle', 'driven']):
                unit_cost = 28.0
            else:
                unit_cost = 35.0
            mat_factor = price_kg / 5.50
            unit_cost = _r2(unit_cost * mat_factor)
            description = f'{mat_name} CNC Machined'
            source = 'CNC Supplier'
        else:
            unit_cost = _r2(hint_cost) if hint_cost else 5.0
            mat_name = mat_result[0] if mat_result else '—'
            description = desc

        bom.append(BomItem(
            idx=idx, item=name, description=description,
            qty=qty, unit_cost=unit_cost, total_cost=_r2(unit_cost * qty),
            material=mat_name, part_type=ptype, source=source,
        ))
        idx += 1

    # NOTE: Do NOT add anodize or DFM review here — those are service costs
    # added separately in the Capital section, not in the BOM parts list.
    return bom


# ── STL ───────────────────────────────────────────────────────

def _is_binary_stl(data: bytes) -> bool:
    if len(data) < 84:
        return False
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    expected_size = 84 + triangle_count * 50
    return len(data) == expected_size and len(data) > 84


def _parse_stl_binary(data: bytes) -> tuple:
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    triangles = []
    for i in range(triangle_count):
        offset = 84 + i * 50
        floats = struct.unpack_from("<9f", data, offset + 12)
        triangles.append((floats[0:3], floats[3:6], floats[6:9]))
    return triangles, "Binary STL"


def _parse_stl_ascii(data: bytes) -> tuple:
    text = data.decode("utf-8", errors="replace")
    matches = re.findall(
        r"vertex\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)", text
    )
    vertices = [(float(m[0]), float(m[1]), float(m[2])) for m in matches]
    triangles = [
        (vertices[i], vertices[i + 1], vertices[i + 2])
        for i in range(0, len(vertices) - 2, 3)
    ]
    return triangles, "ASCII STL"


def _compute_stl_metrics(triangles: list) -> tuple:
    vol = 0.0
    sa = 0.0
    min_x = min_y = min_z = math.inf
    max_x = max_y = max_z = -math.inf

    for v1, v2, v3 in triangles:
        vol += (
            v1[0] * (v2[1] * v3[2] - v2[2] * v3[1])
            + v1[1] * (v2[2] * v3[0] - v2[0] * v3[2])
            + v1[2] * (v2[0] * v3[1] - v2[1] * v3[0])
        ) / 6.0
        ax, ay, az = v2[0]-v1[0], v2[1]-v1[1], v2[2]-v1[2]
        bx, by, bz = v3[0]-v1[0], v3[1]-v1[1], v3[2]-v1[2]
        sa += 0.5 * math.sqrt((ay*bz - az*by)**2 + (az*bx - ax*bz)**2 + (ax*by - ay*bx)**2)
        for v in (v1, v2, v3):
            min_x = min(min_x, v[0]); max_x = max(max_x, v[0])  # noqa: E702
            min_y = min(min_y, v[1]); max_y = max(max_y, v[1])  # noqa: E702
            min_z = min(min_z, v[2]); max_z = max(max_z, v[2])  # noqa: E702

    volume_cm3 = abs(vol) / 1000.0
    sa_cm2 = sa / 100.0
    bbox = BoundingBox(
        x=_r2(max_x - min_x), y=_r2(max_y - min_y), z=_r2(max_z - min_z),
        min_x=min_x, min_y=min_y, min_z=min_z,
    )
    sphere_sa = 4.836 * max(volume_cm3, 0.001) ** (2/3)
    complexity = round(min(max(sa_cm2 / sphere_sa, 1.0), 10.0), 1)
    return _r2(volume_cm3), _r2(sa_cm2), bbox, complexity


def parse_stl(data: bytes) -> GeometryResult:
    if _is_binary_stl(data):
        triangles, fmt_detail = _parse_stl_binary(data)
    else:
        triangles, fmt_detail = _parse_stl_ascii(data)
    if not triangles:
        raise ValueError("No triangles found in STL file")
    volume_cm3, sa_cm2, bbox, complexity = _compute_stl_metrics(triangles)
    return GeometryResult(
        file_format="STL",
        format_detail=fmt_detail,
        volume_cm3=volume_cm3,
        surface_area_cm2=sa_cm2,
        volume_method="Exact (divergence theorem)",
        confidence_interval=0.25,
        triangle_count=len(triangles),
        face_count=None,
        bounding_box_mm=bbox,
        complexity_score=complexity,
        is_assembly=False,
        components=None,
        bom_items=None,
    )


# ── STEP ──────────────────────────────────────────────────────

def parse_step(data: bytes) -> GeometryResult:
    text = data.decode("utf-8", errors="replace")

    # Units
    is_inch = bool(re.search(r"CONVERSION_BASED_UNIT[^)]*'INCH'", text, re.IGNORECASE)) \
              and not bool(re.search(r"LENGTH_UNIT[^)]*MILLIMETRE", text, re.IGNORECASE))
    to_mm = 25.4 if is_inch else 1.0

    # Bounding box from Cartesian points
    pt_pattern = re.compile(
        r"CARTESIAN_POINT\s*\(\s*'[^']*'\s*,\s*\(\s*"
        r"([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\)",
        re.IGNORECASE,
    )
    min_x = min_y = min_z = math.inf
    max_x = max_y = max_z = -math.inf
    pt_count = 0
    for m in pt_pattern.finditer(text):
        x, y, z = float(m.group(1))*to_mm, float(m.group(2))*to_mm, float(m.group(3))*to_mm
        min_x = min(min_x, x); max_x = max(max_x, x)  # noqa: E702
        min_y = min(min_y, y); max_y = max(max_y, y)  # noqa: E702
        min_z = min(min_z, z); max_z = max(max_z, z)  # noqa: E702
        pt_count += 1

    if pt_count == 0:
        raise ValueError("No geometry points found in STEP file")

    bbox = BoundingBox(
        x=_r2(max_x - min_x), y=_r2(max_y - min_y), z=_r2(max_z - min_z),
        min_x=min_x, min_y=min_y, min_z=min_z,
    )

    # B-Rep faces
    face_count = len(re.findall(r"ADVANCED_FACE\s*\(", text, re.IGNORECASE))

    # Simple component list (for backwards compat)
    prod_pattern = re.compile(
        r"PRODUCT\s*\(\s*'([^']*)'\s*,\s*'([^']*)'", re.IGNORECASE
    )
    products = set()
    for m in prod_pattern.finditer(text):
        name = (m.group(2) or m.group(1)).strip()
        if name and len(name) > 1 and not re.match(r'^[\s\d]*$', name):
            products.add(name)
    components = [n for n in products if len(n) > 1][:20]
    is_assembly = len(components) > 1

    # Volume
    bbox_vol_cm3 = (bbox.x * bbox.y * bbox.z) / 1000.0
    aspect_ratio = max(bbox.x, bbox.y, bbox.z) / (min(bbox.x, bbox.y, bbox.z) or 1)
    if is_assembly:
        fill_factor = 0.28
    elif face_count > 50:
        fill_factor = 0.38
    elif face_count > 20:
        fill_factor = 0.48
    else:
        fill_factor = 0.62
    if aspect_ratio > 5:
        fill_factor *= 0.75
    volume_cm3 = _r2(bbox_vol_cm3 * fill_factor)

    # SA
    bbox_sa_cm2 = 2 * (bbox.x*bbox.y + bbox.y*bbox.z + bbox.x*bbox.z) / 100.0
    sa_cm2 = _r2(bbox_sa_cm2 * min(1.0 + face_count/30.0, 4.0))

    complexity = round(min(max(face_count / 8.0, 1.0), 10.0), 1)

    # Rich BOM (with caching to avoid re-parsing the same assembly)
    bom_items = get_cached_bom(data, volume_cm3, is_assembly) if is_assembly or components else None

    return GeometryResult(
        file_format="STEP",
        format_detail=f"STEP Assembly ({len(components)} parts)" if is_assembly else "STEP Part",
        volume_cm3=volume_cm3,
        surface_area_cm2=sa_cm2,
        volume_method=f"BBox×fill({round(fill_factor*100)}%)",
        confidence_interval=0.35,
        triangle_count=None,
        face_count=face_count,
        bounding_box_mm=bbox,
        complexity_score=complexity,
        is_assembly=is_assembly,
        components=components if components else None,
        units="inch→mm" if is_inch else "mm",
        bom_items=bom_items,
    )


# ── 3MF ───────────────────────────────────────────────────────

def parse_3mf(data: bytes) -> GeometryResult:
    """
    Parse a 3MF file (ZIP-based XML).
    3MF contains: exact mesh triangles, units, materials, part names.
    """
    import xml.etree.ElementTree as ET
    import zipfile
    from io import BytesIO

    try:
        zf = zipfile.ZipFile(BytesIO(data))
    except zipfile.BadZipFile:
        raise ValueError("Not a valid 3MF file (bad ZIP)")

    # Find the 3D model file (usually 3D/3dmodel.model)
    model_file = None
    for name in zf.namelist():
        if name.endswith('.model'):
            model_file = name
            break
    if not model_file:
        raise ValueError("No .model file found inside 3MF")

    xml_data = zf.read(model_file)
    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError as e:
        raise ValueError(f"3MF XML parse error: {e}")

    # Namespace from root tag
    ns_uri = ''
    if root.tag.startswith('{'):
        ns_uri = root.tag.split('}')[0][1:]
    ns = {'m': ns_uri} if ns_uri else {}

    def find_all(element, tag):
        if ns_uri:
            return element.findall(f'm:{tag}', ns)
        return element.findall(tag)

    def find_one(element, tag):
        if ns_uri:
            return element.find(f'm:{tag}', ns)
        return element.find(tag)

    # Units — 3MF default is millimeters
    unit = root.get('unit', 'millimeter')
    to_mm = {'millimeter': 1.0, 'centimeter': 10.0, 'inch': 25.4,
             'foot': 304.8, 'meter': 1000.0, 'micron': 0.001}.get(unit, 1.0)

    # Collect triangles and part names from all mesh objects
    triangles = []
    part_names = []

    resources = find_one(root, 'resources')
    if resources is None:
        raise ValueError("No resources found in 3MF")

    for obj in find_all(resources, 'object'):
        obj_name = obj.get('name', f'Part {obj.get("id", "")}').strip()
        if obj_name:
            part_names.append(_clean_part_name(obj_name))

        mesh = find_one(obj, 'mesh')
        if mesh is None:
            continue

        vertices_el = find_one(mesh, 'vertices')
        triangles_el = find_one(mesh, 'triangles')
        if vertices_el is None or triangles_el is None:
            continue

        # Parse vertices
        verts = []
        for v in find_all(vertices_el, 'vertex'):
            verts.append((
                float(v.get('x', 0)) * to_mm,
                float(v.get('y', 0)) * to_mm,
                float(v.get('z', 0)) * to_mm,
            ))

        # Parse triangles
        for t in find_all(triangles_el, 'triangle'):
            v1 = int(t.get('v1', 0))
            v2 = int(t.get('v2', 0))
            v3 = int(t.get('v3', 0))
            if v1 < len(verts) and v2 < len(verts) and v3 < len(verts):
                triangles.append((verts[v1], verts[v2], verts[v3]))

    if not triangles:
        raise ValueError("No mesh triangles found in 3MF file")

    # Extract material names if available
    material_names = []
    for base in find_all(resources, 'basematerials') if ns_uri else resources.findall('basematerials'):
        for base_item in base:
            mat = base_item.get('name', '')
            if mat:
                material_names.append(mat)

    # Compute geometry (same as STL pipeline)
    volume_cm3, sa_cm2, bbox, complexity = _compute_stl_metrics(triangles)

    # Build simple BOM from part names if multiple objects
    is_assembly = len(part_names) > 1
    components = list(dict.fromkeys(part_names))[:30]  # deduplicated

    bom_items = None
    if is_assembly and components:
        # Build a lightweight BOM from names (same classifier as STEP)
        bom = []
        from collections import Counter
        name_counts = Counter(components)
        idx = 1
        for name, qty in sorted(name_counts.items(), key=lambda x: (-x[1], x[0])):
            ptype, desc, hint_cost, source = _classify_part(name)
            mat_result = _detect_material(name)
            if ptype == 'machined':
                mat_name = mat_result[0] if mat_result else 'Al 6061-T6'
                desc = f'{mat_name} CNC Machined'
                unit_cost = 35.0
            else:
                mat_name = mat_result[0] if mat_result else '—'
                unit_cost = hint_cost or 5.0
            bom.append(BomItem(
                idx=idx, item=name, description=desc, qty=qty,
                unit_cost=_r2(unit_cost), total_cost=_r2(unit_cost * qty),
                material=mat_name, part_type=ptype, source=source,
            ))
            idx += 1
        bom_items = bom

    mat_note = f" · {material_names[0]}" if material_names else ""
    parts_note = f" · {len(components)} parts" if is_assembly else ""

    return GeometryResult(
        file_format="3MF",
        format_detail=f"3MF (exact geometry{parts_note}{mat_note})",
        volume_cm3=volume_cm3,
        surface_area_cm2=sa_cm2,
        volume_method="Exact (divergence theorem)",
        confidence_interval=0.22,   # Slightly better than STL (has units/material)
        triangle_count=len(triangles),
        face_count=None,
        bounding_box_mm=bbox,
        complexity_score=complexity,
        is_assembly=is_assembly,
        components=components if is_assembly else None,
        units=unit,
        bom_items=bom_items,
    )


# ── Entry point ───────────────────────────────────────────────

def parse_cad_file(data: bytes, filename: str) -> GeometryResult:
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext in ("stl",):
        return parse_stl(data)
    elif ext == "3mf":
        return parse_3mf(data)
    elif ext in ("stp", "step"):
        return parse_step(data)
    else:
        raise ValueError(f"Unsupported format: .{ext} — supported: STL, 3MF, STEP")


# ── Geometry + STEP Merge ─────────────────────────────────────

def merge_stl_step(stl: GeometryResult, step: GeometryResult) -> GeometryResult:
    """
    Merge geometry file (STL or 3MF) with STEP for best combined result.
    - Volume, surface area, complexity → from geometry file (exact mesh)
    - BOM, components, face_count → from STEP (semantic)
    - Bounding box → geometry file (mesh-exact)
    """
    src_fmt = stl.file_format   # "STL" or "3MF"
    n_parts = len(step.components or [])
    return GeometryResult(
        file_format=f"{src_fmt}+STEP",
        format_detail=f"{src_fmt} (exact geometry) + STEP ({n_parts} parts BOM)",
        volume_cm3=stl.volume_cm3,
        surface_area_cm2=stl.surface_area_cm2,
        volume_method=stl.volume_method + " · STEP BOM merged",
        confidence_interval=0.18,   # Best: exact mesh + semantic BOM
        triangle_count=stl.triangle_count,
        face_count=step.face_count,
        bounding_box_mm=stl.bounding_box_mm,
        complexity_score=stl.complexity_score,
        is_assembly=step.is_assembly,
        components=step.components,
        units=stl.units,
        bom_items=step.bom_items,
    )
