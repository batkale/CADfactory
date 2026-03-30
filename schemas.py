from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, EmailStr, field_validator

# ── Auth ──────────────────────────────────────────────────────

class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str

    @field_validator("username")
    @classmethod
    def username_alphanumeric(cls, v: str) -> str:
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Username must be alphanumeric (underscores/hyphens allowed)")
        if len(v) < 3 or len(v) > 30:
            raise ValueError("Username must be 3–30 characters")
        return v

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if len(v.encode("utf-8")) > 72:
            raise ValueError("Password must be 72 bytes or fewer (bcrypt limit)")
        return v


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: int
    email: str
    username: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserOut


# ── Files ─────────────────────────────────────────────────────

class FileOut(BaseModel):
    id: int
    filename: str
    file_format: str
    file_size_bytes: Optional[int]
    uploaded_at: datetime

    model_config = {"from_attributes": True}


# ── Geometry ──────────────────────────────────────────────────

class BoundingBox(BaseModel):
    x: float
    y: float
    z: float


class GeometryResult(BaseModel):
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
    components: Optional[List[str]]


# ── COGS ──────────────────────────────────────────────────────

class TierBreakdown(BaseModel):
    material: float
    machining: float
    labor: float
    logistics: float
    setup: float
    total: float
    low: float
    high: float


class RegionCOGS(BaseModel):
    name: str
    flag: str
    lead: str
    risk: str
    tiers: Dict[str, TierBreakdown]


class COGSResult(BaseModel):
    regions: Dict[str, RegionCOGS]
    material_grade: str
    process: str
    confidence_interval: float
    notes: str


# ── AI Analysis ──────────────────────────────────────────────

class PhysicsResult(BaseModel):
    score: int
    warning: str
    status: str


class EconomicsResult(BaseModel):
    retail: float
    health: str
    gross_margin_pct: Optional[float]


class CapitalResult(BaseModel):
    prototype: float
    capex: float
    total_ask: float


class SupplyChainResult(BaseModel):
    lead_time: str
    route: str
    risk: str
    suppliers: List[str]


class AIAnalysis(BaseModel):
    physics: PhysicsResult
    economics: EconomicsResult
    capital: CapitalResult
    supply_chain: SupplyChainResult
    optimizations: List[str]
    ai_used: bool


# ── Full Report ───────────────────────────────────────────────

class ReportOut(BaseModel):
    id: int
    file_id: int
    file_format: str
    format_detail: str
    volume_cm3: float
    surface_area_cm2: float
    volume_method: str
    confidence_interval: float
    triangle_count: Optional[int]
    face_count: Optional[int]
    bbox_x_mm: float
    bbox_y_mm: float
    bbox_z_mm: float
    complexity_score: float
    is_assembly: bool
    components: Optional[List[str]]
    cogs_data: Dict[str, Any]
    bom_items: Optional[List[Dict[str, Any]]]
    physics_score: Optional[int]
    physics_warning: Optional[str]
    retail_price_usd: Optional[float]
    capital_prototype: Optional[float]
    capital_capex: Optional[float]
    capital_total_ask: Optional[float]
    supply_lead_time: Optional[str]
    supply_route: Optional[str]
    supply_risk: Optional[str]
    suppliers: Optional[List[str]]
    optimizations: Optional[List[str]]
    ai_used: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class ReportSummary(BaseModel):
    id: int
    file_id: int
    file_format: str
    volume_cm3: float
    complexity_score: float
    capital_total_ask: Optional[float]
    physics_score: Optional[int]
    ai_used: bool
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Analysis request ──────────────────────────────────────────

class AnalysisRequest(BaseModel):
    file_id: int
    step_file_id: Optional[int] = None   # Optional STEP file for BOM
    material: str = "Al6061-T6"
    process: str = "CNC_3axis"
    use_ai: bool = True
