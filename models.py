from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    files = relationship("UploadedFile", back_populates="owner", cascade="all, delete-orphan")
    reports = relationship("Report", back_populates="owner", cascade="all, delete-orphan")


class UploadedFile(Base):
    __tablename__ = "uploaded_files"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    filename = Column(String, nullable=False)           # original filename
    stored_filename = Column(String, nullable=False)    # UUID filename on disk
    file_format = Column(String, nullable=False)        # STL | STEP
    file_size_bytes = Column(Integer)
    upload_path = Column(String, nullable=False)
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())

    owner = relationship("User", back_populates="files")
    reports = relationship("Report", back_populates="file", cascade="all, delete-orphan")


class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    file_id = Column(Integer, ForeignKey("uploaded_files.id"), nullable=False)

    # ── Geometry ─────────────────────────────────────────────
    file_format = Column(String)
    format_detail = Column(String)
    volume_cm3 = Column(Float)
    surface_area_cm2 = Column(Float)
    volume_method = Column(String)
    confidence_interval = Column(Float)
    triangle_count = Column(Integer, nullable=True)
    face_count = Column(Integer, nullable=True)
    bbox_x_mm = Column(Float)
    bbox_y_mm = Column(Float)
    bbox_z_mm = Column(Float)
    complexity_score = Column(Float)
    is_assembly = Column(Boolean, default=False)
    components = Column(JSON, nullable=True)          # list of component names
    bom_items = Column(JSON, nullable=True)            # rich BOM from STEP

    # ── COGS ─────────────────────────────────────────────────
    cogs_data = Column(JSON)                          # full region/tier breakdown

    # ── AI Analysis ──────────────────────────────────────────
    physics_score = Column(Integer, nullable=True)
    physics_warning = Column(Text, nullable=True)
    retail_price_usd = Column(Float, nullable=True)
    capital_prototype = Column(Float, nullable=True)
    capital_capex = Column(Float, nullable=True)
    capital_total_ask = Column(Float, nullable=True)
    supply_lead_time = Column(String, nullable=True)
    supply_route = Column(String, nullable=True)
    supply_risk = Column(String, nullable=True)
    suppliers = Column(JSON, nullable=True)
    optimizations = Column(JSON, nullable=True)
    ai_used = Column(Boolean, default=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    owner = relationship("User", back_populates="reports")
    file = relationship("UploadedFile", back_populates="reports")