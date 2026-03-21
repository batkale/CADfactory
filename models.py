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
    generated_parts = relationship("GeneratedPart", back_populates="owner", cascade="all, delete-orphan")
    feedbacks = relationship("GenerationFeedback", back_populates="owner", cascade="all, delete-orphan")


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


class GeneratedPart(Base):
    __tablename__ = "generated_parts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    description = Column(Text, nullable=False)
    manufacturing_method = Column(String, default="fdm")
    script = Column(Text, nullable=False)
    stl_path = Column(String, nullable=True)
    step_path = Column(String, nullable=True)
    bom_suggestion = Column(JSON, default=list)
    warnings = Column(JSON, default=list)
    attempts = Column(Integer, default=1)
    generation_time_s = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    owner = relationship("User", back_populates="generated_parts")
    feedbacks = relationship("GenerationFeedback", back_populates="part", cascade="all, delete-orphan")


class GenerationFeedback(Base):
    __tablename__ = "generation_feedback"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    part_id = Column(Integer, ForeignKey("generated_parts.id"), nullable=False)
    # +1 = thumbs up (good output), -1 = thumbs down (bad output)
    rating = Column(Integer, nullable=False)
    comment = Column(Text, nullable=True)
    # Snapshot of what was sent/received so we can train on it later
    prompt_snapshot = Column(Text, nullable=True)   # the description used
    script_snapshot = Column(Text, nullable=True)   # the generated script
    manufacturing_method = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    owner = relationship("User")
    part = relationship("GeneratedPart", back_populates="feedbacks")


class TunedModel(Base):
    """Tracks Gemini supervised fine-tuning jobs and their resulting tuned models."""
    __tablename__ = "tuned_models"

    id = Column(Integer, primary_key=True, index=True)
    display_name = Column(String, nullable=False)
    job_name = Column(String, nullable=False, unique=True)       # Gemini job ID / resource name
    tuned_model_name = Column(String, nullable=True)             # Set after job succeeds
    base_model = Column(String, default="models/gemini-1.5-flash-001-tuning")
    state = Column(String, default="PENDING")                    # mirrors Gemini JobState
    training_examples = Column(Integer, default=0)
    is_active = Column(Boolean, default=False)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)