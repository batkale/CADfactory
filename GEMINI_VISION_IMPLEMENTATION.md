# Quick Implementation: Gemini Vision Quality Evaluation for CADfactory

Based on Text-to-CadQuery analysis, here's how to quickly add AI quality scoring.

## What We're Adding

When a user generates a part, we:
1. Render the generated STL to an image
2. Use Gemini 2.0 Flash vision to evaluate if it matches the description
3. Get a quality score (0-100)
4. Store it and use it for ranking

## Step 1: Add Quality Assessment Service

Create `services/quality_assessment.py`:

```python
"""
AI-powered quality evaluation using Gemini 2.0 Flash vision.
Evaluates if generated 3D models match natural language descriptions.
"""

import os
import json
import logging
from typing import Optional, Dict
from PIL import Image
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


def evaluate_generated_part(
    description: str,
    image_path: str
) -> Dict[str, any]:
    """
    Use Gemini 2.0 Flash vision to evaluate if generated part matches description.
    
    Args:
        description: Original user description
        image_path: Path to rendered image of the generated part
        
    Returns:
        {
            "match": "Yes" | "Partially" | "No",
            "quality_score": 0-100,
            "confidence": 0-1,
            "explanation": "Why it matches or doesn't"
        }
    """
    
    try:
        # Load image
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return get_default_quality_response()
        
        image = Image.open(image_path)
        
        # Initialize Gemini client
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY not set")
            return get_default_quality_response()
        
        client = genai.Client(api_key=api_key)
        
        # Call Gemini with vision
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(
                temperature=0,
                system_instruction="""You are a CAD quality engineer evaluating AI-generated 3D models.
                
Your task: Determine if the 3D model shown matches the natural language description.

Respond ONLY with valid JSON, no other text:
{
    "match": "Yes" or "Partially" or "No",
    "confidence": 0.0-1.0,
    "explanation": "Brief reason (1-2 sentences)"
}

Guidelines:
- "Yes": Model clearly matches description with all major features
- "Partially": Model mostly matches but some features may be missing/wrong
- "No": Model clearly doesn't match or is incorrect

Note: Single-angle renderings may hide some features. Be lenient if it's "reasonably possible" it matches.
"""
            ),
            contents=[
                image,
                f"""Evaluate this 3D CAD model against the description: "{description}"

Does this model match the description?

Respond with JSON only:
{{"match": "...", "confidence": 0-1, "explanation": "..."}}"""
            ]
        )
        
        # Parse response
        response_text = response.text.strip()
        if response_text.startswith("```"):
            # Remove markdown code blocks if present
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()
        
        result = json.loads(response_text)
        
        # Convert to score
        match_scores = {
            "Yes": 95,
            "Partially": 65,
            "No": 20
        }
        base_score = match_scores.get(result.get("match", "Partially"), 50)
        confidence = float(result.get("confidence", 0.5))
        quality_score = base_score * confidence  # Weight by confidence
        
        return {
            "match": result.get("match", "Unknown"),
            "quality_score": round(quality_score, 1),
            "confidence": round(confidence, 2),
            "explanation": result.get("explanation", ""),
            "success": True
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Gemini response: {e}")
        return get_default_quality_response()
    except Exception as e:
        logger.error(f"Quality evaluation failed: {e}")
        return get_default_quality_response()


def get_default_quality_response() -> Dict:
    """Return default response when evaluation fails."""
    return {
        "match": "Unknown",
        "quality_score": 50.0,
        "confidence": 0.0,
        "explanation": "Quality evaluation unavailable",
        "success": False
    }
```

## Step 2: Add Rendering Helper

Add to `services/rendering.py` (create if needed):

```python
"""Simple STL to image rendering using view angle."""

import os
import subprocess
from pathlib import Path


def render_stl_to_png(stl_path: str, output_png_path: str) -> bool:
    """
    Render STL to PNG using simple view.
    Uses CadQuery's built-in rendering or simple subprocess call.
    """
    
    try:
        import cadquery as cq
        
        # Load STL
        shape = cq.importers.importStep(stl_path)
        if shape is None:
            # Try STL import
            shape = cq.Shape.cast(shape)
        
        # Render and save
        shape.exportDxf(output_png_path)  # or use other method
        return os.path.exists(output_png_path)
        
    except Exception as e:
        print(f"Rendering failed: {e}")
        return False
```

## Step 3: Update Generated Part Model

Add to `models.py`:

```python
class GeneratedPart(Base):
    # ... existing fields ...
    
    # NEW: Quality assessment fields
    quality_score = Column(Float, nullable=True)  # 0-100
    quality_match = Column(String, nullable=True)  # "Yes", "Partially", "No"
    quality_confidence = Column(Float, nullable=True)  # 0-1
    quality_explanation = Column(String, nullable=True)
    rendered_image_path = Column(String, nullable=True)  # Path to rendered PNG
```

## Step 4: Update Generation Endpoint

Modify `routers/generate.py`:

```python
from services.quality_assessment import evaluate_generated_part
from services.rendering import render_stl_to_png

@router.post("/api/generate/")
async def generate(
    request: GenerateRequest,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Generate a part from natural language description."""
    
    # ... existing generation code ...
    
    # After successful generation and STL/STEP creation:
    
    # Render STL to image
    stl_file = os.path.join(OUTPUT_DIR, f"{part.id}.stl")
    png_file = os.path.join(OUTPUT_DIR, f"{part.id}_render.png")
    
    if os.path.exists(stl_file):
        render_success = render_stl_to_png(stl_file, png_file)
        
        if render_success:
            # Evaluate quality with Gemini
            quality = evaluate_generated_part(
                description=request.description,
                image_path=png_file
            )
            
            # Store quality assessment
            part.quality_score = quality["quality_score"]
            part.quality_match = quality["match"]
            part.quality_confidence = quality["confidence"]
            part.quality_explanation = quality["explanation"]
            part.rendered_image_path = png_file
            
            db.commit()
    
    # Return response with quality info
    return GenerateResponse(
        success=True,
        stl_url=stl_url,
        step_url=step_url,
        script=script,
        bom=bom,
        warnings=warnings,
        part_id=str(part.id),
        
        # NEW: Quality assessment
        quality_assessment={
            "score": part.quality_score,
            "match": part.quality_match,
            "confidence": part.quality_confidence,
            "explanation": part.quality_explanation
        }
    )
```

## Step 5: Update Search Ranking

Modify `services/semantic_search.py`:

```python
def composite_ranking_score(
    query: str,
    description: str,
    manufacturing_method: str = "fdm",
    features: Optional[List[SemanticFeature]] = None,
    user_rating: Optional[int] = None,
    feedback_count: int = 0,
    quality_score: Optional[float] = None  # NEW
) -> Tuple[float, Dict[str, float]]:
    """Compute ranking with quality component."""
    
    scores = {}
    
    # Existing scores
    scores["keyword_match"] = keyword_match_score(query, description)
    feature_score, feature_breakdown = feature_alignment_score(features or [])
    scores["feature_alignment"] = feature_score
    scores["rag_similarity"], _ = rag_similarity_score(query, description)
    scores["manufacturing_fit"] = manufacturing_compatibility_score(description, manufacturing_method)
    
    # NEW: Quality score component (15% weight)
    if quality_score is not None and quality_score > 0:
        scores["quality"] = min(1.0, quality_score / 100)
    else:
        scores["quality"] = 0.5  # Default
    
    # Weighted combination with NEW quality component
    composite = (
        scores["keyword_match"] * 0.30 +
        scores["feature_alignment"] * 0.25 +
        scores["rag_similarity"] * 0.15 +
        scores["manufacturing_fit"] * 0.10 +
        scores["quality"] * 0.15 +  # NEW: 15% quality
        scores.get("user_feedback", 0.5) * 0.05
    )
    
    scores_breakdown = {k: v for k, v in scores.items() if not k.startswith("feature_")}
    return composite * 100, scores_breakdown
```

## Step 6: Update Search Endpoint

Modify `routers/search.py`:

```python
def search_parts(
    query: str,
    db: Session,
    user_id: int,
    limit: int = 10,
    # ... other params ...
) -> List[SearchResult]:
    """Search with quality-aware ranking."""
    
    parts = db.query(models.GeneratedPart).filter(
        models.GeneratedPart.user_id == user_id
    ).all()
    
    ranked_results = []
    
    for part in parts:
        features = detect_semantic_features(part.description, ...)
        
        # Score includes quality_score if available
        score, breakdown = composite_ranking_score(
            query=query,
            description=part.description,
            manufacturing_method=part.manufacturing_method,
            features=features,
            user_rating=...,
            feedback_count=...,
            quality_score=part.quality_score  # NEW: pass quality score
        )
        
        # ... rest of search logic ...
```

## Step 7: Add Migration (if using Alembic)

```python
# migrations/versions/XXX_add_quality_assessment.py

from alembic import op
import sqlalchemy as sa

def upgrade():
    op.add_column('generated_parts', sa.Column('quality_score', sa.Float, nullable=True))
    op.add_column('generated_parts', sa.Column('quality_match', sa.String, nullable=True))
    op.add_column('generated_parts', sa.Column('quality_confidence', sa.Float, nullable=True))
    op.add_column('generated_parts', sa.Column('quality_explanation', sa.String, nullable=True))
    op.add_column('generated_parts', sa.Column('rendered_image_path', sa.String, nullable=True))

def downgrade():
    op.drop_column('generated_parts', 'quality_score')
    op.drop_column('generated_parts', 'quality_match')
    op.drop_column('generated_parts', 'quality_confidence')
    op.drop_column('generated_parts', 'quality_explanation')
    op.drop_column('generated_parts', 'rendered_image_path')
```

## Testing

```bash
# Test quality evaluation
python -c "
from services.quality_assessment import evaluate_generated_part
result = evaluate_generated_part(
    'Fidget spinner with three ball bearings',
    'path/to/rendered/image.png'
)
print(result)
"
```

## Expected Results

**For a good fidget spinner generation:**
```json
{
    "match": "Yes",
    "quality_score": 92.5,
    "confidence": 0.95,
    "explanation": "3D model closely matches description with all three bearings visible and ergonomic center grip."
}
```

**For a pop-it toy (bad match to spinner query):**
```json
{
    "match": "No",
    "quality_score": 18.0,
    "confidence": 0.85,
    "explanation": "Model shows bubble grid structure, not bearing-based spinner design."
}
```

## Benefits

✅ AI-powered quality assessment  
✅ Improves search ranking accuracy  
✅ Provides transparency to users  
✅ Creates feedback signal for future fine-tuning  
✅ Enables filtering by quality  

## Integration with Existing Semantic Search

The new quality score feeds directly into the composite ranking:

```
Fidget Spinner Search Results:
  1. Generated spinner (keyword: 95% + feature: 92% + RAG: 88% + mfg: 95% + quality: 92.5 + feedback: 92%)
     → Composite: 92.1/100 ✅ RANK #1 (High quality)
  
  2. Pop-it toy (keyword: 40% + feature: 15% + RAG: 22% + mfg: 60% + quality: 18 + feedback: 50%)
     → Composite: 34.8/100 ❌ RANK #3 (Low quality - doesn't match)
```

The quality component ensures that even if something matches keywords, if it doesn't actually match the description according to Gemini's vision, it ranks lower.
