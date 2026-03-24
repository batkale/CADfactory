"""
FastAPI router for intelligent semantic search with result ranking.

Endpoints:
    GET  /api/search/parts          — Search generated parts by query
    GET  /api/search/explain/{id}   — Get ranking explanation for a result
"""

import logging
from typing import Optional, List, Tuple
from fastapi import APIRouter, Depends, Query, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

import security as auth_utils
get_current_user = auth_utils.get_current_user

from database import get_db
from services.semantic_search import (
    search_parts,
    SearchResult,
    get_semantic_explanation,
    detect_semantic_features,
    composite_ranking_score
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/search", tags=["search"])


# ============================================================================
# Request/Response Models
# ============================================================================

class SearchResultResponse(BaseModel):
    """Single search result response"""
    part_id: str
    description: str
    score: float = Field(..., description="Ranking score 0-100")
    match_confidence: float = Field(..., description="Query match confidence 0-1")
    features: List[str] = Field(..., description="Detected semantic features")
    
    class Config:
        from_attributes = True


class SearchResponse(BaseModel):
    """Search endpoint response"""
    query: str
    total_results: int
    results: List[SearchResultResponse]
    explanation: Optional[str] = Field(
        None,
        description="Optional explanation of ranking methodology"
    )


class ExplainResponse(BaseModel):
    """Explanation response for a specific result"""
    part_id: str
    description: str
    score: float
    explanation: str
    ranking_breakdown: dict
    features_detail: List[dict]


# ============================================================================
# Endpoints
# ============================================================================

@router.get(
    "/parts",
    response_model=SearchResponse,
    summary="Search generated parts by query",
    description="Search user's generated parts with intelligent ranking using semantic features, keyword matching, and user feedback."
)
async def search_parts_endpoint(
    query: str = Query(
        ...,
        min_length=3,
        max_length=200,
        description="Search query (e.g., 'fidget spinner', 'servo bracket')"
    ),
    limit: int = Query(
        10,
        ge=1,
        le=100,
        description="Max results to return"
    ),
    manufacturing_method: Optional[str] = Query(
        None,
        pattern="^(fdm|sla|sls|cnc|sheet_metal|injection)$",
        description="Filter by manufacturing method"
    ),
    min_complexity: Optional[int] = Query(
        None,
        ge=0,
        le=10,
        description="Minimum complexity score (0-10)"
    ),
    max_complexity: Optional[int] = Query(
        None,
        ge=0,
        le=10,
        description="Maximum complexity score (0-10)"
    ),
    explain: bool = Query(
        False,
        description="Include ranking methodology explanation"
    ),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Search generated parts with intelligent ranking.
    
    The ranking system considers:
    - Keyword matching (does result match search query?)
    - Semantic features (detected rotational bearings, pop bubbles, etc.)
    - RAG similarity (vector embeddings to known good examples)
    - Manufacturing suitability (is design appropriate for the method?)
    - User feedback (have users rated similar parts highly?)
    
    Example: Query "fidget spinner" will rank actual spinners 
    (with bearings) above pop-it toys or other toys.
    """
    
    # Build complexity range filter
    complexity_range = None
    if min_complexity is not None or max_complexity is not None:
        complexity_range = (
            min_complexity or 0,
            max_complexity or 10
        )
    
    # Search
    try:
        results = search_parts(
            query=query,
            db=db,
            user_id=current_user.id,
            limit=limit,
            manufacturing_method=manufacturing_method,
            complexity_range=complexity_range
        )
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")
    
    # Format response
    response_results = [
        SearchResultResponse(
            part_id=r.part_id,
            description=r.description,
            score=r.score,
            match_confidence=r.match_confidence,
            features=[f.name for f in r.semantic_features]
        )
        for r in results
    ]
    
    explanation = None
    if explain:
        explanation = (
            "Ranking Methodology:\n"
            "- Keyword Matching (35%): Does result contain/match the query terms?\n"
            "- Feature Alignment (30%): Does result have expected semantic features?\n"
            "- RAG Similarity (20%): How similar to known good examples?\n"
            "- Manufacturing Fit (10%): Is design suitable for chosen method?\n"
            "- User Feedback (5%): Have users rated similar parts well?\n\n"
            "Example: 'fidget spinner' prioritizes results with:\n"
            "✓ Rotational bearings, symmetry, hand-gripability\n"
            "✗ Avoids pop bubbles, silicone deformability"
        )
    
    return SearchResponse(
        query=query,
        total_results=len(results),
        results=response_results,
        explanation=explanation
    )


@router.get(
    "/explain/{part_id}",
    response_model=ExplainResponse,
    summary="Get ranking explanation for a search result",
    description="Understand why a specific part was ranked the way it was."
)
async def explain_ranking(
    part_id: str,
    query: str = Query(
        ...,
        min_length=3,
        description="Original search query"
    ),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get detailed explanation of why a part received its ranking score.
    Useful for understanding ranking decisions and debugging.
    """
    
    import models
    
    # Fetch part
    part = db.query(models.GeneratedPart).filter(
        models.GeneratedPart.id == int(part_id),
        models.GeneratedPart.user_id == current_user.id
    ).first()
    
    if not part:
        raise HTTPException(status_code=404, detail="Part not found")
    
    # Detect features
    import json
    bom = None
    if part.bom_suggestion:
        try:
            bom = json.loads(part.bom_suggestion) if isinstance(part.bom_suggestion, str) else part.bom_suggestion
        except:
            pass
    
    features = detect_semantic_features(part.description, bom)
    
    # Get user feedback
    feedback = db.query(models.GenerationFeedback).filter(
        models.GenerationFeedback.part_id == part.id
    ).all()
    
    avg_rating = None
    if feedback:
        import numpy as np
        avg_rating = np.mean([f.rating for f in feedback])
    
    # Compute score
    score, breakdown = composite_ranking_score(
        query=query,
        description=part.description,
        manufacturing_method=part.manufacturing_method,
        features=features,
        user_rating=avg_rating,
        feedback_count=len(feedback)
    )
    
    # Format response
    features_detail = [
        {
            "name": f.name,
            "confidence": f.confidence,
            "count": f.count
        }
        for f in features
    ]
    
    explanation = get_semantic_explanation(SearchResult(
        part_id=str(part.id),
        description=part.description,
        score=score,
        ranking_breakdown=breakdown,
        semantic_features=features,
        match_confidence=breakdown.get("keyword_match", 0.5)
    ))
    
    return ExplainResponse(
        part_id=str(part.id),
        description=part.description,
        score=score,
        explanation=explanation,
        ranking_breakdown=breakdown,
        features_detail=features_detail
    )


@router.get(
    "/debug/features",
    summary="Debug: Extract features from a description",
    description="Internal debugging endpoint to see how a description is parsed for features."
)
async def debug_extract_features(
    description: str = Query(..., description="Part description to analyze"),
    current_user = Depends(get_current_user)
):
    """Debug endpoint: see what semantic features are extracted from a description."""
    
    features = detect_semantic_features(description)
    
    return {
        "description": description,
        "features": [
            {
                "name": f.name,
                "confidence": f.confidence,
                "count": f.count
            }
            for f in features
        ]
    }
