"""
Semantic Search & Result Ranking Engine for CADfactory

Combines:
1. Vector embedding similarity (ChromaDB/RAG)
2. Semantic feature matching (object characteristics)
3. Pattern recognition (visual features)
4. Domain-specific scoring (manufacturing suitability)
5. User feedback history (crowdsourced preference signals)

When searching for "fidget spinner", this ranks actual fidget spinners
(with bearings/rotational features) above pop-it toys or other objects.
"""

import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy.orm import Session

import models
from services.rag_store import get_similar_examples

logger = logging.getLogger(__name__)


@dataclass
class SemanticFeature:
    """Identifies characteristic features of an object"""
    name: str  # e.g., "rotational_bearing", "pop_bubble", "ergonomic_grip"
    confidence: float  # 0-1: how confident this feature is present
    count: Optional[int] = None  # e.g., number of bearings, number of bubbles


@dataclass
class SearchResult:
    """Ranked search result with score breakdown"""
    part_id: str
    description: str
    score: float  # 0-100, higher is better
    ranking_breakdown: Dict[str, float]  # Why it ranked here
    semantic_features: List[SemanticFeature]
    match_confidence: float  # 0-1


# ============================================================================
# SEMANTIC FEATURE DETECTION
# ============================================================================

# Object-specific feature signatures
FEATURE_SIGNATURES = {
    "fidget_spinner": {
        "required_features": [
            "rotational_component",
            "bearing",
            "symmetry",
            "hand_holdable"
        ],
        "forbidden_features": [
            "bubble_grid",
            "pop_texture"
        ],
        "keywords": [
            "spinner",
            "bearing",
            "rotat",
            "fidget",
            "gyro",
            "three arm",
            "tri-arm"
        ],
        "anti_keywords": [
            "pop",
            "bubble",
            "squeeze",
            "sensory",
            "dimple"
        ]
    },
    "pop_it": {
        "required_features": [
            "bubble_grid",
            "pop_texture",
            "silicone_material",
            "deformable"
        ],
        "forbidden_features": [
            "bearing",
            "rotational_component",
            "metallic_insert"
        ],
        "keywords": [
            "pop",
            "bubble",
            "fidget",
            "squeeze",
            "dimple",
            "sensory"
        ],
        "anti_keywords": [
            "bearing",
            "rotat",
            "metal",
            "steel"
        ]
    }
}


def detect_semantic_features(description: str, bom: Optional[List[Dict]] = None) -> List[SemanticFeature]:
    """
    Detect semantic features from part description and BOM.

    Args:
        description: Natural language description of the part
        bom: Optional Bill of Materials with component names

    Returns:
        List of detected features with confidence scores
    """
    features = []
    desc_lower = description.lower()

    # Check for rotational/bearing features
    bearing_indicators = ["bearing", "rotat", "spin", "gyro", "axle", "shaft", "center hole"]
    bearing_confidence = sum(1 for ind in bearing_indicators if ind in desc_lower) / len(bearing_indicators)
    if bearing_confidence > 0.2:
        features.append(SemanticFeature(
            name="rotational_bearing",
            confidence=min(0.95, bearing_confidence * 1.5)
        ))

    # Check for pop/bubble features
    bubble_indicators = ["pop", "bubble", "dimple", "deformable", "silicone", "squeeze"]
    bubble_confidence = sum(1 for ind in bubble_indicators if ind in desc_lower) / len(bubble_indicators)
    if bubble_confidence > 0.2:
        features.append(SemanticFeature(
            name="pop_bubble",
            confidence=min(0.95, bubble_confidence * 1.5)
        ))

    # Check symmetry (spinners are usually 3-arm or symmetrical)
    symmetry_indicators = ["three", "symmetric", "balanced", "arm", "lobe", "petal"]
    symmetry_confidence = sum(1 for ind in symmetry_indicators if ind in desc_lower) / len(symmetry_indicators)
    if symmetry_confidence > 0.15:
        features.append(SemanticFeature(
            name="symmetry",
            confidence=min(0.85, symmetry_confidence * 1.3),
            count=3 if "three" in desc_lower else (4 if "four" in desc_lower else None)
        ))

    # Check for hand-hold ergonomics
    ergonomic_indicators = ["grip", "hand", "ergonomic", "palm", "finger", "hold"]
    ergonomic_confidence = sum(1 for ind in ergonomic_indicators if ind in desc_lower) / len(ergonomic_indicators)
    if ergonomic_confidence > 0.15:
        features.append(SemanticFeature(
            name="ergonomic_grip",
            confidence=min(0.85, ergonomic_confidence * 1.3)
        ))

    # Check BOM for metallic/steel components (typical in spinners)
    if bom:
        bom_text = " ".join([str(item) for item in bom]).lower()
        metallic_confidence = sum(1 for term in ["bearing", "steel", "metal", "ball", "raceway"] if term in bom_text) / 5
        if metallic_confidence > 0.2:
            features.append(SemanticFeature(
                name="metallic_insert",
                confidence=min(0.9, metallic_confidence * 1.4)
            ))

    return features


def keyword_match_score(query: str, target: str, mode: str = "fidget_spinner") -> float:
    """
    Score keyword matches between query and target.

    Args:
        query: Search query ("fidget spinner")
        target: Target description/title
        mode: Object type to match against

    Returns:
        Score from 0-1 indicating keyword alignment
    """
    if mode not in FEATURE_SIGNATURES:
        return 0.5

    sig = FEATURE_SIGNATURES[mode]
    target_lower = target.lower()
    query_lower = query.lower()

    # Base score: does target contain the original query?
    base_match = 1.0 if query_lower in target_lower else 0.3

    # Keyword bonus
    keyword_matches = sum(1 for kw in sig["keywords"] if kw in target_lower)
    keyword_score = min(1.0, base_match * 0.7 + (keyword_matches / len(sig["keywords"])) * 0.3)

    # Anti-keyword penalty
    anti_keyword_matches = sum(1 for kw in sig["anti_keywords"] if kw in target_lower)
    anti_keyword_penalty = max(0.0, 1.0 - (anti_keyword_matches * 0.2))

    return keyword_score * anti_keyword_penalty


def feature_alignment_score(features: List[SemanticFeature], mode: str = "fidget_spinner") -> Tuple[float, Dict]:
    """
    Score how well detected features align with object type signature.

    Returns:
        (score 0-1, breakdown dict)
    """
    if mode not in FEATURE_SIGNATURES:
        return 0.5, {}

    sig = FEATURE_SIGNATURES[mode]
    breakdown = {}

    # Required features score
    required_found = [f for f in features if f.name in sig["required_features"]]
    required_score = len(required_found) / len(sig["required_features"]) if sig["required_features"] else 0.5
    breakdown["required_features"] = required_score

    # Forbidden features penalty
    forbidden_found = [f for f in features if f.name in sig["forbidden_features"]]
    forbidden_penalty = 1.0 - (len(forbidden_found) * 0.3)
    breakdown["forbidden_features"] = forbidden_penalty

    # Feature confidence average
    if features:
        avg_confidence = np.mean([f.confidence for f in features])
        breakdown["feature_confidence"] = avg_confidence
    else:
        avg_confidence = 0.3
        breakdown["feature_confidence"] = avg_confidence

    # Weighted combination
    score = (required_score * 0.5) + (forbidden_penalty * 0.3) + (avg_confidence * 0.2)

    return score, breakdown


def rag_similarity_score(query: str, description: str, method: str = "fdm") -> Tuple[float, Optional[float]]:
    """
    Use RAG store to get similarity score to known good examples.

    Returns:
        (rag_score 0-1, raw_distance from ChromaDB)
    """
    try:
        examples = get_similar_examples(query, method, k=1)
        if not examples:
            return 0.5, None

        # Convert ChromaDB distance to similarity (lower distance = higher similarity)
        distance = examples[0].get("distance", 1.5)

        # Exponential decay: distance 0 = 1.0, distance 0.5 = 0.6, distance 1.2+ = 0.1
        similarity = max(0.1, 2.0 ** (-distance * 2))

        return similarity, distance
    except Exception as e:
        logger.warning(f"RAG similarity lookup failed: {e}")
        return 0.5, None


def manufacturing_compatibility_score(description: str, manufacturing_method: str = "fdm") -> float:
    """
    Score how suitable this design is for manufacturing.
    Fidget spinners need precision bearings, making SLA/CNC more suitable than FDM.
    """
    method_scores = {
        "fdm": 0.6,  # Can work but less precise
        "sla": 0.9,  # Good precision for bearing races
        "sls": 0.85,  # Good precision, less post-processing
        "cnc": 0.95,  # Best precision for metallic parts
        "sheet_metal": 0.5,  # Not typical for spinners
        "injection": 0.85  # Good for high-volume production
    }

    base_score = method_scores.get(manufacturing_method, 0.5)

    # Check description for precision indicators
    precision_indicators = ["bearing", "precise", "tight tolerance", "race", "steel"]
    has_precision_needs = any(ind in description.lower() for ind in precision_indicators)

    if has_precision_needs and manufacturing_method == "fdm":
        return base_score * 0.7  # Penalize if precision-critical but FDM

    return base_score


def composite_ranking_score(
    query: str,
    description: str,
    manufacturing_method: str = "fdm",
    features: Optional[List[SemanticFeature]] = None,
    user_rating: Optional[int] = None,
    feedback_count: int = 0
) -> Tuple[float, Dict[str, float]]:
    """
    Compute composite ranking score (0-100) combining all signals.

    This is the main scoring function that determines result ranking.
    """

    if features is None:
        features = detect_semantic_features(description)

    # Component scores
    scores = {}

    # 1. Keyword matching (35% weight)
    scores["keyword_match"] = keyword_match_score(query, description, mode="fidget_spinner")

    # 2. Feature alignment (30% weight)
    feature_score, feature_breakdown = feature_alignment_score(features, mode="fidget_spinner")
    scores["feature_alignment"] = feature_score
    scores.update({f"feature_{k}": v for k, v in feature_breakdown.items()})

    # 3. RAG similarity (20% weight)
    rag_score, _ = rag_similarity_score(query, description, manufacturing_method)
    scores["rag_similarity"] = rag_score

    # 4. Manufacturing suitability (10% weight)
    scores["manufacturing_fit"] = manufacturing_compatibility_score(description, manufacturing_method)

    # 5. User feedback signal (5% weight, optional)
    if user_rating is not None and feedback_count > 0:
        feedback_score = (1 + user_rating) / 2  # Convert ±1 to 0-1
        scores["user_feedback"] = feedback_score
        user_signal = 0.05
    else:
        user_signal = 0.0

    # Weighted combination
    composite = (
        scores["keyword_match"] * 0.35 +
        scores["feature_alignment"] * 0.30 +
        scores["rag_similarity"] * 0.20 +
        scores["manufacturing_fit"] * 0.10 +
        scores.get("user_feedback", 0.5) * user_signal
    )

    # Normalize to 0-100
    final_score = composite * 100

    return final_score, scores


# ============================================================================
# SEARCH ENDPOINT
# ============================================================================

def search_parts(
    query: str,
    db: Session,
    user_id: int,
    limit: int = 10,
    manufacturing_method: Optional[str] = None,
    complexity_range: Optional[Tuple[int, int]] = None,
) -> List[SearchResult]:
    """
    Search generated parts by query, returning ranked results.

    Args:
        query: Search query (e.g., "fidget spinner")
        db: Database session
        user_id: User ID for filtering
        limit: Max results to return
        manufacturing_method: Filter by manufacturing method
        complexity_range: Filter by complexity score (min, max)

    Returns:
        List of SearchResult sorted by score (highest first)
    """

    # Fetch all user's generated parts
    parts_query = db.query(models.GeneratedPart).filter(
        models.GeneratedPart.user_id == user_id
    )

    if manufacturing_method:
        parts_query = parts_query.filter(
            models.GeneratedPart.manufacturing_method == manufacturing_method
        )

    parts = parts_query.all()

    # Score and rank each part
    ranked_results = []

    for part in parts:
        # Detect features from description and BOM
        bom = None
        if part.bom_suggestion:
            try:
                bom = json.loads(part.bom_suggestion) if isinstance(part.bom_suggestion, str) else part.bom_suggestion
            except Exception:
                pass

        features = detect_semantic_features(part.description, bom)

        # Get user feedback if exists
        feedback = db.query(models.GenerationFeedback).filter(
            models.GenerationFeedback.part_id == part.id
        ).all()

        avg_rating = None
        if feedback:
            avg_rating = np.mean([f.rating for f in feedback])

        # Compute ranking score
        score, breakdown = composite_ranking_score(
            query=query,
            description=part.description,
            manufacturing_method=part.manufacturing_method,
            features=features,
            user_rating=avg_rating,
            feedback_count=len(feedback)
        )

        # Apply complexity filter if provided
        if complexity_range:
            min_complexity, max_complexity = complexity_range
            # Estimate complexity from description (rough heuristic)
            desc_lower = part.description.lower()
            estimated_complexity = 5  # default
            if any(term in desc_lower for term in ["complex", "intricate", "detailed"]):
                estimated_complexity = 7
            elif any(term in desc_lower for term in ["simple", "basic", "minimal"]):
                estimated_complexity = 3

            if not (min_complexity <= estimated_complexity <= max_complexity):
                continue

        result = SearchResult(
            part_id=str(part.id),
            description=part.description,
            score=score,
            ranking_breakdown=breakdown,
            semantic_features=features,
            match_confidence=breakdown.get("keyword_match", 0.5)
        )

        ranked_results.append(result)

    # Sort by score descending
    ranked_results.sort(key=lambda r: r.score, reverse=True)

    return ranked_results[:limit]


def get_semantic_explanation(result: SearchResult) -> str:
    """
    Generate human-readable explanation of why a result ranked as it did.
    Useful for debugging and understanding the ranking.
    """
    explanation = f"Score: {result.score:.1f}/100\n"
    explanation += f"Match Confidence: {result.match_confidence:.1%}\n\n"

    explanation += "Scoring Breakdown:\n"
    for key, value in result.ranking_breakdown.items():
        if not key.startswith("feature_"):
            pct = f"{value:.1%}" if value <= 1 else f"{value:.1f}"
            explanation += f"  • {key.replace('_', ' ').title()}: {pct}\n"

    if result.semantic_features:
        explanation += "\nDetected Features:\n"
        for feat in result.semantic_features:
            conf_pct = f"{feat.confidence:.1%}"
            count_str = f" (count: {feat.count})" if feat.count else ""
            explanation += f"  • {feat.name}: {conf_pct}{count_str}\n"

    return explanation
