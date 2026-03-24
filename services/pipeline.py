"""
8-Layer Precision Pipeline Orchestrator

Chains all 8 layers in sequence and returns a unified PipelineResult
containing the final script, per-layer reports, and an overall accuracy score.

Layer execution order:
  1. NLP Contextual Extraction      (nlp_extractor.py)
  2. Parameterization & Sorting     (param_sorter.py)
  3. Mathematical Constraint Check  (constraint_validator.py)
  4. Generative Gap-Filling         (gap_filler.py)
  5. CSG Topology Planning          (csg_guard.py → plan_csg_topology)
  6. Edge-Case Guard                (csg_guard.py → guard_csg_plan)
  7. Granular Script Generation     (granular_builder.py)
  8. Script Linting & Overhaul      (script_refiner.py)

Also includes the bridge to the existing semantic decomposer (which runs between
Layers 4 and 5 to generate the CSG operations that Layers 5–7 work with).

Location: cadfactory-backend/services/pipeline.py
"""

from __future__ import annotations

import time
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ── Result model ───────────────────────────────────────────────────────────────

@dataclass
class LayerReport:
    layer:      int
    name:       str
    elapsed_s:  float = 0.0
    confidence: float = 1.0
    warnings:   List[str] = field(default_factory=list)
    notes:      str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "name": self.name,
            "elapsed_s": round(self.elapsed_s, 3),
            "confidence": round(self.confidence, 3),
            "warnings": self.warnings,
            "notes": self.notes,
        }


@dataclass
class PipelineResult:
    success:              bool
    final_script:         str = ""
    layer_reports:        List[LayerReport] = field(default_factory=list)
    overall_accuracy:     float = 0.0
    total_elapsed_s:      float = 0.0
    error:                Optional[str] = None

    # Extra metadata
    object_name:          str = ""
    decomposed_ops_count: int = 0
    script_line_count:    int = 0
    detail_multiplier:    float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "overall_accuracy": round(self.overall_accuracy, 3),
            "total_elapsed_s": round(self.total_elapsed_s, 2),
            "object_name": self.object_name,
            "decomposed_ops_count": self.decomposed_ops_count,
            "script_line_count": self.script_line_count,
            "detail_multiplier": round(self.detail_multiplier, 2),
            "layer_reports": [r.to_dict() for r in self.layer_reports],
            "error": self.error,
        }


# ── Pipeline ───────────────────────────────────────────────────────────────────

async def run_precision_pipeline(
    prompt:               str,
    manufacturing_method: str = "fdm",
    constraints:          Optional[Dict[str, Any]] = None,
    skip_layer8:          bool = False,
) -> PipelineResult:
    """
    Execute the full 8-layer precision pipeline.

    Returns a PipelineResult with the final script and all layer reports.
    If any layer fails critically, falls back to the existing 2-pass pipeline
    via generate_and_execute().
    """
    pipeline_start = time.time()
    reports: List[LayerReport] = []

    def _t(start: float) -> float:
        return round(time.time() - start, 3)

    # ── Step 0: RAG (fetch similar approved examples) ─────────────────────
    rag_examples = []
    try:
        from services.rag_store import get_similar_examples
        rag_examples = get_similar_examples(prompt, manufacturing_method, k=3)
        if rag_examples:
            logger.info(f"[Pipeline] RAG: fetched {len(rag_examples)} examples")
    except Exception as e:
        logger.warning(f"[Pipeline] RAG lookup failed (non-fatal): {e}")

    # ══════════════════════════════════════════════════════════════════════
    # LAYER 1 — NLP Contextual Extraction
    # ══════════════════════════════════════════════════════════════════════
    t = time.time()
    try:
        from services.nlp_extractor import extract_context
        context = extract_context(prompt)
        reports.append(LayerReport(
            layer=1, name="NLP Contextual Extraction",
            elapsed_s=_t(t),
            confidence=context.confidence,
            warnings=[context.clarification_question] if context.clarification_question else [],
            notes=f"category={context.object_category} intent={context.intent_type}",
        ))
        logger.info(f"[L1] category={context.object_category} conf={context.confidence:.2f}")
    except Exception as exc:
        logger.error(f"[L1] failed: {exc}", exc_info=True)
        return await _fallback(prompt, manufacturing_method, constraints, pipeline_start,
                         error=f"Layer 1 failed: {exc}")

    # ══════════════════════════════════════════════════════════════════════
    # LAYER 1.5 — Reference Matching (Template ID)
    # ══════════════════════════════════════════════════════════════════════
    t = time.time()
    try:
        from services.reference_matcher import match_reference
        match_res = match_reference(prompt, context.object_category)
        if match_res.template_key:
            from services.mechanical_registry import get_dna_prompt_injection
            reference_dna = get_dna_prompt_injection(match_res.template_key)
            logger.info(f"[L1.5] Matched template: {match_res.template_key} (conf={match_res.confidence:.2f})")
        else:
            reference_dna = ""
            logger.info("[L1.5] No specific mechanical template matched.")
            
        reports.append(LayerReport(
            layer=1.5, name="Reference Standard Matching",
            elapsed_s=_t(t),
            confidence=match_res.confidence,
            notes=f"template={match_res.template_key or 'None'} reasoning='{match_res.reasoning[:40]}...'",
        ))
    except Exception as exc:
        logger.error(f"[L1.5] failed: {exc}", exc_info=True)
        reference_dna = ""

    # ══════════════════════════════════════════════════════════════════════
    # LAYER 2 — Parameterization & Sorting
    # ══════════════════════════════════════════════════════════════════════
    t = time.time()
    try:
        from services.param_sorter import sort_parameters
        sorted_p = sort_parameters(context, prompt)
        n_params = len(sorted_p.explicit_dims) + len(sorted_p.implied_dims)
        reports.append(LayerReport(
            layer=2, name="Parameterization & Sorting",
            elapsed_s=_t(t),
            confidence=sorted_p.confidence,
            notes=f"explicit={len(sorted_p.explicit_dims)} implied={len(sorted_p.implied_dims)} "
                  f"functional={len(sorted_p.functional_params)}",
        ))
        logger.info(f"[L2] {n_params} params extracted, conf={sorted_p.confidence:.2f}")
    except Exception as exc:
        logger.error(f"[L2] failed: {exc}", exc_info=True)
        return await _fallback(prompt, manufacturing_method, constraints, pipeline_start,
                         error=f"Layer 2 failed: {exc}")

    # ══════════════════════════════════════════════════════════════════════
    # LAYER 3 — Mathematical Constraint Validation
    # ══════════════════════════════════════════════════════════════════════
    t = time.time()
    try:
        from services.constraint_validator import validate_parameter_bundle
        validated = validate_parameter_bundle(sorted_p, context.object_category)
        reports.append(LayerReport(
            layer=3, name="Mathematical Constraint Validation",
            elapsed_s=_t(t),
            confidence=validated.engineering_score,
            warnings=validated.warnings + validated.errors,
            notes=f"passed={validated.passed} corrections={len(validated.corrections)} "
                  f"score={validated.engineering_score:.2f}",
        ))
        if not validated.passed:
            logger.warning(f"[L3] validation errors: {validated.errors}")
        else:
            logger.info(f"[L3] validation PASSED, score={validated.engineering_score:.2f}")
    except Exception as exc:
        logger.error(f"[L3] failed: {exc}", exc_info=True)
        return await _fallback(prompt, manufacturing_method, constraints, pipeline_start,
                         error=f"Layer 3 failed: {exc}")

    # ══════════════════════════════════════════════════════════════════════
    # LAYER 4 — Generative Gap-Filling
    # ══════════════════════════════════════════════════════════════════════
    t = time.time()
    try:
        from services.gap_filler import fill_gaps
        spec = fill_gaps(context, sorted_p, validated)
        reports.append(LayerReport(
            layer=4, name="Generative Gap-Filling",
            elapsed_s=_t(t),
            confidence=spec.confidence,
            notes=f"wall={spec.wall_mm}mm draft={spec.draft_angle_deg}° "
                  f"material={spec.material_guess} tol={spec.tolerance_class} "
                  f"inferred={len(spec.inferred_fields)} fields",
        ))
        logger.info(
            f"[L4] gap-filled: wall={spec.wall_mm}mm draft={spec.draft_angle_deg}° "
            f"conf={spec.confidence:.2f}"
        )
    except Exception as exc:
        logger.error(f"[L4] failed: {exc}", exc_info=True)
        return await _fallback(prompt, manufacturing_method, constraints, pipeline_start,
                         error=f"Layer 4 failed: {exc}")

    # ══════════════════════════════════════════════════════════════════════
    # CSG DECOMPOSITION (semantic_decomposer — between L4 and L5)
    # Uses the refined prompt enriched by L1-L4 context
    # ══════════════════════════════════════════════════════════════════════
    t = time.time()
    try:
        from services.semantic_decomposer import decompose_prompt
        # Build an enriched prompt from the spec
        enriched_prompt = _build_enriched_prompt(prompt, spec, constraints)
        decomposed = decompose_prompt(
            enriched_prompt, 
            reference_dna=reference_dna,
            specific_guidance=match_res.specific_guidance if 'match_res' in locals() else "",
            rag_examples=rag_examples
        )
        ops = decomposed.operations

        # Merge spec into decomposed for downstream use
        spec.operations = ops
        spec.object_name = decomposed.object_name or spec.object_name
        if decomposed.symmetry:
            spec.symmetry = decomposed.symmetry

        logger.info(
            f"[Decomposer] '{decomposed.object_name}' — {len(ops)} ops, "
            f"conf={decomposed.confidence:.2f}"
        )
    except Exception as exc:
        logger.error(f"[Decomposer] failed: {exc}", exc_info=True)
        return await _fallback(prompt, manufacturing_method, constraints, pipeline_start,
                         error=f"CSG Decomposir failed: {exc}")

    # ══════════════════════════════════════════════════════════════════════
    # LAYER 5 + 6 — CSG Topology Planning + Edge-Case Guard
    # ══════════════════════════════════════════════════════════════════════
    t = time.time()
    try:
        from services.csg_guard import guard_csg_plan
        guarded = guard_csg_plan(
            ops=ops,
            priority_order=spec.priority_order,
            process=manufacturing_method,
        )
        n_flags = len(guarded.risk_flags)
        reports.append(LayerReport(
            layer=5, name="CSG Topology Planning",
            elapsed_s=_t(t) / 2,
            confidence=guarded.guard_confidence,
            warnings=[f.description for f in guarded.risk_flags if f.severity == "critical"],
            notes=f"ops={len(guarded.approved_ops)} phased+ordered | "
                  f"was_patched={guarded.was_patched}",
        ))
        reports.append(LayerReport(
            layer=6, name="Edge-Case Guard",
            elapsed_s=_t(t) / 2,
            confidence=guarded.guard_confidence,
            warnings=[f.description for f in guarded.risk_flags],
            notes=f"risk_flags={n_flags} critical={guarded.critical_risk_count} "
                  f"ai_notes={len(guarded.ai_notes)}",
        ))
        logger.info(
            f"[L5+6] guard: {n_flags} flags, {guarded.critical_risk_count} critical, "
            f"patched={guarded.was_patched}"
        )
    except Exception as exc:
        logger.error(f"[L5+6] failed: {exc}", exc_info=True)
        return await _fallback(prompt, manufacturing_method, constraints, pipeline_start,
                         error=f"Layers 5+6 failed: {exc}")

    # ══════════════════════════════════════════════════════════════════════
    # LAYER 7 — Granular Script Generation
    # ══════════════════════════════════════════════════════════════════════
    t = time.time()
    try:
        from services.granular_builder import build_granular_script
        layer7_script = build_granular_script(
            ops=guarded.approved_ops,
            spec=spec,
            guarded=guarded,
            process=manufacturing_method,
            material=spec.material_guess,
            tol_class=spec.tolerance_class,
            confidence=spec.confidence,
            object_name=spec.object_name,
            aesthetic_class=getattr(decomposed, 'aesthetic_class', 'mechanical'),
        )
        l7_lines = layer7_script.count("\n") + 1
        reports.append(LayerReport(
            layer=7, name="Granular Script Generation",
            elapsed_s=_t(t),
            confidence=0.95,
            notes=f"script_lines={l7_lines} | parametric+draft+fillets+gdnt",
        ))
        logger.info(f"[L7] generated {l7_lines}-line parametric script")
    except Exception as exc:
        logger.error(f"[L7] failed: {exc}", exc_info=True)
        return await _fallback(prompt, manufacturing_method, constraints, pipeline_start,
                         error=f"Layer 7 failed: {exc}")

    # ══════════════════════════════════════════════════════════════════════
    # LAYER 8 — Script Linting & Precision Overhaul
    # ══════════════════════════════════════════════════════════════════════
    final_script = layer7_script  # fallback if L8 skipped or fails
    detail_multiplier = 1.0

    if not skip_layer8:
        t = time.time()
        try:
            from services.script_refiner import refine_script_layer8
            refined = await refine_script_layer8(
                script=layer7_script,
                guarded_plan=guarded,
                object_name=spec.object_name,
                manufacturing_process=manufacturing_method,
                material=spec.material_guess,
                tolerance_class=spec.tolerance_class,
                layer14_confidence=spec.confidence,
            )
            if refined.success:
                final_script = refined.script
                detail_multiplier = refined.detail_multiplier
                l8_note = refined.refiner_notes
            else:
                l8_note = refined.error or "Layer 8 skipped"
                logger.warning(f"[L8] {l8_note}")

            reports.append(LayerReport(
                layer=8, name="Script Linting & Precision Overhaul",
                elapsed_s=_t(t),
                confidence=0.9 if refined.success else 0.7,
                notes=f"detail_multiplier={detail_multiplier:.1f}x | {l8_note}",
            ))
        except Exception as exc:
            logger.error(f"[L8] failed: {exc}", exc_info=True)
            reports.append(LayerReport(
                layer=8, name="Script Linting & Precision Overhaul",
                elapsed_s=0.0, confidence=0.7,
                notes=f"L8 skipped due to error: {exc}",
            ))
    else:
        reports.append(LayerReport(
            layer=8, name="Script Linting & Precision Overhaul (skipped)",
            confidence=0.7, notes="Layer 8 disabled for this run",
        ))

    # ══════════════════════════════════════════════════════════════════════
    # OVERALL ACCURACY SCORE
    # ══════════════════════════════════════════════════════════════════════
    confidences = [r.confidence for r in reports if r.confidence > 0]
    # Weighted geometric mean — punishes low-confidence layers more than arithmetic
    import math
    geo_mean = math.exp(sum(math.log(max(c, 0.01)) for c in confidences) / len(confidences))
    overall = round(geo_mean, 3)

    final_lines = final_script.count("\n") + 1
    total_elapsed = round(time.time() - pipeline_start, 2)
    logger.info(
        f"[Pipeline] complete — {final_lines} lines, "
        f"accuracy={overall:.3f}, elapsed={total_elapsed}s"
    )

    return PipelineResult(
        success=True,
        final_script=final_script,
        layer_reports=reports,
        overall_accuracy=overall,
        total_elapsed_s=total_elapsed,
        object_name=spec.object_name,
        decomposed_ops_count=len(ops),
        script_line_count=final_lines,
        detail_multiplier=detail_multiplier,
    )


# ── Enriched prompt builder ────────────────────────────────────────────────────

def _build_enriched_prompt(
    raw_prompt: str,
    spec: "RefinedSpec",  # type ignore — forward ref
    constraints: Optional[Dict[str, Any]],
) -> str:
    """Build an enriched prompt for the semantic decomposer using L1–L4 spec."""
    enriched = raw_prompt

    # Append dimension hints if they were extracted/inferred
    dim_hints = []
    if spec.width_mm > 0:
        dim_hints.append(f"width approximately {spec.width_mm:.1f}mm")
    if spec.height_mm > 0:
        dim_hints.append(f"height approximately {spec.height_mm:.1f}mm")
    if spec.radius_mm > 0:
        dim_hints.append(f"radius approximately {spec.radius_mm:.1f}mm")
    if spec.wall_mm > 0:
        dim_hints.append(f"wall thickness {spec.wall_mm:.1f}mm")
    if spec.capacity_ml:
        dim_hints.append(f"capacity {spec.capacity_ml:.0f}ml")

    if dim_hints:
        enriched += f"\n\n[Pipeline L1-L4 derived dimensions: {', '.join(dim_hints)}]"

    if spec.geometric_modifiers:
        enriched += f"\n[Geometric modifiers: {', '.join(spec.geometric_modifiers)}]"

    if spec.manufacturing_process:
        enriched += f"\n[Manufacturing: {spec.manufacturing_process.upper()}]"

    if constraints:
        constraint_lines = [f"  - {k}: {v}" for k, v in constraints.items()]
        enriched += "\n[User constraints:\n" + "\n".join(constraint_lines) + "]"

    return enriched


# ── Fallback to legacy pipeline ────────────────────────────────────────────────

async def _fallback(
    prompt: str,
    manufacturing_method: str,
    constraints: Optional[Dict[str, Any]],
    pipeline_start: float,
    error: str,
) -> PipelineResult:
    """
    Fall back to the existing 2-pass generate_and_execute pipeline when
    an earlier layer fails critically.
    """
    logger.warning(f"[Pipeline] falling back to legacy pipeline. Reason: {error}")
    try:
        from services.claude_cad import generate_and_execute
        result = await generate_and_execute(prompt, manufacturing_method, constraints)
        return PipelineResult(
            success=result.success,
            final_script=result.script or "",
            layer_reports=[LayerReport(
                layer=0, name="Legacy 2-Pass Fallback",
                confidence=0.7 if result.success else 0.3,
                notes=f"Fallback used. Reason: {error}",
                warnings=[result.error] if result.error else [],
            )],
            overall_accuracy=0.7 if result.success else 0.3,
            total_elapsed_s=round(time.time() - pipeline_start, 2),
            error=error if not result.success else None,
        )
    except Exception as exc:
        return PipelineResult(
            success=False,
            error=f"Both precision pipeline and legacy fallback failed. "
                  f"Original: {error}. Fallback: {exc}",
            total_elapsed_s=round(time.time() - pipeline_start, 2),
            layer_reports=[],
            overall_accuracy=0.0,
        )
