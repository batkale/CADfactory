# Analysis: Text-to-CadQuery Repository - Integration Opportunities for CADfactory

## Executive Summary

The **Text-to-CadQuery** repository is a research project (NeurIPS submission) that converts natural language descriptions into executable CadQuery Python code for 3D CAD model generation. It provides several methodologies and techniques that could significantly enhance CADfactory's capabilities.

**Key Finding**: Their evaluation and quality assessment pipeline is highly relevant for improving CADfactory's scoring and ranking systems.

---

## Repository Overview

### Purpose
Generate CadQuery-based 3D models from natural language descriptions using fine-tuned open-source LLMs.

### Key Innovation
- Data annotation using Gemini 2.0 Flash on Text2CAD dataset
- Training and fine-tuning of 6 open-source LLMs
- Multi-step evaluation pipeline with AI-powered quality assessment

### Structure
```
Text-to-CadQuery/
├── data_annotation/      (Gemini annotation scripts)
├── train/                (Training scripts for 6 LLMs)
├── inference/            (5-step evaluation pipeline)
│   ├── step1_generate_CadQuery/      (Generate code from LLMs)
│   ├── step2_clean_run_CadQuery/     (Extract & execute code)
│   ├── step3_rendering/              (Render STL with Blender)
│   ├── step4_gemini_eval/            (Gemini vision evaluation)
│   └── step5_compute_metrics/        (Geometric similarity)
└── Data files             (Annotated datasets, test sets)
```

---

## 1. Directly Applicable: Gemini Vision-Based Quality Evaluation

### What They Do (Step 4: Gemini Evaluation)

**File**: `inference/step4_gemini_eval/eval_tuned_model.py`

Their approach:
1. Render 3D models to images
2. Use **Gemini 2.0 Flash with vision capability** to evaluate match quality
3. Ask: "Does the 3D model match this natural language description?"
4. Get Yes/No evaluation from AI

**Key Code Pattern**:
```python
response = client.models.generate_content(
    model="gemini-2.0-flash",
    config=types.GenerateContentConfig(
        temperature=0,
        system_instruction="""
        Format your response like this:
        Match: Yes or No
        """
    ),
    contents=[image, f"""
        You are a product design engineer.
        Evaluate the 3D CAD model shown in the image using the following description:
        "{description}"
        Does the model match this description?
        Answer only: Yes or No
    """]
)
```

### Why It's Valuable for CADfactory

**Current GAP**: 
- CADfactory generates parts but has limited quality assessment beyond geometry parsing
- User feedback is crowdsourced (thumbs up/down)
- No automated visual evaluation

**How CADfactory Could Use It**:
1. **Quality Scoring**: Rate generated parts on semantic match to user descriptions
2. **Ranking Signal**: Add AI-evaluated quality to composite ranking (like in semantic search)
3. **Part Verification**: Ensure parts actually match what was requested
4. **User Confidence**: Show users why a part scores well/poorly
5. **Feedback Loop**: Use Gemini evaluation as training signal for fine-tuning models

**Implementation Approach**:
```python
# In services/gemini.py or new module:
def evaluate_generated_part_quality(
    description: str,
    stl_file_path: str,
    rendering_image_path: str
) -> dict:
    """
    Use Gemini 2.0 Flash vision to evaluate if generated part matches description.
    
    Returns:
    {
        "matches": "Yes" | "No" | "Partially",
        "confidence": 0-1,
        "feedback": "Explanation from Gemini",
        "quality_score": 0-100
    }
    """
    # 1. Render STL to image (if not provided)
    if not rendering_image_path:
        rendering_image_path = render_stl_to_image(stl_file_path)
    
    # 2. Call Gemini with vision
    image = Image.open(rendering_image_path)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[image, f"""
            You are a product design engineer evaluating CAD models.
            Description: "{description}"
            Does this 3D model match the description?
            Respond with:
            - Match: Yes/No/Partially
            - Why: Brief explanation
        """]
    )
    
    # 3. Parse and score
    return parse_quality_response(response)
```

**Integration Points**:
- Add to `/api/generate/` response: quality_score field
- Add to ranking: 20% AI quality evaluation + existing scores
- Store in GeneratedPart table: ai_quality_score column
- Use in `/api/search/` ranking

---

## 2. Code Cleaning & Extraction Methodology

### What They Do (Step 2: Clean & Run Code)

**Notebooks**: `inference/step2_clean_run_CadQuery/*.ipynb`

Pattern: Extract valid Python code from LLM outputs and execute safely.

### Applicable to CADfactory

CADfactory already does this (in `services/cadquery_runner.py`), but Text-to-CadQuery's approach with multiple notebooks suggests they:
1. Handle different model output formats (6 different LLMs)
2. Implement robust error handling
3. Extract code between markers
4. Validate before execution

**CADfactory Enhancement**: 
- Review their extraction patterns
- Strengthen validation for edge cases
- Handle model-specific output quirks

---

## 3. Metric Computation Pipeline

### What They Do (Step 5: Compute Metrics)

Generate metrics for evaluation:
- **Chamfer Distance** (geometric similarity between generated and expected)
- **F1 Score** (whether features are present)
- Other geometric similarity scores

### Application for CADfactory

**Current**: CADfactory stores complexity_score but lacks detailed geometric metrics

**Enhancement**: Compute Chamfer Distance when:
1. User uploads reference STL (expected result)
2. User generates a part
3. Compare generated vs reference geometry
4. Store similarity score
5. Use as ranking/quality signal

**Implementation**:
```python
from scipy.spatial.distance import cdist
import trimesh

def compute_chamfer_distance(stl_file1: str, stl_file2: str) -> float:
    """Compute Chamfer Distance between two STL files."""
    mesh1 = trimesh.load(stl_file1)
    mesh2 = trimesh.load(stl_file2)
    
    points1 = mesh1.sample(5000)
    points2 = mesh2.sample(5000)
    
    # Compute bidirectional distance
    d1 = np.min(cdist(points1, points2), axis=1).mean()
    d2 = np.min(cdist(points2, points1), axis=1).mean()
    
    return (d1 + d2) / 2
```

---

## 4. Fine-Tuning Approach with Gemini

### What They Do (Data Annotation)

Use Gemini 2.0 Flash to annotate Text2CAD dataset with CadQuery code examples.

### Application for CADfactory

**Current**: CADfactory has fine_tuner.py for Gemini supervised fine-tuning

**Enhancement Idea**:
- Use user-generated + AI-evaluated high-quality parts as training data
- Create "Golden Examples" dataset where:
  - Description is clear
  - Generated code is correct
  - Execution successful
  - Quality score high
- Fine-tune Gemini model on this dataset
- Over time, model improves with real usage data

**Pipeline**:
```
User generates part
       ↓
AI evaluates with Gemini vision
       ↓
Quality score > threshold?
       ↓ (Yes)
User thumbs-up
       ↓ (Yes)
Add to training data
       ↓
Periodically fine-tune Gemini model
       ↓
Model improves → Better generations
```

---

## 5. Multi-Model Comparison Framework

### What They Do

Train and evaluate 6 different LLMs:
- CodeGPT
- Gemma
- GPT-2 (medium/large)
- Mistral
- Qwen

Compare performance across models.

### Application for CADfactory

CADfactory currently uses:
- Gemini 2.0 Flash (main)
- Claude (alternative)

**Enhancement**: 
- Track which model performs best for different part types
- Route requests to best model (e.g., Mistral better for mechanical parts)
- Build model-specific prompts and examples
- Fallback to alternative if primary fails

---

## 6. Rendering & Visual Evaluation

### What They Do (Step 3: Rendering)

Render STL to images using Blender for visual inspection and Gemini evaluation.

### Application for CADfactory

**Current**: CADfactory doesn't render to images for user preview

**Enhancement Opportunities**:
1. Generate preview images automatically
2. Use for visual quality check
3. Display in UI to users
4. Enable Gemini vision evaluation
5. Store rendered images for comparison

---

## 7. Systematic Evaluation Framework

### What They Do

Multi-step pipeline:
```
Text Description
       ↓
Step 1: Generate CadQuery Code (6 models)
       ↓
Step 2: Extract & Execute Code (validation)
       ↓
Step 3: Render to Image (Blender)
       ↓
Step 4: Evaluate with Gemini (semantic match)
       ↓
Step 5: Compute Metrics (geometric similarity)
       ↓
Results: Pass/Fail + Score
```

### Application for CADfactory

This is **directly applicable** to CADfactory's existing pipeline (steps 1-8 in `services/pipeline.py`).

**Enhance CADfactory Pipeline**:
```
Current (8 layers + AI refinement):
  Layer 1: NLP extraction
  Layer 1.5: Reference matching
  Layer 2: Parameterization
  Layer 3: Constraint validation
  Layer 4: Gap filling
  Layer 5: CSG planning
  Layer 6: Script generation
  Layer 8: AI linting

ENHANCED (add evaluation):
  ... (all existing layers)
       ↓
  Step Post-Gen: Render to image
       ↓
  Step Quality: Gemini vision evaluation
       ↓
  Step Metrics: Compute geometric similarity
       ↓
  Confidence Score: Combined all signals
```

---

## 8. Dataset & Benchmarking

### What They Have

- **Text2CAD Dataset**: Natural language → CAD sequences
- **Annotated Dataset**: 7.4 MB JSONL test set with descriptions
- **Multiple Models**: Trained versions on HuggingFace

### Application for CADfactory

**Enhancement Idea**:
- Build CADfactory-specific dataset from successful user generations
- Include:
  - User description
  - Generated code
  - Execution success/failure
  - Geometric metrics
  - User satisfaction
  - AI quality evaluation
- Use for benchmarking
- Train fine-tuned models on CADfactory data

---

## Recommended Integration Strategy

### Phase 1: Quick Wins (This Week)
1. **Add Gemini Vision Quality Evaluation**
   - After part generation, render to image
   - Use Gemini to evaluate match quality
   - Store quality_score in database
   - Add to ranking system (weighted 10-15%)

2. **Implement Image Rendering**
   - Use existing Blender setup (if available)
   - Auto-render successful parts
   - Store in database
   - Display in UI

### Phase 2: Ranking Enhancement (Next Week)
3. **Update Semantic Search Ranking**
   - Add AI Quality Score component
   - Weight: 35% keyword + 30% feature + 20% RAG + 10% quality + 5% feedback

4. **Add Geometric Metrics**
   - Compute Chamfer Distance if reference geometry exists
   - Store metrics in database
   - Use for filtering/sorting

### Phase 3: Model Improvement (Ongoing)
5. **Build Training Dataset**
   - Collect high-quality generations
   - Annotate with quality scores
   - Create fine-tuning dataset
   - Periodically re-train Gemini model

6. **Multi-Model Comparison**
   - Test alternative models (Claude, Mistral, etc.)
   - Route based on part type
   - Fall back to best performer

---

## Implementation Code Examples

### 1. Gemini Vision Quality Evaluation

```python
# services/quality_assessment.py

from PIL import Image
import google.genai as genai

def evaluate_generated_part(
    description: str,
    stl_path: str,
    image_path: str = None
) -> dict:
    """
    Use Gemini 2.0 Flash vision to evaluate if part matches description.
    """
    # Render if needed
    if not image_path:
        image_path = render_stl_simple(stl_path)
    
    # Load image
    image = Image.open(image_path)
    
    # Call Gemini
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            temperature=0,
            system_instruction="""You are a CAD quality engineer.
            Evaluate if the 3D model matches the description.
            Respond only with JSON:
            {"match": "Yes|Partially|No", "confidence": 0-1, "reason": "..."}
            """
        ),
        contents=[
            image,
            f"""Evaluate this 3D model against description: "{description}"
            Does it match? Rate confidence 0-1."""
        ]
    )
    
    # Parse response
    result = json.loads(response.text)
    
    # Convert to score
    score_map = {"Yes": 95, "Partially": 60, "No": 20}
    quality_score = score_map.get(result["match"], 50) * result["confidence"]
    
    return {
        "match": result["match"],
        "quality_score": quality_score,
        "confidence": result["confidence"],
        "reason": result["reason"]
    }
```

### 2. Add to Generated Part Response

```python
# routers/generate.py - update generate endpoint

response = GenerateResponse(
    success=True,
    stl_url=stl_url,
    step_url=step_url,
    script=script,
    bom=bom,
    warnings=warnings,
    part_id=part.id,
    
    # NEW: Quality evaluation
    quality_assessment={
        "match": "Partially",
        "quality_score": 87.5,
        "confidence": 0.95,
        "reason": "3D model closely matches description. All major features present."
    }
)
```

### 3. Update Semantic Search Ranking

```python
# services/semantic_search.py - add quality component

def composite_ranking_score(
    query: str,
    description: str,
    manufacturing_method: str = "fdm",
    features: Optional[List[SemanticFeature]] = None,
    user_rating: Optional[int] = None,
    feedback_count: int = 0,
    quality_score: Optional[float] = None  # NEW
) -> Tuple[float, Dict[str, float]]:
    
    scores = {}
    # ... existing scores ...
    
    # NEW: AI Quality Score (15% weight)
    if quality_score is not None:
        scores["quality_score"] = quality_score / 100  # Normalize
    else:
        scores["quality_score"] = 0.5  # Default if not available
    
    # Updated weights
    composite = (
        scores["keyword_match"] * 0.30 +
        scores["feature_alignment"] * 0.25 +
        scores["rag_similarity"] * 0.15 +
        scores["manufacturing_fit"] * 0.10 +
        scores.get("quality_score", 0.5) * 0.15 +  # NEW
        scores.get("user_feedback", 0.5) * 0.05
    )
    
    return composite * 100, scores
```

---

## Files to Study in Text-to-CadQuery

1. **`inference/step4_gemini_eval/eval_tuned_model.py`** ⭐⭐⭐
   - Gemini vision evaluation pattern
   - Most directly applicable

2. **`inference/step2_clean_run_CadQuery/*.ipynb`**
   - Code extraction and validation
   - Error handling patterns

3. **`data_annotation/`**
   - How they use Gemini for annotation
   - Few-shot prompting patterns

4. **Training scripts** (in `train/`)
   - Multi-model fine-tuning approach
   - Dataset preparation

---

## Recommendations Summary

| Recommendation | Priority | Effort | Impact | Timeline |
|---|---|---|---|---|
| Add Gemini Vision Quality Evaluation | 🔴 High | 2-3 days | 🟢 High | Week 1 |
| Auto-render parts to images | 🔴 High | 1-2 days | 🟢 High | Week 1 |
| Add quality score to ranking | 🟡 Medium | 2-3 days | 🟢 High | Week 1 |
| Compute geometric metrics | 🟡 Medium | 3-4 days | 🟡 Medium | Week 2 |
| Build training dataset | 🟡 Medium | Ongoing | 🟢 High | Ongoing |
| Multi-model comparison | 🟢 Low | 1-2 weeks | 🟡 Medium | Week 3+ |
| Fine-tune on CADfactory data | 🟢 Low | 1-2 weeks | 🟢 High | Month 1 |

---

## Conclusion

**The Text-to-CadQuery repository provides valuable techniques for:**
1. ✅ AI-powered quality evaluation (Gemini vision)
2. ✅ Systematic multi-step evaluation pipeline
3. ✅ Geometric similarity metrics
4. ✅ Multi-model training and comparison
5. ✅ Dataset annotation and benchmarking

**Most immediately useful**: **Gemini 2.0 Flash vision evaluation** for quality scoring.

**Integration**: The quality evaluation perfectly complements the semantic search ranking system just implemented, creating a comprehensive scoring framework:
- Search: 35% keyword + 30% feature + 20% RAG + 10% mfg + 5% feedback
- Generation: Add 15% AI quality evaluation
- Overall: User gets relevant, high-quality results ranked intelligently.

**Next Steps**:
1. Implement Gemini vision quality evaluation
2. Add to semantic search ranking weights
3. Build training dataset from successful generations
4. Fine-tune models on CADfactory data
5. Continuously improve based on user feedback
