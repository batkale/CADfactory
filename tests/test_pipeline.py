"""
Pipeline integration test — runs 12 prompts through the full
decompose_prompt() → build_from_prompt_result() chain and reports
per-prompt and aggregate statistics.

Run from project root:
    ./venv/bin/python tests/test_pipeline.py
"""

import os
import sys
import time

# Add project root to path so imports resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.csg_builder import build_from_prompt_result, validate_csg_tree
from services.semantic_decomposer import decompose_prompt

# ── Test prompts ───────────────────────────────────────────────────────────────

PROMPTS = [
    "stres çarkı",
    "üçgen stres çarkı",
    "makyaj süngeri",
    "hex vida M8",
    "yuvarlak kutu kapağı",
    "L-profil braket",
    "delikli plaka",
    "dişli çark",
    "içi boş silindir",
    "kalp şeklinde kolye ucu",
    "kelebek şekli",
    "soyut organik küre",
]

# ── Helpers ────────────────────────────────────────────────────────────────────

def separator(char="─", width=70):
    print(char * width)

def run_single(idx: int, prompt: str) -> dict:
    """Run one prompt through the pipeline and return a result dict."""
    result = {
        "idx": idx,
        "prompt": prompt,
        "object_name": "—",
        "confidence": 0.0,
        "clarification_needed": False,
        "ops_count": 0,
        "validation_errors": [],
        "build_success": False,
        "script_lines": 0,
        "error": None,
        "elapsed_s": 0.0,
    }

    t0 = time.time()
    try:
        decomposed = decompose_prompt(prompt)
        result["object_name"] = decomposed.object_name
        result["confidence"] = decomposed.confidence
        result["clarification_needed"] = decomposed.clarification_needed
        result["ops_count"] = len(decomposed.operations)

        val_errors = validate_csg_tree(decomposed)
        result["validation_errors"] = val_errors

        build = build_from_prompt_result(decomposed)
        result["build_success"] = build["success"]

        if build["success"] and build["script"]:
            result["script_lines"] = build["script"].count("\n") + 1
        elif not build["success"]:
            hard = [e for e in build["errors"] if not e.startswith("WARNING:")]
            result["error"] = "; ".join(hard) if hard else "unknown build error"

    except Exception as exc:
        result["error"] = str(exc)

    result["elapsed_s"] = round(time.time() - t0, 2)
    return result


def print_result(r: dict):
    separator()
    print(f"[{r['idx']:>2}] prompt          : {r['prompt']}")
    print(f"     object_name     : {r['object_name']}")
    print(f"     confidence      : {r['confidence']:.2f}")
    print(f"     clarification?  : {r['clarification_needed']}")
    print(f"     ops_count       : {r['ops_count']}")

    if r["validation_errors"]:
        for e in r["validation_errors"]:
            tag = "WARN " if e.startswith("WARNING:") else "ERROR"
            print(f"     validation [{tag}]: {e}")
    else:
        print("     validation_errors: (none)")

    status = "✓ OK" if r["build_success"] else "✗ FAIL"
    print(f"     build_success   : {status}")
    print(f"     script_lines    : {r['script_lines']}")

    if r["error"]:
        print(f"     hata            : {r['error']}")

    print(f"     elapsed         : {r['elapsed_s']}s")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    separator("═")
    print("  CADFactory — Pipeline Test (decompose → build)")
    print(f"  {len(PROMPTS)} prompts  ·  Gemini Flash decomposition  ·  deterministic build")
    separator("═")

    results = []
    for i, prompt in enumerate(PROMPTS, start=1):
        print(f"\nRunning [{i:>2}/{len(PROMPTS)}]: {prompt!r}  ...", flush=True)
        r = run_single(i, prompt)
        results.append(r)
        print_result(r)

    # ── Summary ────────────────────────────────────────────────────────────────
    separator("═")
    print("  ÖZET / SUMMARY")
    separator("═")

    n = len(results)
    successful_builds  = sum(1 for r in results if r["build_success"])
    clarification_cnt  = sum(1 for r in results if r["clarification_needed"])
    avg_confidence     = sum(r["confidence"] for r in results) / n
    total_elapsed      = sum(r["elapsed_s"] for r in results)

    # Collect all hard validation errors
    all_hard_errors = []
    for r in results:
        all_hard_errors.extend(
            e for e in r["validation_errors"] if not e.startswith("WARNING:")
        )

    # Find most frequent error type (by first 40 chars)
    from collections import Counter
    error_counts = Counter(e[:60] for e in all_hard_errors)
    most_common_error = (
        error_counts.most_common(1)[0] if error_counts else None
    )

    print(f"  Başarılı build        : {successful_builds}/{n}")
    print(f"  Ortalama confidence   : {avg_confidence:.2f}")
    print(f"  clarification_needed  : {clarification_cnt}/{n}")
    print(f"  Toplam validation hata: {len(all_hard_errors)}")

    if most_common_error:
        pattern, count = most_common_error
        print(f"  En sık validation hatası ({count}x): {pattern}…")
    else:
        print("  En sık validation hatası : (yok)")

    print(f"  Toplam süre           : {total_elapsed:.1f}s")

    separator("═")

    # Per-prompt compact table
    print("\n  PROMPT TABLOSU")
    separator()
    print(f"  {'#':>2}  {'build':^5}  {'conf':^5}  {'ops':^4}  {'lines':^6}  {'clarify':^7}  prompt")
    separator()
    for r in results:
        ok    = "✓" if r["build_success"] else "✗"
        conf  = f"{r['confidence']:.2f}"
        ops   = str(r["ops_count"])
        lines = str(r["script_lines"])
        clar  = "✓" if r["clarification_needed"] else " "
        print(f"  {r['idx']:>2}  {ok:^5}  {conf:^5}  {ops:^4}  {lines:^6}  {clar:^7}  {r['prompt']}")
    separator()

    # Exit code: 0 if all builds succeeded, 1 otherwise
    sys.exit(0 if successful_builds == n else 1)


if __name__ == "__main__":
    main()
