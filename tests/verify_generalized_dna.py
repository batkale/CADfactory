import asyncio
import os
import sys

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.pipeline import run_precision_pipeline


async def test_generalization():
    print("--- TESTING GENERALIZED MECHANICAL DNA ---")

    prompts = [
        "industrial cooling fan with 7 blades",
        "protective case for a raspberry pi",
        "heavy duty wall shelf bracket",
        "fidget spinner with 3 arms",
        "stylish soap dispenser with pump neck"
    ]

    for prompt in prompts:
        print(f"\n[TESTING] '{prompt}'")
        try:
            res = await run_precision_pipeline(prompt)
            if not res.success:
                print(f"FAILED: {res.error}")
                continue

            # Print Layer 1.5 Match
            match_report = next((r for r in res.layer_reports if r.layer == 1.5), None)
            if match_report:
                print(f"L1.5 Match: {match_report.notes}")
            else:
                print("L1.5 Match: NOT FOUND (possibly fell back to legacy)")
                if res.layer_reports:
                    print(f"First report: {res.layer_reports[0].name} - {res.layer_reports[0].notes}")

            # Print Decomposer result (Layer 4)
            decomp_report = next((r for r in res.layer_reports if r.name == "Semantic Decomposition"), None)
            if decomp_report:
                print(f"L4 Decomposer: {decomp_report.notes}")

        except Exception as e:
            print(f"EXCEPTION: {e}")

if __name__ == "__main__":
    asyncio.run(test_generalization())
