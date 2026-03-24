import asyncio
import os
import sys
from dotenv import load_dotenv

# Add current directory to path
sys.path.append(os.getcwd())

load_dotenv()

# Import the pipeline
from services.pipeline import run_precision_pipeline

async def verify_thinking():
    prompt = "fidget spinner with 3 arms and standard 608 bearing holes"
    manufacturing_method = "fdm"
    
    print(f"--- VERIFYING THINKING: '{prompt}' ---")
    
    try:
        # Run the full 8-layer pipeline
        result = await run_precision_pipeline(prompt, manufacturing_method)
        
        script = result.final_script or ""
        print("\n--- GENERATED SCRIPT PREVIEW ---")
        print("\n".join(script.split("\n")[:60]))
        print("\n--- ... ---")
        
        # 1. Check for 608 bearing / 22mm constants
        if "608" in script or "22" in script or "11" in script:
            print("✅ Mechanical DNA: 608 bearing / 22mm OD constants detected.")
        else:
            print("❌ Mechanical DNA: Standard bearing sizes NOT found in script.")
            
        # 2. Check for radial symmetry (3 copies)
        if "copies" in script and "3" in script and "120" in script:
             print("✅ Symmetry: Radial copy (3, 120 deg) detected.")
        elif "polarArray" in script:
             print("✅ Symmetry: polarArray detected.")
        else:
             print("❌ Symmetry: Radial symmetry logic missing.")

        # 3. Check for union logic (no disconnected parts)
        if ".union(" in script or "part_" in script:
             print("✅ Assembly: Boolean union / component tracking detected.")
        else:
             print("❌ Assembly: Missing component unions.")
             
        # 4. Accuracy score
        print(f"\nPipeline overall accuracy: {result.overall_accuracy:.2f}")
        for report in result.layer_reports:
            print(f"  Layer {report.layer}: {report.name:<30} | Confidence: {report.confidence:.2f}")

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(verify_thinking())
