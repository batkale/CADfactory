"""
Sandboxed CadQuery script execution.
Runs user/AI-generated CadQuery Python scripts in an isolated subprocess
with timeout and resource limits, exports STL and STEP files.

Location: cadfactory-backend/services/cadquery_runner.py
"""

import os
import subprocess
import tempfile
import shutil
import uuid
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Directory where generated files are stored persistently
OUTPUT_DIR = os.environ.get("CADFACTORY_OUTPUT_DIR", "generated_files")

# Execution limits
EXECUTION_TIMEOUT = int(os.environ.get("CADQUERY_TIMEOUT", "60"))
MAX_SCRIPT_LENGTH = 10_000  # characters


@dataclass
class ExecutionResult:
    """Result from running a CadQuery script."""
    success: bool
    stl_path: Optional[str] = None
    step_path: Optional[str] = None
    error: Optional[str] = None
    stdout: Optional[str] = None
    execution_time_s: float = 0.0


def execute_cadquery_sandboxed(
    script: str,
    timeout: int = EXECUTION_TIMEOUT,
    export_step: bool = True,
    extra_files: Optional[dict] = None,
) -> ExecutionResult:
    """
    Execute a CadQuery script in an isolated subprocess.
    
    The script is written to a temp file with export commands injected,
    then run via subprocess with timeout. Produces STL and optionally STEP.
    
    Args:
        script: Valid CadQuery Python code. Must define a `result` variable.
        timeout: Max execution time in seconds.
        export_step: Also export STEP format alongside STL.
        
    Returns:
        ExecutionResult with paths to generated files or error details.
    """
    import time
    start_time = time.time()
    
    # Basic sanity checks
    if len(script) > MAX_SCRIPT_LENGTH:
        return ExecutionResult(
            success=False,
            error=f"Script too long ({len(script)} chars, max {MAX_SCRIPT_LENGTH})"
        )
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate unique filenames
    file_id = uuid.uuid4().hex[:12]
    final_stl = os.path.join(OUTPUT_DIR, f"{file_id}.stl")
    final_step = os.path.join(OUTPUT_DIR, f"{file_id}.step") if export_step else None
    
    with tempfile.TemporaryDirectory(prefix="cadquery_") as tmpdir:
        script_path = os.path.join(tmpdir, "part.py")
        tmp_stl = os.path.join(tmpdir, "part.stl")
        tmp_step = os.path.join(tmpdir, "part.step")
        
        # Copy any extra files (e.g. reference STEP) into tmpdir
        if extra_files:
            for dest_name, src_path in extra_files.items():
                if os.path.exists(src_path):
                    shutil.copy2(src_path, os.path.join(tmpdir, dest_name))

        # Build the execution script with exports injected
        exec_script = _build_execution_script(script, tmp_stl, tmp_step, export_step)

        with open(script_path, "w") as f:
            f.write(exec_script)
        
        try:
            result = subprocess.run(
                ["python", script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmpdir,
                env={
                    **os.environ,
                    "OMP_NUM_THREADS": "2",        # Limit CPU threads
                    "OPENBLAS_NUM_THREADS": "2",
                    "MKL_NUM_THREADS": "2",
                }
            )
            
            elapsed = time.time() - start_time
            
            if result.returncode != 0:
                error_msg = _clean_error(result.stderr)
                logger.warning(f"CadQuery script failed ({elapsed:.1f}s): {error_msg[:200]}")
                return ExecutionResult(
                    success=False,
                    error=error_msg,
                    stdout=result.stdout,
                    execution_time_s=elapsed
                )
            
            # Check that STL was actually created
            if not os.path.exists(tmp_stl) or os.path.getsize(tmp_stl) == 0:
                return ExecutionResult(
                    success=False,
                    error="Script ran but produced no STL output. "
                          "Ensure the script defines a `result` variable.",
                    stdout=result.stdout,
                    execution_time_s=elapsed
                )
            
            # Move files to persistent storage
            shutil.copy2(tmp_stl, final_stl)
            if export_step and os.path.exists(tmp_step):
                shutil.copy2(tmp_step, final_step)
            
            logger.info(f"CadQuery script succeeded ({elapsed:.1f}s): {final_stl}")
            
            return ExecutionResult(
                success=True,
                stl_path=final_stl,
                step_path=final_step if (final_step and os.path.exists(final_step)) else None,
                stdout=result.stdout,
                execution_time_s=elapsed
            )
            
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            logger.warning(f"CadQuery script timed out after {timeout}s")
            return ExecutionResult(
                success=False,
                error=f"Script execution timed out after {timeout} seconds. "
                      "Try simplifying the geometry or reducing detail.",
                execution_time_s=elapsed
            )
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Unexpected error running CadQuery: {e}")
            return ExecutionResult(
                success=False,
                error=f"Unexpected error: {str(e)}",
                execution_time_s=elapsed
            )


def _build_execution_script(
    user_script: str,
    stl_path: str,
    step_path: str,
    export_step: bool
) -> str:
    """
    Wrap the user's CadQuery script with imports and export commands.
    
    Injects:
    - cadquery import (if missing)
    - A no-op show_object function (since we're headless)
    - STL and STEP export at the end
    """
    # Escape the paths for Windows compatibility
    stl_path_escaped = stl_path.replace("\\", "\\\\")
    step_path_escaped = step_path.replace("\\", "\\\\")
    
    header = """
import sys
import cadquery as cq
from cadquery import exporters

# Patch cadquery.Vector to auto-coerce numpy scalars → Python float.
# Prevents "Multiplied(): incompatible function arguments" from OCC when
# numpy types (float32/float64/int64 etc.) are passed as dimension values.
try:
    import cadquery.occ_impl.geom as _geom
    _orig_vec = _geom.Vector.__init__
    def _safe_vec_init(self, *args, **kwargs):
        def _c(v):
            if hasattr(v, 'item'):  # numpy scalar
                return v.item()
            return v
        _orig_vec(self, *(_c(a) for a in args), **{k: _c(v) for k, v in kwargs.items()})
    _geom.Vector.__init__ = _safe_vec_init
except Exception:
    pass  # cadquery internals differ — best effort

# No-op show_object for headless execution
def show_object(obj, name=None, options=None):
    pass

"""
    
    # Remove any existing cadquery import from user script to avoid duplicates
    lines = user_script.split("\n")
    filtered_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped in ("import cadquery as cq", "import cadquery", "from cadquery import *"):
            continue
        if stripped == "from cadquery import exporters":
            continue
        filtered_lines.append(line)
    
    clean_script = "\n".join(filtered_lines)
    
    # Export commands — high quality mesh
    export_block = f"""

# === Auto-injected export ===
try:
    # High quality STL: tolerance=0.01mm linear, 0.05 rad angular
    result.val().exportStl("{stl_path_escaped}", tolerance=0.01, angularTolerance=0.05)
    print(f"STL exported: {{os.path.getsize('{stl_path_escaped}')}} bytes")
"""
    
    if export_step:
        export_block += f"""
    exporters.export(result, "{step_path_escaped}", exportType="STEP")
    print(f"STEP exported: {{os.path.getsize('{step_path_escaped}')}} bytes")
"""
    
    export_block += """
except NameError:
    print("ERROR: No 'result' variable defined in script", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"ERROR exporting: {e}", file=sys.stderr)
    sys.exit(1)
"""
    
    # Need os for file size check
    full_script = "import os\n" + header + clean_script + export_block
    return full_script


def _clean_error(stderr: str) -> str:
    """
    Clean up Python traceback to show only the relevant error.
    Strips the temp file paths and internal frames.
    """
    if not stderr:
        return "Unknown error (no stderr output)"
    
    lines = stderr.strip().split("\n")
    
    # Find the actual error message (usually the last line)
    error_line = lines[-1] if lines else "Unknown error"
    
    # Also grab context if it's a CadQuery-specific error
    relevant_lines = []
    capture = False
    for line in lines:
        if "cadquery" in line.lower() or "ocp" in line.lower() or "Error" in line:
            capture = True
        if capture:
            # Remove temp file paths
            cleaned = line.replace("/tmp/", "").strip()
            if cleaned:
                relevant_lines.append(cleaned)
    
    if relevant_lines:
        return "\n".join(relevant_lines[-5:])  # Last 5 relevant lines
    
    return error_line


def get_file_url(file_path: str) -> str:
    """Convert a local file path to an API-servable URL."""
    filename = os.path.basename(file_path)
    return f"/api/generate/files/{filename}"


def cleanup_old_files(max_age_hours: int = 24):
    """Remove generated files older than max_age_hours."""
    import time
    
    if not os.path.exists(OUTPUT_DIR):
        return
    
    cutoff = time.time() - (max_age_hours * 3600)
    removed = 0
    
    for filename in os.listdir(OUTPUT_DIR):
        filepath = os.path.join(OUTPUT_DIR, filename)
        if os.path.getmtime(filepath) < cutoff:
            os.remove(filepath)
            removed += 1
    
    if removed:
        logger.info(f"Cleaned up {removed} old generated files")