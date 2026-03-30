"""
3D Topology Optimization Service — SIMP method.

Pipeline:
  STL → voxelize (numpy ray-casting) → SIMP FEA loop (scipy.sparse)
      → threshold density → surface extraction → binary STL bytes
      → cost/weight savings using existing material database.

No external CAD libraries required (no trimesh, no scikit-image).
Dependencies: numpy (already installed), scipy (added to requirements.txt).

Performance notes:
  - Uses float32 for voxel grids and mesh data (halves memory vs float64)
  - FEA solver uses float64 for numerical stability (sparse linear solve)
  - Seed management via services.seed_manager for reproducible results
"""

import logging
import math
import os
import struct
import xml.etree.ElementTree as ET
import zipfile
from itertools import product as iproduct

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

from services.seed_manager import get_seed_from_env, set_seed

logger = logging.getLogger(__name__)

# ── Material defaults (g/cm³ and $/kg) ───────────────────────────────────────
MATERIAL_PROPS = {
    "pla":        {"density": 1.24, "cost_per_kg": 25.0,  "name": "PLA"},
    "abs":        {"density": 1.05, "cost_per_kg": 22.0,  "name": "ABS"},
    "petg":       {"density": 1.27, "cost_per_kg": 28.0,  "name": "PETG"},
    "nylon":      {"density": 1.14, "cost_per_kg": 45.0,  "name": "Nylon"},
    "al6061":     {"density": 2.70, "cost_per_kg": 5.50,  "name": "Al 6061-T6"},
    "steel":      {"density": 7.85, "cost_per_kg": 1.20,  "name": "Steel"},
    "titanium":   {"density": 4.43, "cost_per_kg": 80.0,  "name": "Titanium"},
}


# ─────────────────────────────────────────────────────────────────────────────
# STL I/O
# ─────────────────────────────────────────────────────────────────────────────

def _load_3mf_triangles(path: str) -> np.ndarray:
    """Load a 3MF file → (N, 3, 3) float32 array of triangles."""
    _3MF_NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
    all_tris = []
    with zipfile.ZipFile(path, "r") as zf:
        # Find model file (always 3D/3dmodel.model per spec, but scan just in case)
        model_names = [n for n in zf.namelist() if n.endswith(".model")]
        if not model_names:
            raise ValueError("3MF file contains no .model file")
        for model_name in model_names:
            xml_data = zf.read(model_name)
            root = ET.fromstring(xml_data)
            for mesh in root.iter(f"{{{_3MF_NS}}}mesh"):
                verts_el = mesh.find(f"{{{_3MF_NS}}}vertices")
                tris_el  = mesh.find(f"{{{_3MF_NS}}}triangles")
                if verts_el is None or tris_el is None:
                    continue
                verts = np.array(
                    [[float(v.get("x", 0)), float(v.get("y", 0)), float(v.get("z", 0))]
                     for v in verts_el.iter(f"{{{_3MF_NS}}}vertex")],
                    dtype=np.float32,
                )
                if len(verts) == 0:
                    continue
                for tri in tris_el.iter(f"{{{_3MF_NS}}}triangle"):
                    i0 = int(tri.get("v1", 0))
                    i1 = int(tri.get("v2", 1))
                    i2 = int(tri.get("v3", 2))
                    all_tris.append([verts[i0], verts[i1], verts[i2]])
    if not all_tris:
        raise ValueError("3MF file contains no mesh geometry")
    return np.array(all_tris, dtype=np.float32)


def load_stl_triangles(stl_path: str) -> np.ndarray:
    """Load binary STL, ASCII STL, or 3MF → (N, 3, 3) float32 array of triangles."""
    with open(stl_path, "rb") as f:
        raw = f.read()

    # 3MF files are ZIP archives (magic bytes PK\x03\x04)
    if raw[:4] == b"PK\x03\x04":
        return _load_3mf_triangles(stl_path)

    if len(raw) < 84:
        raise ValueError("STL file is too small to be valid")

    # Check if binary triangle count is consistent with file size.
    # Many binary STLs write "solid <name>" in the 80-byte header, so we
    # cannot rely on the "solid" prefix alone — validate the size first.
    n_binary = struct.unpack("<I", raw[80:84])[0]
    expected_binary_size = 84 + n_binary * 50
    is_valid_binary = (expected_binary_size == len(raw))

    if is_valid_binary:
        tris = np.zeros((n_binary, 3, 3), dtype=np.float32)
        offset = 84
        for i in range(n_binary):
            offset += 12  # normal
            for j in range(3):
                tris[i, j] = struct.unpack_from("<fff", raw, offset)
                offset += 12
            offset += 2   # attribute
        return tris

    # Binary size doesn't match → try ASCII (handles UTF-8 BOM, uppercase SOLID,
    # or any binary STL whose header triangle count is corrupt/missing)
    text = raw.decode("utf-8", errors="replace").lstrip("\ufeff \t\r\n")
    if text.lower().startswith("solid"):
        tris = _load_stl_ascii(text)
        if len(tris) > 0:
            return tris

    # Last resort: if the file is slightly larger than expected (trailing padding)
    # try reading as binary with the declared count anyway.
    if len(raw) >= expected_binary_size and n_binary > 0:
        tris = np.zeros((n_binary, 3, 3), dtype=np.float32)
        offset = 84
        for i in range(n_binary):
            offset += 12
            for j in range(3):
                tris[i, j] = struct.unpack_from("<fff", raw, offset)
                offset += 12
            offset += 2
        return tris

    raise ValueError(
        f"Could not parse STL file ({len(raw)} bytes). "
        "Try re-exporting as binary STL from your CAD tool."
    )


def _load_stl_ascii(text: str) -> np.ndarray:
    """Parse ASCII STL text → (N, 3, 3) float32 array."""
    tris = []
    current = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("vertex"):
            parts = line.split()
            current.append([float(parts[1]), float(parts[2]), float(parts[3])])
            if len(current) == 3:
                tris.append(current)
                current = []
    if not tris:
        raise ValueError("ASCII STL contains no vertices")
    return np.array(tris, dtype=np.float32)


def write_stl_binary(vertices: np.ndarray, faces: np.ndarray) -> bytes:
    """Write mesh (Vx3, Fx3 int) → binary STL bytes."""
    tris = vertices[faces].astype(np.float32)   # (F, 3, 3)
    return _tris_to_stl_bytes(tris)


def _tris_to_stl_bytes(tris: np.ndarray) -> bytes:
    """Vectorised (F,3,3) float32 → binary STL bytes. No Python loop over triangles."""
    n = len(tris)
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    normals = np.cross(v1 - v0, v2 - v0).astype(np.float32)
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    np.maximum(lens, 1e-10, out=lens)
    normals /= lens

    # Binary STL record: 12B normal + 36B verts + 2B attr = 50B each
    rec = np.zeros(n, dtype=np.dtype([
        ('n',    np.float32, (3,)),
        ('v0',   np.float32, (3,)),
        ('v1',   np.float32, (3,)),
        ('v2',   np.float32, (3,)),
        ('attr', np.uint16),
    ]))
    rec['n']  = normals
    rec['v0'] = v0
    rec['v1'] = v1
    rec['v2'] = v2
    header = b'\x00' * 80 + struct.pack('<I', n)
    return header + rec.tobytes()


def _write_stl_fast(tris: np.ndarray, path: str) -> None:
    """Write (N,3,3) float32 triangles directly to a binary STL file."""
    with open(path, 'wb') as f:
        f.write(_tris_to_stl_bytes(tris))


# ─────────────────────────────────────────────────────────────────────────────
# Voxelization (numpy ray-casting, no trimesh)
# ─────────────────────────────────────────────────────────────────────────────

def _ray_z_intersections(triangles: np.ndarray, cx: float, cy: float) -> np.ndarray:
    """Möller–Trumbore: find all Z where a +Z ray at (cx,cy) hits the mesh."""
    v0, v1, v2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    e1 = v1 - v0
    e2 = v2 - v0

    # D = (0,0,1) → h = D × e2 = (-e2_y, e2_x, 0)
    hx = -e2[:, 1]
    hy =  e2[:, 0]
    a  = e1[:, 0] * hx + e1[:, 1] * hy   # dot(e1, h); e1·h_z=0

    valid = np.abs(a) > 1e-10
    if not valid.any():
        return np.empty(0)

    with np.errstate(divide='ignore', invalid='ignore'):
        f = np.where(valid, 1.0 / a, 0.0)

    sx = cx - v0[:, 0]
    sy = cy - v0[:, 1]
    # sz not needed: s·h_z = 0

    u = f * (sx * hx + sy * hy)
    mask2 = valid & (u >= -1e-8) & (u <= 1 + 1e-8)
    if not mask2.any():
        return np.empty(0)

    # q = s × e1 (only z-component needed for v)
    qz = sx * e1[:, 1] - sy * e1[:, 0]
    v_ = f * qz                            # v = f * dot(D, q) = f * q_z
    mask3 = mask2 & (v_ >= -1e-8) & (u + v_ <= 1 + 1e-8)
    if not mask3.any():
        return np.empty(0)

    # z = v0_z + e1_z*u + e2_z*v
    z = v0[:, 2][mask3] + e1[:, 2][mask3] * u[mask3] + e2[:, 2][mask3] * v_[mask3]
    return z


def _detect_geometry_type(extents: np.ndarray) -> str:
    """
    Classify geometry as 'plate', 'beam', or 'bulk' based on aspect ratios.

    - plate: one dimension is much thinner than the other two (aspect > 3:1)
    - beam:  two dimensions are much thinner than the third (aspect > 3:1)
    - bulk:  roughly equal dimensions
    """
    sorted_ext = np.sort(extents)  # ascending: thin, mid, thick
    if sorted_ext[0] < 1e-6:
        return "plate"
    ratio_thin = sorted_ext[2] / sorted_ext[0]   # thick / thinnest
    ratio_mid  = sorted_ext[2] / sorted_ext[1]   # thick / middle

    if ratio_thin > 3.0 and ratio_mid < 2.0:
        # One axis much thinner, other two similar → plate
        return "plate"
    elif ratio_thin > 3.0 and ratio_mid > 3.0:
        # Two axes much thinner → beam
        return "beam"
    return "bulk"


def voxelize(triangles: np.ndarray, resolution: int = 20):
    """
    Convert triangle mesh to boolean voxel grid using Z-axis ray casting.

    For plate-like geometries (one thin dimension), ensures adequate voxels
    through the thickness so topology optimization can create through-holes.

    Returns:
        grid  : (nx, ny, nz) bool array  — True = inside mesh
        origin: (3,) float               — world-space origin of grid corner
        pitch : float                    — voxel size in mm
    """
    verts = triangles.reshape(-1, 3)
    mn = verts.min(axis=0)
    mx = verts.max(axis=0)
    extents = mx - mn
    extents = np.maximum(extents, 1e-6)

    geo_type = _detect_geometry_type(extents)
    max_ext = extents.max()

    # For plate-like shapes, use the thin dimension to set the pitch so that
    # every axis gets adequate voxels. This ensures through-holes can form.
    # Minimum 4 voxels in any dimension for meaningful topology variation.
    MIN_VOXELS_THIN = 4

    if geo_type == "plate":
        min_ext = extents.min()
        # Pitch must be small enough for MIN_VOXELS_THIN in the thin direction
        pitch_from_thin = min_ext / MIN_VOXELS_THIN
        pitch_from_res  = max_ext / resolution
        pitch = min(pitch_from_thin, pitch_from_res)
        # Cap total voxel count to avoid memory explosion
        total_est = (extents[0] / pitch) * (extents[1] / pitch) * (extents[2] / pitch)
        max_voxels = resolution ** 3 * 2  # allow up to 2x the nominal budget
        if total_est > max_voxels:
            scale = (total_est / max_voxels) ** (1.0 / 3.0)
            pitch *= scale
        logger.info(
            f"Plate geometry detected (extents {extents[0]:.1f}×{extents[1]:.1f}×{extents[2]:.1f}), "
            f"pitch={pitch:.2f}mm for through-hole support"
        )
    else:
        pitch = max_ext / resolution

    nx = max(2, int(math.ceil(extents[0] / pitch)))
    ny = max(2, int(math.ceil(extents[1] / pitch)))
    nz = max(2, int(math.ceil(extents[2] / pitch)))

    # Enforce minimum voxels in every dimension
    for dim_size, dim_name in [(nx, 'x'), (ny, 'y'), (nz, 'z')]:
        if dim_size < MIN_VOXELS_THIN:
            logger.warning(
                f"Thin {dim_name}-dimension: {dim_size} voxels < {MIN_VOXELS_THIN} minimum"
            )

    grid = np.zeros((nx, ny, nz), dtype=bool)

    for ix in range(nx):
        for iy in range(ny):
            cx = mn[0] + (ix + 0.5) * pitch
            cy = mn[1] + (iy + 0.5) * pitch

            z_hits = np.sort(_ray_z_intersections(triangles, cx, cy))

            # Parity rule: pairs of intersections define inside regions
            for k in range(0, len(z_hits) - 1, 2):
                z_lo, z_hi = z_hits[k], z_hits[k + 1]
                iz_lo = max(0, int((z_lo - mn[2]) / pitch))
                iz_hi = min(nz - 1, int((z_hi - mn[2]) / pitch))
                grid[ix, iy, iz_lo : iz_hi + 1] = True

    return grid, mn, pitch


# ─────────────────────────────────────────────────────────────────────────────
# 3-D SIMP topology optimization (hexahedral FEA)
# ─────────────────────────────────────────────────────────────────────────────

def _unit_ke(nu: float = 0.3) -> np.ndarray:
    """
    24×24 stiffness matrix for a unit hexahedral element with E=1.
    Computed via 2×2×2 Gauss integration.
    Node ordering: (±1, ±1, ±1) corners, same as standard FEM.
    """
    nodes_ref = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1],
    ], dtype=float)

    lam = nu / ((1 + nu) * (1 - 2 * nu))
    mu  = 0.5 / (1 + nu)
    D = np.array([
        [lam + 2*mu, lam,       lam,       0,  0,  0 ],
        [lam,        lam + 2*mu, lam,      0,  0,  0 ],
        [lam,        lam,       lam + 2*mu, 0,  0,  0 ],
        [0,          0,          0,         mu, 0,  0 ],
        [0,          0,          0,         0,  mu, 0 ],
        [0,          0,          0,         0,  0,  mu],
    ])

    g = 1.0 / math.sqrt(3)
    KE = np.zeros((24, 24))

    for xi, eta, zeta in iproduct([-g, g], repeat=3):
        dN = np.zeros((3, 8))
        for i, (a, b, c) in enumerate(nodes_ref):
            dN[0, i] = a * (1 + b * eta) * (1 + c * zeta) / 8
            dN[1, i] = b * (1 + a * xi)  * (1 + c * zeta) / 8
            dN[2, i] = c * (1 + a * xi)  * (1 + b * eta)  / 8

        J = dN @ nodes_ref          # 3×3; for unit element this = I
        detJ = abs(np.linalg.det(J))
        dN_dx = np.linalg.solve(J, dN)  # 3×8

        B = np.zeros((6, 24))
        for i in range(8):
            B[0, 3*i]   = dN_dx[0, i]
            B[1, 3*i+1] = dN_dx[1, i]
            B[2, 3*i+2] = dN_dx[2, i]
            B[3, 3*i]   = dN_dx[1, i]; B[3, 3*i+1] = dN_dx[0, i]  # noqa: E702
            B[4, 3*i+1] = dN_dx[2, i]; B[4, 3*i+2] = dN_dx[1, i]  # noqa: E702
            B[5, 3*i]   = dN_dx[2, i]; B[5, 3*i+2] = dN_dx[0, i]  # noqa: E702

        KE += B.T @ D @ B * detJ  # weight = 1 for 2-point Gauss

    return KE


def _elem_dofs(ex: int, ey: int, ez: int, nx: int, ny: int) -> np.ndarray:
    """Return 24 global DOF indices for element (ex, ey, ez)."""
    nodes = np.array([
        (ex,   ey,   ez),   (ex+1, ey,   ez),
        (ex+1, ey+1, ez),   (ex,   ey+1, ez),
        (ex,   ey,   ez+1), (ex+1, ey,   ez+1),
        (ex+1, ey+1, ez+1), (ex,   ey+1, ez+1),
    ])
    # Node global index: iz*(ny+1)*(nx+1) + iy*(nx+1) + ix
    ni = nodes[:, 2] * (ny + 1) * (nx + 1) + nodes[:, 1] * (nx + 1) + nodes[:, 0]
    dofs = np.zeros(24, dtype=int)
    for k, n in enumerate(ni):
        dofs[3*k:3*k+3] = [3*n, 3*n+1, 3*n+2]
    return dofs


def _heaviside_projection(
    x: np.ndarray, beta: float, eta: float = 0.5,
) -> np.ndarray:
    """
    Smooth Heaviside projection to push intermediate densities toward 0/1.

    Parameters
    ----------
    x    : physical densities in [0, 1]
    beta : sharpness parameter (higher = sharper threshold)
    eta  : threshold (default 0.5)

    Returns projected densities closer to binary 0/1.
    """
    num = np.tanh(beta * eta) + np.tanh(beta * (x - eta))
    den = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    return num / den


def _heaviside_derivative(
    x: np.ndarray, beta: float, eta: float = 0.5,
) -> np.ndarray:
    """Derivative of smooth Heaviside projection w.r.t. x."""
    den = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    cosh_val = np.cosh(beta * (x - eta))
    # Protect against overflow for large beta
    cosh_val = np.minimum(cosh_val, 1e10)
    return beta / (cosh_val ** 2 * den)


def run_simp(
    grid: np.ndarray,
    volfrac: float = 0.4,
    penal: float = 3.0,
    rmin: float = 1.4,
    fixed_side: str = "bottom",
    load_side: str = "top",
    load_dir: int = 2,           # 0=x, 1=y, 2=z
    load_magnitude: float = -1.0,
    max_iter: int = 80,
    E0: float = 1.0,
    Emin: float = 1e-9,
) -> np.ndarray:
    """
    3-D SIMP topology optimisation with continuation, density filtering,
    and Heaviside projection for crisp 0/1 results.

    Parameters
    ----------
    grid        : (nx, ny, nz) bool — solid design domain from voxelisation
    volfrac     : target volume fraction (0–1), e.g. 0.4 = keep 40 % of material
    penal       : final SIMP penalisation exponent (typically 3).
                  A continuation scheme ramps from 1 up to this value.
    rmin        : density filter radius (in voxels)
    fixed_side  : which face is clamped — 'bottom'|'top'|'left'|'right'|'front'|'back'
    load_side   : which face receives the load
    load_dir    : force direction (0=x, 1=y, 2=z)
    load_magnitude : signed force (negative = inward / compressive)
    max_iter    : maximum SIMP iterations
    E0, Emin    : full and void Young's moduli (relative units)

    Returns
    -------
    density : (nx, ny, nz) float — optimised element densities [0, 1]
    """
    nx, ny, nz = grid.shape

    nelem = nx * ny * nz
    ndof  = 3 * (nx + 1) * (ny + 1) * (nz + 1)

    # Precompute element stiffness matrix (same for all elements, unit size)
    # KE stays float64 for numerical stability in the linear solve
    KE = _unit_ke()

    # ── Precompute element→DOF mapping and COO row/col indices ────────────────
    logger.info(f"SIMP: assembling DOF maps for {nx}×{ny}×{nz} grid ({nelem} elements)")

    all_edofs = np.zeros((nelem, 24), dtype=int)
    eidx = 0
    for ez in range(nz):
        for ey in range(ny):
            for ex in range(nx):
                all_edofs[eidx] = _elem_dofs(ex, ey, ez, nx, ny)
                eidx += 1

    # COO indices for sparse K assembly
    rows = np.repeat(all_edofs, 24, axis=1).reshape(-1)   # each row repeated 24 times
    cols = np.tile(all_edofs, (1, 24)).reshape(-1)         # each col tiled 24 times
    ke_flat = KE.flatten()                                  # shape (576,)

    # ── Identify active (solid) elements ─────────────────────────────────────
    active = grid.reshape(-1).astype(float)  # 1=solid, 0=void; indexed [ez,ey,ex] flattened

    # ── Initial density (float32 for memory efficiency) ────────────────────
    x = np.full(nelem, volfrac, dtype=np.float32)
    x[active == 0] = Emin   # void cells stay void

    # ── Fixed DOFs ────────────────────────────────────────────────────────────
    fixed_dofs = _boundary_dofs(fixed_side, nx, ny, nz)
    free_dofs  = np.setdiff1d(np.arange(ndof), fixed_dofs)

    # ── Load vector ───────────────────────────────────────────────────────────
    F = _load_vector(load_side, load_dir, load_magnitude, nx, ny, nz, ndof)

    # ── Density filter weights ────────────────────────────────────────────────
    H, Hs = _filter_weights(nx, ny, nz, rmin)

    # ── Continuation schedule ─────────────────────────────────────────────────
    # Ramp penalization from 1 → penal_final over the first ~60% of iterations.
    # This avoids local minima that occur when starting with high penalization.
    penal_final = penal
    penal_start = 1.0
    continuation_iters = int(max_iter * 0.6)

    # Heaviside projection: ramp beta from 1 → 32 to progressively sharpen
    # intermediate densities toward 0/1. This produces cleaner topology results.
    beta = 1.0
    beta_max = 32.0
    beta_increase_interval = max(8, max_iter // 8)

    prev_obj = float("inf")

    # ── SIMP iterations ───────────────────────────────────────────────────────
    for iteration in range(max_iter):
        # Continuation: ramp penalization
        if iteration < continuation_iters:
            cur_penal = penal_start + (penal_final - penal_start) * (iteration / continuation_iters)
        else:
            cur_penal = penal_final

        # Increase Heaviside sharpness periodically
        if iteration > 0 and iteration % beta_increase_interval == 0 and beta < beta_max:
            beta = min(beta * 2, beta_max)
            logger.debug(f"  Heaviside beta → {beta}")

        # Apply density filter: x_filtered = H @ x / Hs
        x_filt = np.asarray(H @ x).flatten() / Hs
        x_filt = np.clip(x_filt, Emin, 1.0)

        # Apply Heaviside projection for crisp 0/1 boundaries
        x_phys = _heaviside_projection(x_filt, beta)
        x_phys = np.clip(x_phys, Emin, 1.0).astype(np.float32)
        x_phys[active == 0] = Emin

        # Effective modulus per element
        xp = Emin + x_phys ** cur_penal * (E0 - Emin)

        # Assemble global K
        vals = np.outer(xp, ke_flat).reshape(-1)
        K = coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()

        # Solve (free DOFs only)
        Kff = K[free_dofs][:, free_dofs]
        Ff  = F[free_dofs]
        try:
            Uf = spsolve(Kff, Ff)
        except Exception as e:
            logger.error(f"SIMP solve failed at iter {iteration}: {e}")
            break

        U = np.zeros(ndof)
        U[free_dofs] = Uf

        # Element sensitivities: ce = u_e^T * KE * u_e
        ue = U[all_edofs]                    # (nelem, 24)
        ce = np.einsum("ij,jk,ik->i", ue, KE, ue)   # element compliance
        obj = float(np.sum(xp * ce))

        # Sensitivity: dc/dx_phys = -p * x_phys^(p-1) * (E0 - Emin) * ce
        dc = -cur_penal * x_phys ** (cur_penal - 1) * (E0 - Emin) * ce
        dv = np.ones(nelem)

        # Chain rule through Heaviside projection
        dh = _heaviside_derivative(x_filt, beta)
        dc = dc * dh
        dv = dv * dh

        # Chain rule through density filter: df/dx = H^T @ (df/dx_filt) / Hs
        dc = np.asarray(H.T @ (dc / Hs)).flatten()
        dv = np.asarray(H.T @ (dv / Hs)).flatten()

        # OC update (optimality criteria) on design variables
        x_new = _oc_update(x, dc, dv, volfrac, active)

        change = np.max(np.abs(x_new - x))
        obj_change = abs(obj - prev_obj) / max(abs(obj), 1e-12)
        x = x_new
        prev_obj = obj

        logger.debug(
            f"  iter {iteration+1:3d}  p={cur_penal:.2f}  β={beta:.0f}  "
            f"obj={obj:.4e}  vol={x_phys.mean():.3f}  Δx={change:.4f}"
        )

        # Converge only after continuation is done and changes are small
        if iteration > continuation_iters and change < 0.005 and obj_change < 1e-4:
            logger.info(f"SIMP converged at iteration {iteration+1}")
            break

    # Return the final physical (filtered + projected) densities
    x_filt = np.asarray(H @ x).flatten() / Hs
    x_filt = np.clip(x_filt, Emin, 1.0)
    x_phys = _heaviside_projection(x_filt, beta)
    x_phys = np.clip(x_phys, Emin, 1.0).astype(np.float32)
    x_phys[active == 0] = Emin

    return x_phys.reshape(nx, ny, nz)


def _boundary_dofs(side: str, nx: int, ny: int, nz: int) -> np.ndarray:
    """Return global DOF indices for all nodes on the specified face."""
    dofs = []

    def nn(ix, iy, iz):
        return iz * (ny + 1) * (nx + 1) + iy * (nx + 1) + ix

    if side == "bottom":    # iz=0
        for ix in range(nx + 1):
            for iy in range(ny + 1):
                n = nn(ix, iy, 0)
                dofs += [3*n, 3*n+1, 3*n+2]
    elif side == "top":     # iz=nz
        for ix in range(nx + 1):
            for iy in range(ny + 1):
                n = nn(ix, iy, nz)
                dofs += [3*n, 3*n+1, 3*n+2]
    elif side == "left":    # ix=0
        for iy in range(ny + 1):
            for iz in range(nz + 1):
                n = nn(0, iy, iz)
                dofs += [3*n, 3*n+1, 3*n+2]
    elif side == "right":   # ix=nx
        for iy in range(ny + 1):
            for iz in range(nz + 1):
                n = nn(nx, iy, iz)
                dofs += [3*n, 3*n+1, 3*n+2]
    elif side == "front":   # iy=0
        for ix in range(nx + 1):
            for iz in range(nz + 1):
                n = nn(ix, 0, iz)
                dofs += [3*n, 3*n+1, 3*n+2]
    elif side == "back":    # iy=ny
        for ix in range(nx + 1):
            for iz in range(nz + 1):
                n = nn(ix, ny, iz)
                dofs += [3*n, 3*n+1, 3*n+2]

    return np.unique(np.array(dofs, dtype=int))


def _load_vector(
    side: str, direction: int, magnitude: float,
    nx: int, ny: int, nz: int, ndof: int,
) -> np.ndarray:
    """Apply distributed unit load on the specified face in the given direction."""
    F = np.zeros(ndof)

    def nn(ix, iy, iz):
        return iz * (ny + 1) * (nx + 1) + iy * (nx + 1) + ix

    nodes = []
    if side == "top":
        nodes = [nn(ix, iy, nz) for ix in range(nx+1) for iy in range(ny+1)]
    elif side == "bottom":
        nodes = [nn(ix, iy, 0)  for ix in range(nx+1) for iy in range(ny+1)]
    elif side == "left":
        nodes = [nn(0,  iy, iz) for iy in range(ny+1) for iz in range(nz+1)]
    elif side == "right":
        nodes = [nn(nx, iy, iz) for iy in range(ny+1) for iz in range(nz+1)]
    elif side == "front":
        nodes = [nn(ix, 0,  iz) for ix in range(nx+1) for iz in range(nz+1)]
    elif side == "back":
        nodes = [nn(ix, ny, iz) for ix in range(nx+1) for iz in range(nz+1)]

    if nodes:
        unit_load = magnitude / len(nodes)
        for n in nodes:
            F[3 * n + direction] += unit_load

    return F


def _filter_weights(nx: int, ny: int, nz: int, rmin: float):
    """Build sparse sensitivity filter weight matrix H and its row sums Hs."""
    nelem = nx * ny * nz
    rmin2 = rmin ** 2
    rows, cols, vals = [], [], []

    for ez in range(nz):
        for ey in range(ny):
            for ex in range(nx):
                ei = ez * ny * nx + ey * nx + ex
                # Only check elements within rmin+1 integer radius
                r = int(math.ceil(rmin))
                for dz in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        for dx in range(-r, r + 1):
                            fx, fy, fz = ex + dx, ey + dy, ez + dz
                            if 0 <= fx < nx and 0 <= fy < ny and 0 <= fz < nz:
                                dist2 = dx*dx + dy*dy + dz*dz
                                if dist2 <= rmin2:
                                    fi = fz * ny * nx + fy * nx + fx
                                    w = rmin - math.sqrt(dist2)
                                    rows.append(ei)
                                    cols.append(fi)
                                    vals.append(w)

    H  = csr_matrix((vals, (rows, cols)), shape=(nelem, nelem))
    Hs = np.array(H.sum(axis=1)).flatten()
    return H, Hs


def _oc_update(
    x: np.ndarray, dc: np.ndarray, dv: np.ndarray,
    volfrac: float, active: np.ndarray,
    x_min: float = 1e-3, x_max: float = 1.0, move: float = 0.2,
) -> np.ndarray:
    """Optimality criteria density update with bisection on the Lagrange multiplier."""
    l1, l2 = 0.0, 1e9
    xnew = x.copy()

    while (l2 - l1) / (l1 + l2 + 1e-40) > 1e-4:
        lmid = 0.5 * (l1 + l2)
        B = np.sqrt(np.maximum(0, -dc / (dv * lmid)))
        xnew = np.clip(x * B, np.maximum(x_min, x - move), np.minimum(x_max, x + move))
        xnew[active == 0] = x_min   # keep void cells void

        if xnew[active > 0].mean() > volfrac:
            l1 = lmid
        else:
            l2 = lmid

    return xnew


# ─────────────────────────────────────────────────────────────────────────────
# Surface extraction (voxel boundary faces → STL mesh)
# ─────────────────────────────────────────────────────────────────────────────

def extract_surface(
    density: np.ndarray,
    threshold: float,
    origin: np.ndarray,
    pitch: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract a surface mesh from a density field by thresholding and finding
    boundary voxel faces. No marching cubes required.

    Returns (vertices, faces) — faces index into vertices.
    """
    solid = density > threshold
    nx, ny, nz = solid.shape

    verts = {}   # (ix,iy,iz corner) → vertex index
    tri_faces = []

    def vi(ix, iy, iz):
        key = (ix, iy, iz)
        if key not in verts:
            verts[key] = len(verts)
        return verts[key]

    # For each solid voxel, check 6 face neighbours
    # Face normal directions: ±x, ±y, ±z
    neighbours = [
        ((1,0,0),  [(0,0,0),(0,1,0),(0,1,1),(0,0,1)], False),   # +x face (ex+1, ...)
        ((-1,0,0), [(0,0,0),(0,0,1),(0,1,1),(0,1,0)], True ),   # -x face
        ((0,1,0),  [(0,0,0),(1,0,0),(1,0,1),(0,0,1)], True ),   # +y face
        ((0,-1,0), [(0,0,0),(0,0,1),(1,0,1),(1,0,0)], False),   # -y face
        ((0,0,1),  [(0,0,0),(1,0,0),(1,1,0),(0,1,0)], True ),   # +z face
        ((0,0,-1), [(0,0,0),(0,1,0),(1,1,0),(1,0,0)], False),   # -z face
    ]

    for ez in range(nz):
        for ey in range(ny):
            for ex in range(nx):
                if not solid[ex, ey, ez]:
                    continue
                for (dx,dy,dz), corners, flip in neighbours:
                    nx_ = ex+dx
                    ny_ = ey+dy
                    nz_ = ez+dz
                    # Expose face if neighbour is empty or out of bounds
                    if (0 <= nx_ < nx and 0 <= ny_ < ny and 0 <= nz_ < nz
                            and solid[nx_, ny_, nz_]):
                        continue
                    # Map corners to node indices for this voxel
                    c = [(ex+cx, ey+cy, ez+cz) for cx,cy,cz in corners]
                    v0,v1,v2,v3 = (vi(*p) for p in c)
                    if flip:
                        tri_faces.append((v0, v2, v1))
                        tri_faces.append((v0, v3, v2))
                    else:
                        tri_faces.append((v0, v1, v2))
                        tri_faces.append((v0, v2, v3))

    if not tri_faces:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

    # Build vertex array (node index → world coordinate)
    vertex_arr = np.zeros((len(verts), 3))
    for (ix, iy, iz), idx in verts.items():
        vertex_arr[idx] = origin + np.array([ix, iy, iz]) * pitch

    face_arr = np.array(tri_faces, dtype=np.int32)
    return vertex_arr, face_arr


# ─────────────────────────────────────────────────────────────────────────────
# Mesh post-processing
# ─────────────────────────────────────────────────────────────────────────────

def smooth_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    iterations: int = 8,
    lam: float = 0.5,
    mu: float = -0.53,
) -> np.ndarray:
    """
    Taubin mesh smoothing — removes staircase voxel artifacts while
    preserving mesh volume (unlike pure Laplacian which causes shrinkage).

    Alternates a shrinking step (λ > 0) with an un-shrinking step (μ < 0).
    This acts as a low-pass filter that removes high-frequency noise (staircase
    edges) without the volume loss of pure Laplacian smoothing.

    Parameters
    ----------
    iterations : how many Taubin pass-pairs (each pair = shrink + unshrink)
    lam        : positive smoothing factor for the shrink step
    mu         : negative smoothing factor for the unshrink step.
                 Must satisfy mu < -lam to act as a band-pass filter.
                 Default -0.53 (Taubin's recommended value for lam=0.5).
    """
    n = len(vertices)
    if n == 0 or len(faces) == 0:
        return vertices

    # Build sparse adjacency from face edges (COO, then deduplicate)
    edges = np.vstack([
        faces[:, [0, 1]], faces[:, [1, 0]],
        faces[:, [0, 2]], faces[:, [2, 0]],
        faces[:, [1, 2]], faces[:, [2, 1]],
    ])
    r, c = edges[:, 0], edges[:, 1]
    A = csr_matrix((np.ones(len(r), dtype=np.float32), (r, c)), shape=(n, n))

    # Row-normalise → each row sums to 1 (mean of neighbours)
    row_sums = np.array(A.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    D_inv = csr_matrix(
        (1.0 / row_sums, (np.arange(n), np.arange(n))), shape=(n, n)
    )
    L = D_inv @ A  # normalised Laplacian

    verts = vertices.copy()
    for _ in range(iterations):
        # Shrink step (positive λ — standard Laplacian smoothing)
        verts = (1.0 - lam) * verts + lam * (L @ verts)
        # Un-shrink step (negative μ — inflate back to counteract shrinkage)
        verts = (1.0 - mu) * verts + mu * (L @ verts)

    return verts


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def topology_optimize(
    stl_path: str,
    output_stl_path: str,
    volfrac: float = 0.4,
    resolution: int = 20,
    fixed_side: str = "bottom",
    load_side: str = "top",
    load_dir: int = 2,
    load_magnitude: float = -1.0,
    material_key: str = "pla",
    penal: float = 3.0,
    rmin: float = 1.5,
    max_iter: int = 80,
) -> dict:
    """
    Full topology optimization pipeline.

    Returns dict with:
        original_volume_cm3, optimized_volume_cm3, volume_saved_pct,
        original_mass_g, optimized_mass_g, mass_saved_g, mass_saved_pct,
        cost_saved_per_unit_usd, material_key, output_stl_path
    """
    # Set seed for reproducible topology optimization results
    set_seed(get_seed_from_env(default=42))

    mat = MATERIAL_PROPS.get(material_key, MATERIAL_PROPS["pla"])
    density_gcc = mat["density"]        # g/cm³
    cost_per_kg  = mat["cost_per_kg"]

    logger.info(f"Loading STL: {stl_path}")
    triangles = load_stl_triangles(stl_path)

    # Write original mesh as binary STL for the "before" viewer
    # (handles 3MF/ASCII inputs — converts everything to the same binary format)
    # Uses vectorised numpy to avoid slow per-triangle Python loop on large meshes.
    orig_fn = os.path.splitext(os.path.basename(output_stl_path))[0].replace("_opt", "_orig") + ".stl"
    original_stl_path = os.path.join(os.path.dirname(output_stl_path), orig_fn)
    _write_stl_fast(triangles, original_stl_path)

    logger.info(f"Voxelizing at resolution={resolution}")
    grid, origin, pitch = voxelize(triangles, resolution=resolution)

    nx, ny, nz = grid.shape
    voxel_vol_mm3  = pitch ** 3
    voxel_vol_cm3  = voxel_vol_mm3 / 1000.0

    original_voxels  = int(grid.sum())
    if original_voxels == 0:
        raise ValueError(
            "Voxelization produced an empty grid — the STL may be too small, "
            "degenerate, or in an unsupported format. Try a higher resolution."
        )

    original_vol_cm3 = original_voxels * voxel_vol_cm3
    original_mass_g  = original_vol_cm3 * density_gcc

    logger.info(
        f"Grid: {nx}×{ny}×{nz}, {original_voxels} solid voxels, "
        f"pitch={pitch:.2f}mm, vol={original_vol_cm3:.2f}cm³"
    )

    # ── Plate-aware boundary condition adjustment ────────────────────────────
    # For plate-like shapes, fixing/loading the large flat faces (top/bottom)
    # creates uniform compression with no interesting load paths — the optimizer
    # can't create holes. Automatically redirect to edge constraints.
    verts_all = triangles.reshape(-1, 3)
    extents = verts_all.max(axis=0) - verts_all.min(axis=0)
    geo_type = _detect_geometry_type(extents)

    if geo_type == "plate":
        thin_axis = int(np.argmin(extents))  # 0=x, 1=y, 2=z
        # Map thin axis to the face-pair that spans the thin dimension
        thin_faces = {0: ("left", "right"), 1: ("front", "back"), 2: ("bottom", "top")}
        flat_lo, flat_hi = thin_faces[thin_axis]

        # If fixed_side or load_side are on the flat faces, redirect to edges
        adjusted = False
        # Determine the two in-plane axes (the large ones)
        in_plane = [i for i in range(3) if i != thin_axis]
        # Pick the longer in-plane axis for fixed edge, shorter for load edge
        if extents[in_plane[0]] >= extents[in_plane[1]]:
            long_ax, short_ax = in_plane[0], in_plane[1]
        else:
            long_ax, short_ax = in_plane[1], in_plane[0]

        edge_map = {
            0: ("left", "right"),
            1: ("front", "back"),
            2: ("bottom", "top"),
        }

        if fixed_side in (flat_lo, flat_hi):
            old_fixed = fixed_side
            fixed_side = edge_map[long_ax][0]  # fix the low-side of the long edge
            adjusted = True
            logger.info(
                f"Plate: redirected fixed_side '{old_fixed}' → '{fixed_side}' "
                f"(edge constraint for better load paths)"
            )

        if load_side in (flat_lo, flat_hi):
            old_load = load_side
            load_side = edge_map[long_ax][1]  # load the opposite edge
            # Also adjust load direction to be along the long in-plane axis
            load_dir = long_ax
            adjusted = True
            logger.info(
                f"Plate: redirected load_side '{old_load}' → '{load_side}', "
                f"load_dir → {load_dir} (in-plane loading for through-holes)"
            )

        if fixed_side == load_side:
            # Ensure they're not the same after adjustment
            load_side = edge_map[short_ax][1]
            load_dir = short_ax
            logger.info(f"Plate: load_side → '{load_side}' to avoid same-face conflict")

        if adjusted:
            logger.info(
                f"Plate optimization: fixed='{fixed_side}', load='{load_side}', "
                f"dir={load_dir} (in-plane BCs enable through-holes)"
            )

    # Scale filter radius with resolution for consistent results across
    # different grid sizes. A radius of ~1.5 voxels works well for resolution=20;
    # scale proportionally for other resolutions.
    effective_rmin = rmin * (resolution / 20.0)
    effective_rmin = max(1.2, min(effective_rmin, 3.5))  # clamp to reasonable range

    logger.info(f"Running SIMP optimisation (rmin={effective_rmin:.2f})…")
    density = run_simp(
        grid, volfrac=volfrac, penal=penal, rmin=effective_rmin,
        fixed_side=fixed_side, load_side=load_side,
        load_dir=load_dir, load_magnitude=load_magnitude,
        max_iter=max_iter,
    )

    # ── Plate through-hole enforcement ───────────────────────────────────────
    # For plate-like shapes, project densities through the thin axis so that
    # partial-thickness voids become clean through-holes. This averages the
    # density along the thin axis and assigns the mean to every voxel in that
    # column — creating crisp solid/void decisions that go all the way through.
    if geo_type == "plate":
        thin_axis = int(np.argmin(extents))
        logger.info(f"Plate: projecting densities through thin axis {thin_axis} for through-holes")
        # Average density along the thin axis, broadcast back
        mean_along_thin = density.mean(axis=thin_axis, keepdims=True)
        density = np.broadcast_to(mean_along_thin, density.shape).copy()

    # Adaptive threshold: start at 0.5, fall back progressively so that a
    # degraded SIMP solve (e.g. bad BCs) still produces a usable mesh.
    threshold = 0.5
    verts, faces = np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32)
    for t in [0.5, 0.35, 0.2, 0.1]:
        logger.info(f"Extracting surface mesh at threshold={t}…")
        verts, faces = extract_surface(density, t, origin, pitch)
        if len(faces) > 0:
            threshold = t
            if t < 0.5:
                logger.warning(f"SIMP densities low — fell back to threshold={t}")
            break

    if len(faces) == 0:
        raise ValueError(
            "Topology optimisation produced no solid material. "
            "Try increasing 'Material to retain', changing the fixed/load faces, "
            "or using a higher voxel resolution."
        )

    # Taubin smoothing: removes voxel staircase artifacts without volume shrinkage
    logger.info("Smoothing mesh (Taubin)…")
    verts = smooth_mesh(verts, faces, iterations=10, lam=0.5, mu=-0.53)

    # Write optimized STL
    stl_bytes = write_stl_binary(verts, faces)
    with open(output_stl_path, "wb") as f:
        f.write(stl_bytes)

    # Compute metrics (use the same threshold chosen above)
    opt_voxels   = int((density > threshold).sum())
    opt_vol_cm3  = opt_voxels * voxel_vol_cm3
    opt_mass_g   = opt_vol_cm3 * density_gcc

    mass_saved_g   = max(0, original_mass_g - opt_mass_g)
    mass_saved_pct = 100.0 * mass_saved_g / max(original_mass_g, 1e-9)
    vol_saved_pct  = 100.0 * (original_vol_cm3 - opt_vol_cm3) / max(original_vol_cm3, 1e-9)

    # Cost saved (per unit): mass_saved_kg * cost_per_kg
    cost_saved_usd = mass_saved_g / 1000.0 * cost_per_kg

    # FDM print time estimate: ~10g/hr at standard settings
    fdm_speed_g_per_min = 10.0 / 60.0   # g/min
    print_time_saved_min = mass_saved_g / fdm_speed_g_per_min

    logger.info(
        f"Done: {original_mass_g:.1f}g → {opt_mass_g:.1f}g "
        f"(saved {mass_saved_pct:.0f}%)"
    )

    return {
        "original_volume_cm3":   round(float(original_vol_cm3), 2),
        "optimized_volume_cm3":  round(float(opt_vol_cm3), 2),
        "volume_saved_pct":      round(float(vol_saved_pct), 1),
        "original_mass_g":       round(float(original_mass_g), 1),
        "optimized_mass_g":      round(float(opt_mass_g), 1),
        "mass_saved_g":          round(float(mass_saved_g), 1),
        "mass_saved_pct":        round(float(mass_saved_pct), 1),
        "cost_saved_per_unit_usd": round(float(cost_saved_usd), 3),
        "print_time_saved_min":  round(float(print_time_saved_min), 1),
        "material":              mat["name"],
        "material_key":          material_key,
        "volfrac":               volfrac,
        "resolution":            resolution,
        "fixed_side":            fixed_side,
        "load_side":             load_side,
        "output_stl_path":       output_stl_path,
        "original_stl_path":     original_stl_path,
    }
