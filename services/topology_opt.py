"""
3D Topology Optimization Service — SIMP method.

Pipeline:
  STL/3MF → voxelize (numpy ray-casting) → SIMP FEA loop (scipy.sparse)
           → threshold density → surface extraction → binary STL
           → cost/weight savings using material database.

Performance notes:
  - DOF assembly and filter weights are fully vectorised (no Python element loops)
  - FEA solver uses float64 for numerical stability (sparse linear solve)
  - float32 for density grids (halves memory vs float64)
  - progress_callback wired through for real-time SSE streaming
"""

import logging
import math
import os
import struct
import xml.etree.ElementTree as ET
import zipfile
from typing import Callable, Optional

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

from services.seed_manager import get_seed_from_env, set_seed

logger = logging.getLogger(__name__)

# ── Material database (FDM / additive) ───────────────────────────────────────
MATERIAL_PROPS = {
    "pla":      {"density": 1.24, "cost_per_kg": 25.0,  "name": "PLA"},
    "abs":      {"density": 1.05, "cost_per_kg": 22.0,  "name": "ABS"},
    "petg":     {"density": 1.27, "cost_per_kg": 28.0,  "name": "PETG"},
    "nylon":    {"density": 1.14, "cost_per_kg": 45.0,  "name": "Nylon"},
    "al6061":   {"density": 2.70, "cost_per_kg": 5.50,  "name": "Al 6061-T6"},
    "steel":    {"density": 7.85, "cost_per_kg": 1.20,  "name": "Steel"},
    "titanium": {"density": 4.43, "cost_per_kg": 80.0,  "name": "Titanium"},
}


# ─────────────────────────────────────────────────────────────────────────────
# STL / 3MF I/O
# ─────────────────────────────────────────────────────────────────────────────

def _load_3mf_triangles(path: str) -> np.ndarray:
    """Load a 3MF file → (N, 3, 3) float32 triangle array."""
    NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
    all_tris = []
    with zipfile.ZipFile(path, "r") as zf:
        model_names = [n for n in zf.namelist() if n.endswith(".model")]
        if not model_names:
            raise ValueError("3MF file contains no .model file")
        for model_name in model_names:
            root = ET.fromstring(zf.read(model_name))
            for mesh in root.iter(f"{{{NS}}}mesh"):
                verts_el = mesh.find(f"{{{NS}}}vertices")
                tris_el  = mesh.find(f"{{{NS}}}triangles")
                if verts_el is None or tris_el is None:
                    continue
                verts = np.array(
                    [[float(v.get("x", 0)), float(v.get("y", 0)), float(v.get("z", 0))]
                     for v in verts_el.iter(f"{{{NS}}}vertex")],
                    dtype=np.float32,
                )
                if len(verts) == 0:
                    continue
                for tri in tris_el.iter(f"{{{NS}}}triangle"):
                    i0, i1, i2 = int(tri.get("v1", 0)), int(tri.get("v2", 1)), int(tri.get("v3", 2))
                    all_tris.append([verts[i0], verts[i1], verts[i2]])
    if not all_tris:
        raise ValueError("3MF file contains no mesh geometry")
    return np.array(all_tris, dtype=np.float32)


def load_stl_triangles(stl_path: str) -> np.ndarray:
    """Load binary STL, ASCII STL, or 3MF → (N, 3, 3) float32 triangle array."""
    with open(stl_path, "rb") as f:
        raw = f.read()

    if raw[:4] == b"PK\x03\x04":
        return _load_3mf_triangles(stl_path)

    if len(raw) < 84:
        raise ValueError("STL file is too small to be valid")

    n_binary = struct.unpack("<I", raw[80:84])[0]
    expected_size = 84 + n_binary * 50

    if expected_size == len(raw):
        tris = np.zeros((n_binary, 3, 3), dtype=np.float32)
        offset = 84
        for i in range(n_binary):
            offset += 12  # normal
            for j in range(3):
                tris[i, j] = struct.unpack_from("<fff", raw, offset)
                offset += 12
            offset += 2
        return tris

    text = raw.decode("utf-8", errors="replace").lstrip("\ufeff \t\r\n")
    if text.lower().startswith("solid"):
        tris = _load_stl_ascii(text)
        if len(tris) > 0:
            return tris

    if len(raw) >= expected_size and n_binary > 0:
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


def _tris_to_stl_bytes(tris: np.ndarray) -> bytes:
    """Vectorised (F,3,3) float32 → binary STL bytes."""
    n = len(tris)
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    normals = np.cross(v1 - v0, v2 - v0).astype(np.float32)
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    np.maximum(lens, 1e-10, out=lens)
    normals /= lens
    rec = np.zeros(n, dtype=np.dtype([
        ('n',    np.float32, (3,)),
        ('v0',   np.float32, (3,)),
        ('v1',   np.float32, (3,)),
        ('v2',   np.float32, (3,)),
        ('attr', np.uint16),
    ]))
    rec['n'] = normals; rec['v0'] = v0; rec['v1'] = v1; rec['v2'] = v2
    return b'\x00' * 80 + struct.pack('<I', n) + rec.tobytes()


def write_stl_binary(vertices: np.ndarray, faces: np.ndarray) -> bytes:
    tris = vertices[faces].astype(np.float32)
    return _tris_to_stl_bytes(tris)


def _write_stl_fast(tris: np.ndarray, path: str) -> None:
    with open(path, 'wb') as f:
        f.write(_tris_to_stl_bytes(tris))


# ─────────────────────────────────────────────────────────────────────────────
# Voxelization
# ─────────────────────────────────────────────────────────────────────────────

def _ray_z_intersections(triangles: np.ndarray, cx: float, cy: float) -> np.ndarray:
    """Möller–Trumbore: find all Z values where a +Z ray at (cx, cy) hits the mesh."""
    v0, v1, v2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    e1 = v1 - v0
    e2 = v2 - v0
    hx = -e2[:, 1]
    hy =  e2[:, 0]
    a  = e1[:, 0] * hx + e1[:, 1] * hy
    valid = np.abs(a) > 1e-10
    if not valid.any():
        return np.empty(0)
    with np.errstate(divide='ignore', invalid='ignore'):
        f = np.where(valid, 1.0 / a, 0.0)
    sx = cx - v0[:, 0]
    sy = cy - v0[:, 1]
    u = f * (sx * hx + sy * hy)
    mask2 = valid & (u >= -1e-8) & (u <= 1 + 1e-8)
    if not mask2.any():
        return np.empty(0)
    qz = sx * e1[:, 1] - sy * e1[:, 0]
    v_ = f * qz
    mask3 = mask2 & (v_ >= -1e-8) & (u + v_ <= 1 + 1e-8)
    if not mask3.any():
        return np.empty(0)
    z = v0[:, 2][mask3] + e1[:, 2][mask3] * u[mask3] + e2[:, 2][mask3] * v_[mask3]
    return z


def voxelize(triangles: np.ndarray, resolution: int = 20):
    """
    Convert triangle mesh → boolean voxel grid via Z-axis ray casting.
    Returns (grid: (nx,ny,nz) bool, origin: (3,) float, pitch: float mm).
    """
    verts = triangles.reshape(-1, 3)
    mn = verts.min(axis=0)
    mx = verts.max(axis=0)
    extents = np.maximum(mx - mn, 1e-6)
    pitch = extents.max() / resolution
    nx = max(2, int(math.ceil(extents[0] / pitch)))
    ny = max(2, int(math.ceil(extents[1] / pitch)))
    nz = max(2, int(math.ceil(extents[2] / pitch)))
    grid = np.zeros((nx, ny, nz), dtype=bool)
    for ix in range(nx):
        for iy in range(ny):
            cx = mn[0] + (ix + 0.5) * pitch
            cy = mn[1] + (iy + 0.5) * pitch
            z_hits = np.sort(_ray_z_intersections(triangles, cx, cy))
            for k in range(0, len(z_hits) - 1, 2):
                z_lo, z_hi = z_hits[k], z_hits[k + 1]
                iz_lo = max(0, int((z_lo - mn[2]) / pitch))
                iz_hi = min(nz - 1, int((z_hi - mn[2]) / pitch))
                grid[ix, iy, iz_lo: iz_hi + 1] = True
    return grid, mn, pitch


# ─────────────────────────────────────────────────────────────────────────────
# Shell preservation
# ─────────────────────────────────────────────────────────────────────────────

def _shell_mask(grid: np.ndarray, layers: int = 1) -> np.ndarray:
    """
    Return a boolean mask of the outer shell of the solid (1 voxel thick by default).
    Shell voxels are locked at full density so the exterior shape is preserved.
    """
    eroded = grid.copy()
    for _ in range(layers):
        px = np.pad(eroded, 1, constant_values=False)
        eroded = (
            px[0:-2, 1:-1, 1:-1] &
            px[2:,   1:-1, 1:-1] &
            px[1:-1, 0:-2, 1:-1] &
            px[1:-1, 2:,   1:-1] &
            px[1:-1, 1:-1, 0:-2] &
            px[1:-1, 1:-1, 2:  ]
        ) & grid
    return grid & ~eroded


# ─────────────────────────────────────────────────────────────────────────────
# FEM helpers — vectorised
# ─────────────────────────────────────────────────────────────────────────────

def _unit_ke(nu: float = 0.3) -> np.ndarray:
    """24×24 stiffness matrix for a unit hex element with E=1 (2×2×2 Gauss)."""
    from itertools import product as iproduct
    nodes_ref = np.array([
        [-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
        [-1,-1, 1],[1,-1, 1],[1,1, 1],[-1,1, 1],
    ], dtype=float)
    lam = nu / ((1 + nu) * (1 - 2 * nu))
    mu  = 0.5 / (1 + nu)
    D = np.array([
        [lam+2*mu, lam,       lam,       0,  0,  0],
        [lam,      lam+2*mu,  lam,       0,  0,  0],
        [lam,      lam,       lam+2*mu,  0,  0,  0],
        [0,        0,         0,         mu, 0,  0],
        [0,        0,         0,         0,  mu, 0],
        [0,        0,         0,         0,  0,  mu],
    ])
    g = 1.0 / math.sqrt(3)
    KE = np.zeros((24, 24))
    for xi, eta, zeta in iproduct([-g, g], repeat=3):
        dN = np.zeros((3, 8))
        for i, (a, b, c) in enumerate(nodes_ref):
            dN[0, i] = a * (1 + b * eta)  * (1 + c * zeta) / 8
            dN[1, i] = b * (1 + a * xi)   * (1 + c * zeta) / 8
            dN[2, i] = c * (1 + a * xi)   * (1 + b * eta)  / 8
        J = dN @ nodes_ref
        dN_dx = np.linalg.solve(J, dN)
        B = np.zeros((6, 24))
        for i in range(8):
            B[0, 3*i]   = dN_dx[0, i]
            B[1, 3*i+1] = dN_dx[1, i]
            B[2, 3*i+2] = dN_dx[2, i]
            B[3, 3*i]   = dN_dx[1, i]; B[3, 3*i+1] = dN_dx[0, i]
            B[4, 3*i+1] = dN_dx[2, i]; B[4, 3*i+2] = dN_dx[1, i]
            B[5, 3*i]   = dN_dx[2, i]; B[5, 3*i+2] = dN_dx[0, i]
        KE += B.T @ D @ B * abs(np.linalg.det(J))
    return KE


def _build_all_edofs(nx: int, ny: int, nz: int) -> np.ndarray:
    """
    Return (nelem, 24) int array of global DOF indices for every element.
    Iteration order: ez=0..nz-1, ey=0..ny-1, ex=0..nx-1 (C-order on transposed grid).
    Fully vectorised — no Python loop over elements.
    """
    # Element coordinate grids in iteration order (ez, ey, ex)
    ez, ey, ex = np.mgrid[0:nz, 0:ny, 0:nx]
    ez = ez.ravel(); ey = ey.ravel(); ex = ex.ravel()

    # Node global index: iz*(ny+1)*(nx+1) + iy*(nx+1) + ix
    W = (ny + 1) * (nx + 1)   # stride for iz
    S = (nx + 1)               # stride for iy

    # 8 corner nodes per hex element
    n0 = ez    * W + ey    * S + ex
    n1 = ez    * W + ey    * S + (ex+1)
    n2 = ez    * W + (ey+1)* S + (ex+1)
    n3 = ez    * W + (ey+1)* S + ex
    n4 = (ez+1)* W + ey    * S + ex
    n5 = (ez+1)* W + ey    * S + (ex+1)
    n6 = (ez+1)* W + (ey+1)* S + (ex+1)
    n7 = (ez+1)* W + (ey+1)* S + ex

    nodes = np.stack([n0, n1, n2, n3, n4, n5, n6, n7], axis=1)  # (nelem, 8)
    # Each node contributes 3 DOFs: 3n, 3n+1, 3n+2
    dofs = np.stack([3*nodes, 3*nodes+1, 3*nodes+2], axis=2)    # (nelem, 8, 3)
    return dofs.reshape(len(ez), 24)                              # (nelem, 24)


def _boundary_dofs_vec(side: str, nx: int, ny: int, nz: int) -> np.ndarray:
    """Return global DOF indices for all nodes on the specified face (vectorised)."""
    W = (ny + 1) * (nx + 1)
    S = (nx + 1)

    if side == "bottom":
        iy, ix = np.mgrid[0:ny+1, 0:nx+1]
        ni = 0 * W + iy.ravel() * S + ix.ravel()
    elif side == "top":
        iy, ix = np.mgrid[0:ny+1, 0:nx+1]
        ni = nz * W + iy.ravel() * S + ix.ravel()
    elif side == "left":
        iz, iy = np.mgrid[0:nz+1, 0:ny+1]
        ni = iz.ravel() * W + iy.ravel() * S + 0
    elif side == "right":
        iz, iy = np.mgrid[0:nz+1, 0:ny+1]
        ni = iz.ravel() * W + iy.ravel() * S + nx
    elif side == "front":
        iz, ix = np.mgrid[0:nz+1, 0:nx+1]
        ni = iz.ravel() * W + 0 * S + ix.ravel()
    elif side == "back":
        iz, ix = np.mgrid[0:nz+1, 0:nx+1]
        ni = iz.ravel() * W + ny * S + ix.ravel()
    else:
        return np.empty(0, dtype=int)

    return np.unique(np.stack([3*ni, 3*ni+1, 3*ni+2]).ravel())


def _load_vector_vec(
    side: str, direction: int, magnitude: float,
    nx: int, ny: int, nz: int, ndof: int,
) -> np.ndarray:
    """Build distributed load vector on a face (vectorised)."""
    W = (ny + 1) * (nx + 1)
    S = (nx + 1)
    F = np.zeros(ndof)

    if side == "top":
        iy, ix = np.mgrid[0:ny+1, 0:nx+1]
        ni = nz * W + iy.ravel() * S + ix.ravel()
    elif side == "bottom":
        iy, ix = np.mgrid[0:ny+1, 0:nx+1]
        ni = 0 * W + iy.ravel() * S + ix.ravel()
    elif side == "left":
        iz, iy = np.mgrid[0:nz+1, 0:ny+1]
        ni = iz.ravel() * W + iy.ravel() * S + 0
    elif side == "right":
        iz, iy = np.mgrid[0:nz+1, 0:ny+1]
        ni = iz.ravel() * W + iy.ravel() * S + nx
    elif side == "front":
        iz, ix = np.mgrid[0:nz+1, 0:nx+1]
        ni = iz.ravel() * W + 0 * S + ix.ravel()
    elif side == "back":
        iz, ix = np.mgrid[0:nz+1, 0:nx+1]
        ni = iz.ravel() * W + ny * S + ix.ravel()
    else:
        return F

    np.add.at(F, 3 * ni + direction, magnitude / len(ni))
    return F


def _filter_weights_vec(nx: int, ny: int, nz: int, rmin: float):
    """
    Build sparse sensitivity filter matrix H and row sums Hs.

    Vectorised: loops over ~(2r+1)^3 offset positions (typically ≤19 for rmin=1.5)
    instead of the old O(nx*ny*nz*(2r+1)^3) pure-Python triple-nested loop.
    """
    nelem = nx * ny * nz
    r = int(math.ceil(rmin))

    # All offsets in the (2r+1)^3 cube
    dz_g, dy_g, dx_g = np.mgrid[-r:r+1, -r:r+1, -r:r+1]
    dist2 = (dx_g**2 + dy_g**2 + dz_g**2).astype(float)
    valid = dist2 <= rmin ** 2
    dz_v = dz_g[valid].ravel()
    dy_v = dy_g[valid].ravel()
    dx_v = dx_g[valid].ravel()
    w_v  = rmin - np.sqrt(dist2[valid].ravel())

    # Element coordinate arrays in iteration order (ez, ey, ex)
    ez_f, ey_f, ex_f = np.mgrid[0:nz, 0:ny, 0:nx]
    ez_f = ez_f.ravel(); ey_f = ey_f.ravel(); ex_f = ex_f.ravel()
    ei_f = ez_f * ny * nx + ey_f * nx + ex_f

    rows_list, cols_list, vals_list = [], [], []
    for k in range(len(dz_v)):
        fz = ez_f + dz_v[k]
        fy = ey_f + dy_v[k]
        fx = ex_f + dx_v[k]
        mask = (fz >= 0) & (fz < nz) & (fy >= 0) & (fy < ny) & (fx >= 0) & (fx < nx)
        fi = fz[mask] * ny * nx + fy[mask] * nx + fx[mask]
        rows_list.append(ei_f[mask])
        cols_list.append(fi)
        vals_list.append(np.full(mask.sum(), w_v[k]))

    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    vals = np.concatenate(vals_list)
    H  = csr_matrix((vals, (rows, cols)), shape=(nelem, nelem))
    Hs = np.array(H.sum(axis=1)).flatten()
    return H, Hs


def _heaviside_projection(x: np.ndarray, beta: float, eta: float = 0.5) -> np.ndarray:
    num = np.tanh(beta * eta) + np.tanh(beta * (x - eta))
    den = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    return num / den


def _heaviside_derivative(x: np.ndarray, beta: float, eta: float = 0.5) -> np.ndarray:
    den = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    cosh_val = np.minimum(np.cosh(beta * (x - eta)), 1e10)
    return beta / (cosh_val ** 2 * den)


def _oc_update(
    x: np.ndarray, dc: np.ndarray, dv: np.ndarray,
    volfrac: float, active_type: np.ndarray,
    x_min: float = 1e-3, x_max: float = 1.0, move: float = 0.2,
) -> np.ndarray:
    """OC density update. Only interior (active_type==1) voxels move."""
    l1, l2 = 0.0, 1e9
    xnew = x.copy()
    interior = (active_type == 1)
    while (l2 - l1) / (l1 + l2 + 1e-40) > 1e-4:
        lmid = 0.5 * (l1 + l2)
        B = np.sqrt(np.maximum(0, -dc / (dv * lmid)))
        xnew = np.clip(x * B, np.maximum(x_min, x - move), np.minimum(x_max, x + move))
        xnew[active_type == 0] = x_min
        xnew[active_type == 2] = 1.0
        if interior.any() and xnew[interior].mean() > volfrac:
            l1 = lmid
        else:
            l2 = lmid
    return xnew


# ─────────────────────────────────────────────────────────────────────────────
# SIMP solver
# ─────────────────────────────────────────────────────────────────────────────

def run_simp(
    grid: np.ndarray,
    volfrac: float = 0.4,
    penal: float = 3.0,
    rmin: float = 1.4,
    fixed_side: str = "bottom",
    load_side: str = "top",
    load_dir: int = 2,
    load_magnitude: float = -1.0,
    max_iter: int = 80,
    E0: float = 1.0,
    Emin: float = 1e-9,
    shell: Optional[np.ndarray] = None,
    progress_callback: Optional[Callable[[int, int, float, float], None]] = None,
) -> np.ndarray:
    """
    3-D SIMP topology optimisation with:
      - Shell preservation (outer voxels locked at full density)
      - Continuation penalisation schedule
      - Density filtering + Heaviside projection
      - Real-time progress via optional callback(iteration, max_iter, obj, vol_frac)

    Returns density (nx, ny, nz) float in [0, 1].
    """
    nx, ny, nz = grid.shape
    nelem = nx * ny * nz
    ndof  = 3 * (nx + 1) * (ny + 1) * (nz + 1)

    KE = _unit_ke()

    logger.info(f"SIMP: assembling DOF maps for {nx}×{ny}×{nz} grid ({nelem} elements)")
    all_edofs = _build_all_edofs(nx, ny, nz)   # (nelem, 24) — vectorised

    # COO indices for sparse K assembly
    rows = np.repeat(all_edofs, 24, axis=1).reshape(-1)
    cols = np.tile(all_edofs, (1, 24)).reshape(-1)
    ke_flat = KE.flatten()

    # ── Active element classification ─────────────────────────────────────────
    # Iteration order is (ez, ey, ex) so transpose grid to (nz, ny, nx) before flatten.
    active_type = grid.transpose(2, 1, 0).reshape(-1).astype(np.int8)
    if shell is not None:
        active_type[shell.transpose(2, 1, 0).reshape(-1)] = 2

    n_shell    = int((active_type == 2).sum())
    n_interior = int((active_type == 1).sum())
    interior_volfrac = float(np.clip(volfrac, 0.05, 0.95))
    logger.info(
        f"Shell: {n_shell} locked voxels, Interior: {n_interior} optimisable, "
        f"interior_volfrac={interior_volfrac:.3f}"
    )

    x = np.full(nelem, interior_volfrac, dtype=np.float32)
    x[active_type == 0] = Emin
    x[active_type == 2] = 1.0

    fixed_dofs = _boundary_dofs_vec(fixed_side, nx, ny, nz)
    free_dofs  = np.setdiff1d(np.arange(ndof), fixed_dofs)
    F = _load_vector_vec(load_side, load_dir, load_magnitude, nx, ny, nz, ndof)

    logger.info("Building filter weights…")
    H, Hs = _filter_weights_vec(nx, ny, nz, rmin)

    penal_start = 1.0
    continuation_iters = int(max_iter * 0.6)
    beta = 1.0
    beta_max = 32.0
    beta_increase_interval = max(8, max_iter // 8)
    prev_obj = float("inf")

    for iteration in range(max_iter):
        # Continuation: ramp penalisation
        if iteration < continuation_iters:
            cur_penal = penal_start + (penal - penal_start) * (iteration / continuation_iters)
        else:
            cur_penal = penal

        if iteration > 0 and iteration % beta_increase_interval == 0 and beta < beta_max:
            beta = min(beta * 2, beta_max)

        # Density filter
        x_filt = np.asarray(H @ x).flatten() / Hs
        x_filt = np.clip(x_filt, Emin, 1.0)
        x_filt[active_type == 2] = 1.0
        x_filt[active_type == 0] = Emin

        # Heaviside projection
        x_phys = _heaviside_projection(x_filt, beta)
        x_phys = np.clip(x_phys, Emin, 1.0).astype(np.float32)
        x_phys[active_type == 2] = 1.0
        x_phys[active_type == 0] = Emin

        xp = Emin + x_phys ** cur_penal * (E0 - Emin)
        xp[active_type == 2] = E0

        # Assemble and solve
        vals = np.outer(xp, ke_flat).reshape(-1)
        K = coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()
        Kff = K[free_dofs][:, free_dofs]
        try:
            Uf = spsolve(Kff, F[free_dofs])
        except Exception as e:
            logger.error(f"SIMP solve failed at iter {iteration}: {e}")
            break

        U = np.zeros(ndof)
        U[free_dofs] = Uf

        # Sensitivities
        ue = U[all_edofs]
        ce = np.einsum("ij,jk,ik->i", ue, KE, ue)
        obj = float(np.sum(xp * ce))

        dc = -cur_penal * x_phys ** (cur_penal - 1) * (E0 - Emin) * ce
        dv = np.ones(nelem)
        dh = _heaviside_derivative(x_filt, beta)
        dc *= dh; dv *= dh
        dc = np.asarray(H.T @ (dc / Hs)).flatten()
        dv = np.asarray(H.T @ (dv / Hs)).flatten()

        x_new = _oc_update(x, dc, dv, interior_volfrac, active_type)

        interior_mask = (active_type == 1)
        change = float(np.max(np.abs(x_new[interior_mask] - x[interior_mask]))) if interior_mask.any() else 0.0
        obj_change = abs(obj - prev_obj) / max(abs(obj), 1e-12)
        x = x_new
        prev_obj = obj

        cur_vol = float(x_phys[interior_mask].mean()) if interior_mask.any() else interior_volfrac
        logger.debug(
            f"  iter {iteration+1:3d}  p={cur_penal:.2f}  β={beta:.0f}  "
            f"obj={obj:.4e}  vol={cur_vol:.3f}  Δx={change:.4f}"
        )

        if progress_callback is not None:
            progress_callback(iteration + 1, max_iter, obj, cur_vol)

        if iteration > continuation_iters and change < 0.005 and obj_change < 1e-4:
            logger.info(f"SIMP converged at iteration {iteration+1}")
            break

    # Final densities
    x_filt = np.asarray(H @ x).flatten() / Hs
    x_filt = np.clip(x_filt, Emin, 1.0)
    x_filt[active_type == 2] = 1.0
    x_filt[active_type == 0] = Emin
    x_phys = _heaviside_projection(x_filt, beta)
    x_phys = np.clip(x_phys, Emin, 1.0).astype(np.float32)
    x_phys[active_type == 2] = 1.0
    x_phys[active_type == 0] = Emin

    # Reshape back to (nx, ny, nz) — was flattened in (nz,ny,nx) order
    return x_phys.reshape(nz, ny, nx).transpose(2, 1, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Surface extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_surface(
    density: np.ndarray,
    threshold: float,
    origin: np.ndarray,
    pitch: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract boundary voxel faces from a thresholded density field.
    Returns (vertices, faces) where faces index into vertices.
    """
    solid = density > threshold
    nx, ny, nz = solid.shape

    verts: dict = {}
    tri_faces = []

    def vi(ix, iy, iz):
        key = (ix, iy, iz)
        if key not in verts:
            verts[key] = len(verts)
        return verts[key]

    neighbours = [
        ((1,0,0),  [(0,0,0),(0,1,0),(0,1,1),(0,0,1)], False),
        ((-1,0,0), [(0,0,0),(0,0,1),(0,1,1),(0,1,0)], True),
        ((0,1,0),  [(0,0,0),(1,0,0),(1,0,1),(0,0,1)], True),
        ((0,-1,0), [(0,0,0),(0,0,1),(1,0,1),(1,0,0)], False),
        ((0,0,1),  [(0,0,0),(1,0,0),(1,1,0),(0,1,0)], True),
        ((0,0,-1), [(0,0,0),(0,1,0),(1,1,0),(1,0,0)], False),
    ]

    for ez in range(nz):
        for ey in range(ny):
            for ex in range(nx):
                if not solid[ex, ey, ez]:
                    continue
                for (dx, dy, dz), corners, flip in neighbours:
                    nx_ = ex + dx; ny_ = ey + dy; nz_ = ez + dz
                    if (0 <= nx_ < nx and 0 <= ny_ < ny and 0 <= nz_ < nz
                            and solid[nx_, ny_, nz_]):
                        continue
                    c = [(ex+cx, ey+cy, ez+cz) for cx, cy, cz in corners]
                    v0, v1, v2, v3 = (vi(*p) for p in c)
                    if flip:
                        tri_faces.append((v0, v2, v1))
                        tri_faces.append((v0, v3, v2))
                    else:
                        tri_faces.append((v0, v1, v2))
                        tri_faces.append((v0, v2, v3))

    if not tri_faces:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

    vertex_arr = np.zeros((len(verts), 3))
    for (ix, iy, iz), idx in verts.items():
        vertex_arr[idx] = origin + np.array([ix, iy, iz]) * pitch

    return vertex_arr, np.array(tri_faces, dtype=np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# Mesh post-processing — Taubin smoothing
# ─────────────────────────────────────────────────────────────────────────────

def smooth_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    iterations: int = 8,
    lam: float = 0.5,
    mu: float = -0.53,
) -> np.ndarray:
    """
    Taubin smoothing — removes voxel staircase artifacts while preserving volume.
    Alternates a shrink step (λ>0) with an un-shrink step (μ<0).
    """
    n = len(vertices)
    if n == 0 or len(faces) == 0:
        return vertices
    edges = np.vstack([
        faces[:, [0, 1]], faces[:, [1, 0]],
        faces[:, [0, 2]], faces[:, [2, 0]],
        faces[:, [1, 2]], faces[:, [2, 1]],
    ])
    r, c = edges[:, 0], edges[:, 1]
    A = csr_matrix((np.ones(len(r), dtype=np.float32), (r, c)), shape=(n, n))
    row_sums = np.array(A.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    D_inv = csr_matrix((1.0 / row_sums, (np.arange(n), np.arange(n))), shape=(n, n))
    L = D_inv @ A
    verts = vertices.copy()
    for _ in range(iterations):
        verts = (1.0 - lam) * verts + lam * (L @ verts)
        verts = (1.0 - mu)  * verts + mu  * (L @ verts)
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
    progress_callback: Optional[Callable[[int, int, float, float], None]] = None,
) -> dict:
    """
    Full topology optimization pipeline.

    progress_callback(iteration, max_iter, objective, vol_frac) — called each SIMP
    iteration for real-time SSE streaming in the router.

    Returns a dict with mass/volume/cost/time savings and output paths.
    """
    set_seed(get_seed_from_env(default=42))

    mat = MATERIAL_PROPS.get(material_key, MATERIAL_PROPS["pla"])
    density_gcc = mat["density"]
    cost_per_kg  = mat["cost_per_kg"]

    logger.info(f"Loading mesh: {stl_path}")
    triangles = load_stl_triangles(stl_path)

    # Save original as binary STL (normalises 3MF/ASCII input for the viewer)
    orig_fn = (
        os.path.splitext(os.path.basename(output_stl_path))[0].replace("_opt", "_orig")
        + ".stl"
    )
    original_stl_path = os.path.join(os.path.dirname(output_stl_path), orig_fn)
    _write_stl_fast(triangles, original_stl_path)

    logger.info(f"Voxelizing at resolution={resolution}")
    grid, origin, pitch = voxelize(triangles, resolution=resolution)

    nx, ny, nz = grid.shape
    voxel_vol_cm3 = (pitch ** 3) / 1000.0

    original_voxels = int(grid.sum())
    if original_voxels == 0:
        raise ValueError(
            "Voxelization produced an empty grid — STL may be too small or degenerate. "
            "Try a higher resolution."
        )

    original_vol_cm3 = original_voxels * voxel_vol_cm3
    original_mass_g  = original_vol_cm3 * density_gcc
    logger.info(
        f"Grid: {nx}×{ny}×{nz}, {original_voxels} solid voxels, "
        f"pitch={pitch:.2f}mm, vol={original_vol_cm3:.2f}cm³"
    )

    # Scale filter radius proportionally to resolution
    effective_rmin = float(np.clip(rmin * (resolution / 20.0), 1.2, 3.5))

    # Outer shell locked at full density — interior only gets optimised
    shell = _shell_mask(grid, layers=1)
    logger.info(
        f"Shell: {int(shell.sum())} locked voxels, "
        f"Interior: {original_voxels - int(shell.sum())} optimisable"
    )

    logger.info(f"Running SIMP (rmin={effective_rmin:.2f}, max_iter={max_iter})…")
    density = run_simp(
        grid,
        volfrac=volfrac,
        penal=penal,
        rmin=effective_rmin,
        fixed_side=fixed_side,
        load_side=load_side,
        load_dir=load_dir,
        load_magnitude=load_magnitude,
        max_iter=max_iter,
        shell=shell,
        progress_callback=progress_callback,
    )

    # Adaptive threshold — fall back if densities are unexpectedly low
    threshold = 0.5
    verts, faces = np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32)
    for t in [0.5, 0.35, 0.2, 0.1]:
        logger.info(f"Extracting surface at threshold={t}…")
        verts, faces = extract_surface(density, t, origin, pitch)
        if len(faces) > 0:
            threshold = t
            if t < 0.5:
                logger.warning(f"SIMP densities low — fell back to threshold={t}")
            break

    if len(faces) == 0:
        raise ValueError(
            "Topology optimisation produced no solid material. "
            "Try increasing material retention, changing fixed/load faces, "
            "or using a higher voxel resolution."
        )

    logger.info("Smoothing mesh (Taubin)…")
    verts = smooth_mesh(verts, faces, iterations=10, lam=0.5, mu=-0.53)

    stl_bytes = write_stl_binary(verts, faces)
    with open(output_stl_path, "wb") as f:
        f.write(stl_bytes)

    opt_voxels  = int((density > threshold).sum())
    opt_vol_cm3 = opt_voxels * voxel_vol_cm3
    opt_mass_g  = opt_vol_cm3 * density_gcc

    mass_saved_g   = max(0.0, original_mass_g - opt_mass_g)
    mass_saved_pct = 100.0 * mass_saved_g / max(original_mass_g, 1e-9)
    vol_saved_pct  = 100.0 * (original_vol_cm3 - opt_vol_cm3) / max(original_vol_cm3, 1e-9)
    cost_saved_usd = mass_saved_g / 1000.0 * cost_per_kg
    print_time_saved_min = mass_saved_g / (10.0 / 60.0)   # ~10 g/hr FDM estimate

    logger.info(
        f"Done: {original_mass_g:.1f}g → {opt_mass_g:.1f}g (saved {mass_saved_pct:.0f}%)"
    )

    return {
        "original_volume_cm3":     round(float(original_vol_cm3), 2),
        "optimized_volume_cm3":    round(float(opt_vol_cm3), 2),
        "volume_saved_pct":        round(float(vol_saved_pct), 1),
        "original_mass_g":         round(float(original_mass_g), 1),
        "optimized_mass_g":        round(float(opt_mass_g), 1),
        "mass_saved_g":            round(float(mass_saved_g), 1),
        "mass_saved_pct":          round(float(mass_saved_pct), 1),
        "cost_saved_per_unit_usd": round(float(cost_saved_usd), 3),
        "print_time_saved_min":    round(float(print_time_saved_min), 1),
        "material":                mat["name"],
        "material_key":            material_key,
        "volfrac":                 volfrac,
        "resolution":              resolution,
        "fixed_side":              fixed_side,
        "load_side":               load_side,
        "output_stl_path":         output_stl_path,
        "original_stl_path":       original_stl_path,
    }
