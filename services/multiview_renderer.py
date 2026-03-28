"""
Multi-view thumbnail renderer for generated CAD parts.

Generates front/side/top/isometric PNG thumbnails from STL files
using a lightweight software rasterizer (no GPU required).

Inspired by GenCAD's multi-view rendering — but parallelized across
views for better performance.
"""

import logging
import math
import os
import struct
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ViewConfig:
    """Camera configuration for a single view."""
    name: str
    azimuth_deg: float      # horizontal rotation
    elevation_deg: float    # vertical rotation
    label: str


# Standard engineering views + isometric
STANDARD_VIEWS = [
    ViewConfig("front",     0,    0,  "Front"),
    ViewConfig("right",    90,    0,  "Right"),
    ViewConfig("top",       0,   90,  "Top"),
    ViewConfig("iso",      45,   35,  "Isometric"),
]


def _load_stl_verts(stl_path: str) -> np.ndarray:
    """Load STL file and return (N, 3, 3) triangle vertices."""
    with open(stl_path, "rb") as f:
        data = f.read()

    if len(data) < 84:
        raise ValueError("STL file too small")

    n_tris = struct.unpack("<I", data[80:84])[0]
    expected = 84 + n_tris * 50

    if len(data) >= expected and n_tris > 0:
        tris = np.zeros((n_tris, 3, 3), dtype=np.float32)
        offset = 84
        for i in range(n_tris):
            offset += 12  # skip normal
            for j in range(3):
                tris[i, j] = struct.unpack_from("<fff", data, offset)
                offset += 12
            offset += 2  # skip attr
        return tris

    raise ValueError("Could not parse STL for rendering")


def _rotation_matrix(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Build a 3x3 rotation matrix from azimuth and elevation angles."""
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)

    # Rotate around Y (azimuth), then X (elevation)
    cos_az, sin_az = math.cos(az), math.sin(az)
    cos_el, sin_el = math.cos(el), math.sin(el)

    Ry = np.array([
        [cos_az, 0, sin_az],
        [0,      1, 0     ],
        [-sin_az, 0, cos_az],
    ], dtype=np.float32)

    Rx = np.array([
        [1, 0,      0     ],
        [0, cos_el, -sin_el],
        [0, sin_el,  cos_el],
    ], dtype=np.float32)

    return Rx @ Ry


def _render_view_to_buffer(
    triangles: np.ndarray,
    view: ViewConfig,
    width: int = 256,
    height: int = 256,
) -> np.ndarray:
    """
    Software-rasterize a single view of the mesh into an RGB buffer.

    Uses a simple depth-buffer approach with flat shading.
    Returns (height, width, 3) uint8 array.
    """
    # Center and normalize the mesh
    verts = triangles.reshape(-1, 3)
    center = (verts.max(axis=0) + verts.min(axis=0)) / 2
    extent = (verts.max(axis=0) - verts.min(axis=0)).max()
    if extent < 1e-6:
        extent = 1.0

    normalized = (verts - center) / extent  # fits in [-0.5, 0.5]

    # Apply rotation
    R = _rotation_matrix(view.azimuth_deg, view.elevation_deg)
    rotated = (R @ normalized.T).T

    # Project to 2D (orthographic)
    scale = width * 0.8
    cx, cy = width / 2, height / 2

    screen_x = (rotated[:, 0] * scale + cx).astype(np.float32)
    screen_y = (-rotated[:, 1] * scale + cy).astype(np.float32)  # flip Y
    screen_z = rotated[:, 2]

    # Reshape back to triangles
    sx = screen_x.reshape(-1, 3)
    sy = screen_y.reshape(-1, 3)
    sz = screen_z.reshape(-1, 3)

    # Initialize buffers
    image = np.full((height, width, 3), 240, dtype=np.uint8)  # light gray bg
    zbuf = np.full((height, width), -np.inf, dtype=np.float32)

    # Simple flat shading: compute face normals in camera space
    tri_rotated = rotated.reshape(-1, 3, 3)
    e1 = tri_rotated[:, 1] - tri_rotated[:, 0]
    e2 = tri_rotated[:, 2] - tri_rotated[:, 0]
    normals = np.cross(e1, e2)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    normals = normals / norms

    # Light direction (from camera-upper-right)
    light = np.array([0.3, 0.5, 0.8], dtype=np.float32)
    light /= np.linalg.norm(light)

    # Compute per-face brightness
    dots = np.abs(np.sum(normals * light, axis=1))
    brightness = (0.3 + 0.7 * dots)  # ambient + diffuse

    # Rasterize each triangle (simple scanline)
    n_tris = len(sx)
    for i in range(n_tris):
        _rasterize_triangle(
            image, zbuf,
            sx[i], sy[i], sz[i],
            brightness[i],
            width, height,
        )

    return image


def _rasterize_triangle(
    image: np.ndarray,
    zbuf: np.ndarray,
    sx: np.ndarray,    # 3 screen x coords
    sy: np.ndarray,    # 3 screen y coords
    sz: np.ndarray,    # 3 screen z coords (depth)
    brightness: float,
    width: int,
    height: int,
) -> None:
    """Rasterize a single triangle with z-buffering and flat shading."""
    # Bounding box
    min_x = max(0, int(np.floor(sx.min())))
    max_x = min(width - 1, int(np.ceil(sx.max())))
    min_y = max(0, int(np.floor(sy.min())))
    max_y = min(height - 1, int(np.ceil(sy.max())))

    if min_x > max_x or min_y > max_y:
        return

    # Barycentric coordinate setup
    x0, y0 = sx[0], sy[0]
    x1, y1 = sx[1], sy[1]
    x2, y2 = sx[2], sy[2]

    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(denom) < 1e-10:
        return

    inv_denom = 1.0 / denom

    # Steel-blue color
    base_r, base_g, base_b = 100, 140, 180

    for py in range(min_y, max_y + 1):
        for px in range(min_x, max_x + 1):
            w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) * inv_denom
            w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) * inv_denom
            w2 = 1.0 - w0 - w1

            if w0 >= 0 and w1 >= 0 and w2 >= 0:
                z = w0 * sz[0] + w1 * sz[1] + w2 * sz[2]
                if z > zbuf[py, px]:
                    zbuf[py, px] = z
                    b = min(1.0, brightness)
                    image[py, px, 0] = int(base_r * b)
                    image[py, px, 1] = int(base_g * b)
                    image[py, px, 2] = int(base_b * b)


def _write_ppm(image: np.ndarray, path: str) -> None:
    """Write an RGB image as PPM (portable pixmap) — no PIL dependency."""
    h, w, _ = image.shape
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode())
        f.write(image.tobytes())


def _render_single_view(args: tuple) -> tuple[str, str]:
    """Worker function for parallel rendering. Returns (view_name, output_path)."""
    stl_path, view_dict, output_dir, file_id, width, height = args
    view = ViewConfig(**view_dict)

    triangles = _load_stl_verts(stl_path)
    image = _render_view_to_buffer(triangles, view, width, height)

    out_path = os.path.join(output_dir, f"{file_id}_{view.name}.ppm")
    _write_ppm(image, out_path)

    return view.name, out_path


def render_multiview(
    stl_path: str,
    output_dir: str,
    file_id: str,
    views: Optional[list[ViewConfig]] = None,
    width: int = 256,
    height: int = 256,
    max_workers: int = 4,
) -> dict[str, str]:
    """
    Render multiple views of an STL file in parallel.

    Args:
        stl_path: Path to the STL file.
        output_dir: Directory to write thumbnail images.
        file_id: Unique identifier for output filenames.
        views: List of ViewConfig objects. Defaults to standard engineering views.
        width: Image width in pixels.
        height: Image height in pixels.
        max_workers: Max parallel rendering processes.

    Returns:
        Dict mapping view names to output file paths.
        e.g. {"front": "/path/to/abc123_front.ppm", ...}
    """
    if views is None:
        views = STANDARD_VIEWS

    os.makedirs(output_dir, exist_ok=True)

    # Prepare args for parallel execution
    tasks = [
        (stl_path, {
            "name": v.name,
            "azimuth_deg": v.azimuth_deg,
            "elevation_deg": v.elevation_deg,
            "label": v.label,
        }, output_dir, file_id, width, height)
        for v in views
    ]

    results: dict[str, str] = {}

    # Render views in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=min(max_workers, len(tasks))) as pool:
        futures = {pool.submit(_render_single_view, t): t for t in tasks}
        for future in as_completed(futures):
            try:
                view_name, out_path = future.result()
                results[view_name] = out_path
            except Exception as e:
                task = futures[future]
                logger.warning(f"Failed to render view {task[1]['name']}: {e}")

    logger.info(f"Rendered {len(results)}/{len(views)} views for {file_id}")
    return results
