"""
Simple panorama preview GUI

This standalone GUI accepts an equirectangular (2:1) photo and overlays
the outlines of the nine perspective views used by the panorama_sfm script.

It does NOT modify existing scripts; it reuses the same yaw/pitch pairs and
projection math (spherical mapping, 90° FOV) to visualize where each planar
image samples from on the panorama.

Dependencies: tkinter (bundled), pillow, numpy, scipy (for Rotation), tkinterdnd2 (optional for drag & drop)
Environment file already includes these.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageTk, ImageDraw
import json

try:
    # SciPy is already in environment.yml
    from scipy.spatial.transform import Rotation
except Exception as e:
    raise RuntimeError("scipy is required for rotation math") from e

try:
    # Optional but present in environment.yml for nicer UX
    from tkinterdnd2 import TkinterDnD, DND_FILES
    TK_ROOT = TkinterDnD.Tk
    DND_AVAILABLE = True
except Exception:
    from tkinter import Tk as _Tk
    TK_ROOT = _Tk
    DND_FILES = None
    DND_AVAILABLE = False

import tkinter as tk
from tkinter import filedialog, messagebox


# --- Parameters mirroring panorama_sfm ---

# Built-in yaw/pitch pairs used in panorama_sfm.render_perspective_images()
# Order matters; index 0 is the reference view.
BUILT_IN_PITCH_YAW_PAIRS: List[Tuple[float, float]] = [
    (0.0, 90.0),
    (42.0, 0.0),
    (-42.0, 0.0),
    (0.0, 42.0),
    (0.0, -42.0),
    (42.0, 180.0),
    (-42.0, 180.0),
    (0.0, 222.0),
    (0.0, 138.0),
]

FOV_DEG = 90.0  # matches create_virtual_camera in panorama_sfm


def _compute_virtual_camera_params(pano_height: int, fov_deg: float = FOV_DEG):
    """Return (W, H, fx, fy, cx, cy) for a square SIMPLE_PINHOLE camera.

    This mirrors panorama_sfm.create_virtual_camera without requiring pycolmap.
    image_size = pano_height * (fov/180)
    focal = image_size / (2 * tan(fov/2))
    Principal point at center.
    """
    image_size = int(round(pano_height * (fov_deg / 180.0)))
    if image_size <= 0:
        image_size = max(1, pano_height // 2)
    focal = image_size / (2.0 * np.tan(np.deg2rad(fov_deg) / 2.0))
    W = H = image_size
    cx = (W / 2.0)
    cy = (H / 2.0)
    return W, H, focal, focal, cx, cy


def _rays_in_cam_grid(W: int, H: int, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Compute per-pixel unit rays in the camera frame.

    Equivalent to COLMAP SIMPLE_PINHOLE cam_from_img() + homogeneous to rays.
    Returns array of shape (H*W, 3).
    """
    y, x = np.indices((H, W)).astype(np.float32)
    # Pixel centers
    x = x + 0.5
    y = y + 0.5
    xn = (x - cx) / fx
    yn = (y - cy) / fy
    rays = np.stack([xn, yn, np.ones_like(xn)], axis=-1)
    rays = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays.reshape(-1, 3)


def _spherical_map(pano_size: Tuple[int, int], rays_cam: np.ndarray) -> np.ndarray:
    """Map camera-frame rays to equirectangular pixel coordinates (u,v).

    pano_size = (W, H).
    Mirrors panorama_sfm.spherical_img_from_cam, but vectorized here.
    """
    pano_W, pano_H = pano_size
    if pano_W != 2 * pano_H:
        raise ValueError("Only 360° equirectangular panoramas (2:1) are supported.")
    r = rays_cam.T
    yaw = np.arctan2(r[0], r[2])
    pitch = -np.arctan2(r[1], np.linalg.norm(r[[0, 2]], axis=0))
    u = (1.0 + (yaw / np.pi)) / 2.0
    v = (1.0 - (pitch * 2.0 / np.pi)) / 2.0
    xy = np.stack([u * pano_W, v * pano_H], axis=-1)
    return xy


def _cams_from_pano_rotations(pairs: List[Tuple[float, float]]):
    """Return list of 3x3 rotation matrices for each (pitch, yaw).

    Matches panorama_sfm: Rotation.from_euler("YX", [yaw, pitch]).as_matrix()
    """
    mats = []
    for pitch_deg, yaw_deg in pairs:
        R = Rotation.from_euler("YX", [yaw_deg, pitch_deg], degrees=True).as_matrix()
        mats.append(R)
    return mats


def _wrap_angle_deg(a: float) -> float:
    return ((float(a) + 180.0) % 360.0) - 180.0


def _measure_center_angles(pitch_deg: float, yaw_deg: float) -> Tuple[float, float]:
    """Compute the spherical yaw/pitch (deg) of the view center for a pair.
    Uses a unit center ray [0,0,1] rotated by R and the same mapping.
    Returns (pitch_meas_deg, yaw_meas_deg)
    """
    R = Rotation.from_euler("YX", [yaw_deg, pitch_deg], degrees=True).as_matrix()
    ray_cam = np.array([0.0, 0.0, 1.0], dtype=float)
    r = ray_cam @ R
    yaw = np.degrees(np.arctan2(r[0], r[2]))
    pitch = -np.degrees(np.arctan2(r[1], np.linalg.norm(r[[0, 2]])))
    return float(pitch), float(yaw)


def _outline_for_view(pano_W: int, pano_H: int, R_cam_from_pano: np.ndarray, border_samples: int = 128) -> np.ndarray:
    """Compute polyline points (N,2) on the panorama corresponding to the borders
    of the virtual camera image for a 90° FOV.
    """
    # Virtual camera intrinsics derived from pano height
    W, H, fx, fy, cx, cy = _compute_virtual_camera_params(pano_H, FOV_DEG)

    # Build a 4-edge border: top, right, bottom, left
    # Sample in pixel space [0..W) x [0..H)
    def edge_points():
        ts = np.linspace(0.0, 1.0, border_samples, dtype=np.float32)
        # top: (t, 0)
        for t in ts:
            yield (t * (W - 1), 0.0)
        # right: (W-1, t)
        for t in ts:
            yield ((W - 1), t * (H - 1))
        # bottom: (1 - t, H-1)
        for t in ts:
            yield ((1.0 - t) * (W - 1), (H - 1))
        # left: (0, 1 - t)
        for t in ts:
            yield (0.0, (1.0 - t) * (H - 1))

    # Collect border points as (N,2) float32 array. Using list->array avoids
    # numpy.fromiter tuple packing issues (fromiter expects scalars).
    pts = np.array(list(edge_points()), dtype=np.float32)
    # Convert to unit rays in cam frame
    x = pts[:, 0] + 0.5
    y = pts[:, 1] + 0.5
    xn = (x - cx) / fx
    yn = (y - cy) / fy
    rays_cam = np.stack([xn, yn, np.ones_like(xn)], axis=-1)
    rays_cam /= np.linalg.norm(rays_cam, axis=-1, keepdims=True)

    # Rotate into panorama frame: rays_pano = rays_cam @ R_cam_from_pano
    rays_pano = rays_cam @ R_cam_from_pano

    xy = _spherical_map((pano_W, pano_H), rays_pano)
    return xy.astype(np.float32)


def _unwrap_u_along_seam(pts: np.ndarray, W: int) -> np.ndarray:
    """Unwrap u across the yaw seam so consecutive points have small deltas.
    Returns a copy with u possibly shifted by +/- k*W.
    """
    if len(pts) == 0:
        return pts.copy()
    unwrapped = pts.astype(np.float32).copy()
    prev_u = float(unwrapped[0, 0])
    offset = 0.0
    for i in range(1, len(unwrapped)):
        u = float(unwrapped[i, 0])
        du = u - prev_u
        if du > W * 0.5:
            offset -= W
        elif du < -W * 0.5:
            offset += W
        unwrapped[i, 0] = u + offset
        prev_u = u
    return unwrapped


def _draw_polygon_with_wrap(base_overlay: Image.Image,
                             pts: np.ndarray,
                             color: tuple[int, int, int],
                             fill_a: int,
                             W: int,
                             H: int,
                             target_u: float | None = None) -> None:
    """Draw polygon and outline on base_overlay handling yaw seam wrapping.
    If target_u is provided (0..W), align the polygon's centroid near that u
    in the final image to disambiguate seam placement.
    """
    if len(pts) < 3:
        return
    # Unwrap
    unwrapped = _unwrap_u_along_seam(pts, W)
    min_u = float(np.min(unwrapped[:, 0]))
    max_u = float(np.max(unwrapped[:, 0]))
    cen_u = float(np.mean(unwrapped[:, 0]))

    # Choose shift so the polygon lies inside [W,2W] and its centroid maps near target_u
    if target_u is None:
        base_shift = W - (np.floor(min_u / W) * W)
        s = base_shift
    else:
        # Desired absolute centroid within the crop band
        desired_abs = W + float(target_u)
        s = desired_abs - cen_u
        # Clamp to keep polygon within the crop band
        s = max(s, W - min_u)
        s = min(s, 2 * W - max_u)

    shifted = unwrapped.copy()
    shifted[:, 0] = shifted[:, 0] + s

    # Draw on a 3W-wide canvas and crop center
    bigW = 3 * W
    tmp = Image.new("RGBA", (bigW, H), (0, 0, 0, 0))
    dtmp = ImageDraw.Draw(tmp, "RGBA")
    seq = [tuple(map(float, p)) for p in shifted]
    dtmp.polygon(seq, fill=(color[0], color[1], color[2], fill_a))
    dtmp.line(seq + [seq[0]], fill=(color[0], color[1], color[2], 255), width=2, joint="curve")

    crop = tmp.crop((W, 0, 2 * W, H))
    base_overlay.paste(crop, (0, 0), crop)


class PanoPreviewApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Panorama View Overlay Preview")
        self.root.geometry("1280x800")

        # Top controls
        top = tk.Frame(root)
        top.pack(fill=tk.X, padx=10, pady=8)

        self.open_btn = tk.Button(top, text="Open Panorama...", command=self.open_file)
        self.open_btn.pack(side=tk.LEFT)

        self.show_labels_var = tk.BooleanVar(value=True)
        tk.Checkbutton(top, text="Show labels", variable=self.show_labels_var, command=self._on_params_changed).pack(side=tk.LEFT, padx=10)

        self.show_angles_var = tk.BooleanVar(value=False)
        tk.Checkbutton(top, text="Show yaw/pitch", variable=self.show_angles_var, command=self._on_params_changed).pack(side=tk.LEFT, padx=6)

        self.alpha_var = tk.DoubleVar(value=0.25)
        tk.Label(top, text="Fill alpha:").pack(side=tk.LEFT)
        tk.Scale(top, from_=0.0, to=0.8, resolution=0.05, orient=tk.HORIZONTAL,
                 variable=self.alpha_var,
                 command=lambda _=None: self._on_params_changed(),
                 length=160).pack(side=tk.LEFT, padx=(4, 14))

        # Status label for angle source
        self.angle_src_var = tk.StringVar(value="Angles: built-in (panorama_sfm)")
        tk.Label(top, textvariable=self.angle_src_var, fg="#666").pack(side=tk.LEFT, padx=(12,0))

        # Canvas for image display
        self.canvas = tk.Canvas(root, bg="#111")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas_img_id = None

        # Drag & drop support
        if DND_AVAILABLE:
            try:
                self.root.drop_target_register(DND_FILES)
                self.root.dnd_bind('<<Drop>>', self._on_drop)
            except Exception:
                pass

        # State
        self.src_img_path: Path | None = None
        self.src_img: Image.Image | None = None
        self.display_img: Image.Image | None = None
        self.tk_img: ImageTk.PhotoImage | None = None

        # Active yaw/pitch list and derived rotations
        self.active_pairs: List[Tuple[float, float]] = BUILT_IN_PITCH_YAW_PAIRS[:]
        self.rotations = _cams_from_pano_rotations(self.active_pairs)

        # Overlay caching and debounce
        self._overlay_comp: Image.Image | None = None  # composited panorama+overlay (RGB)
        self._overlay_dirty: bool = True
        self._debounce_job: str | None = None

        # View colors (distinct, readable)
        self.colors = [
            (255, 99, 71),   # tomato
            (30, 144, 255),  # dodger blue
            (60, 179, 113),  # medium sea green
            (238, 130, 238), # violet
            (255, 215, 0),   # gold
            (255, 140, 0),   # dark orange
            (72, 61, 139),   # dark slate blue
            (205, 92, 92),   # indian red
            (70, 130, 180),  # steel blue
        ]

        # Redraw on resize (only rescale cached image)
        self.root.bind("<Configure>", lambda e: self._on_resize())

        # --- Angle controls grid ---
        self.angles_frame = tk.LabelFrame(root, text="Angles (pitch, yaw) — 9 views", padx=8, pady=6)
        self.angles_frame.pack(fill=tk.X, padx=10, pady=8)
        self.pitch_vars: list[tk.StringVar] = []
        self.yaw_vars: list[tk.StringVar] = []
        header = tk.Frame(self.angles_frame)
        header.pack(fill=tk.X, pady=(0,6))
        tk.Label(header, text="#", width=2).grid(row=0, column=0)
        tk.Label(header, text="Pitch", width=10).grid(row=0, column=1)
        tk.Label(header, text="Yaw", width=10).grid(row=0, column=2)
        grid = tk.Frame(self.angles_frame)
        grid.pack(fill=tk.X)
        for i in range(9):
            tk.Label(grid, text=str(i), width=2).grid(row=i, column=0, sticky="w")
            pv = tk.StringVar(value=str(BUILT_IN_PITCH_YAW_PAIRS[i][0]))
            yv = tk.StringVar(value=str(BUILT_IN_PITCH_YAW_PAIRS[i][1]))
            self.pitch_vars.append(pv)
            self.yaw_vars.append(yv)
            tk.Entry(grid, textvariable=pv, width=10).grid(row=i, column=1, padx=2)
            tk.Entry(grid, textvariable=yv, width=10).grid(row=i, column=2, padx=2)
        # Controls row
        ctrl = tk.Frame(self.angles_frame)
        ctrl.pack(fill=tk.X, pady=(8,0))
        tk.Button(ctrl, text="Load Built-in", command=self._load_builtin_into_grid).pack(side=tk.LEFT)
        tk.Button(ctrl, text="Load Override (if found)", command=self._load_override_into_grid).pack(side=tk.LEFT, padx=6)
        tk.Button(ctrl, text="Apply To Preview", command=self._apply_grid_to_preview).pack(side=tk.LEFT, padx=6)
        tk.Button(ctrl, text="Mirror Horiz (yaw→−yaw)", command=self._mirror_horiz).pack(side=tk.LEFT, padx=6)
        tk.Button(ctrl, text="Mirror Vert (pitch→−pitch)", command=self._mirror_vert).pack(side=tk.LEFT, padx=6)
        tk.Button(ctrl, text="Rotate 180° (yaw+=180)", command=self._rotate_180).pack(side=tk.LEFT, padx=6)
        tk.Button(ctrl, text="Right→Left only (auto)", command=self._right_to_left_only).pack(side=tk.LEFT, padx=6)
        tk.Button(ctrl, text="Seam→Left (±180→−179)", command=self._seam_to_left).pack(side=tk.LEFT, padx=6)
        self.save_btn = tk.Button(ctrl, text="Save override.json next to image", command=self._save_override_json, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=6)

    # --- File handling ---
    def _on_drop(self, event):
        try:
            # event.data could be a list-like string; pick the first path
            path = event.data.strip().strip('{}')
            p = Path(path)
            if p.is_file():
                self.load_image(p)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dropped file:\n{e}")

    def open_file(self):
        p = filedialog.askopenfilename(title="Open equirectangular image",
                                       filetypes=[
                                           ("Images", "*.jpg;*.jpeg;*.png;*.tif;*.tiff"),
                                           ("All", "*.*"),
                                       ])
        if not p:
            return
        self.load_image(Path(p))

    def _load_rotation_override_if_any(self, image_path: Path) -> tuple[list[tuple[float, float]] | None, int | None, Path | None]:
        """Mimic panorama_sfm.load_rotation_override_if_any search order.
        Search order:
          1) image folder
          2) parent of image folder
          3) current working directory
        Return (pairs, ref_idx, src_path) or (None, None, None)
        """
        img_dir = image_path.parent
        candidates = [
            img_dir / "rotation_override.json",
            img_dir.parent / "rotation_override.json",
            Path.cwd() / "rotation_override.json",
        ]
        for p in candidates:
            try:
                if p.exists():
                    data = json.loads(p.read_text(encoding="utf-8"))
                    pairs = data.get("pitch_yaw_pairs", None)
                    ref_idx = data.get("ref_idx", 0)
                    if isinstance(pairs, list) and len(pairs) > 0:
                        pairs = [(float(a), float(b)) for (a, b) in pairs]
                        return pairs, int(ref_idx), p
            except Exception:
                continue
        return None, None, None

    def load_image(self, path: Path):
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image:\n{e}")
            return

        W, H = img.size
        if W != 2 * H:
            messagebox.showerror("Invalid panorama", f"Image must be 2:1 equirectangular (got {W}x{H}).")
            return

        # Try to load rotation_override.json near the image
        pairs, ref_idx, src = self._load_rotation_override_if_any(path)
        if pairs is not None:
            self.active_pairs = pairs
            self.rotations = _cams_from_pano_rotations(self.active_pairs)
            self.angle_src_var.set(f"Angles: override JSON (ref_idx={ref_idx}) at {src}")
            # Populate grid
            self._fill_grid_from_pairs(self.active_pairs)
        else:
            self.active_pairs = BUILT_IN_PITCH_YAW_PAIRS[:]
            self.rotations = _cams_from_pano_rotations(self.active_pairs)
            self.angle_src_var.set("Angles: built-in (panorama_sfm)")
            self._fill_grid_from_pairs(self.active_pairs)

        self.src_img_path = path
        self.src_img = img
        self._overlay_dirty = True
        if self.save_btn is not None:
            self.save_btn.config(state=tk.NORMAL)
        self.refresh()

    # --- Rendering ---
    def refresh(self):
        if self.src_img is None:
            return

        # Recompute overlay only when dirty; otherwise only rescale
        if self._overlay_dirty or self._overlay_comp is None:
            self._overlay_comp = self._render_overlay()
            self._overlay_dirty = False

        disp = self._fit_to_canvas(self._overlay_comp)
        self.display_img = disp
        self.tk_img = ImageTk.PhotoImage(disp)
        self.canvas.delete("all")
        self.canvas_img_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
        self.canvas.config(scrollregion=(0, 0, disp.width, disp.height))

    def _render_overlay(self) -> Image.Image:
        """Compute the composited panorama + overlay at full image resolution."""
        pano = self.src_img.copy()
        pano_W, pano_H = pano.size
        overlay = Image.new("RGBA", (pano_W, pano_H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")
        alpha = float(self.alpha_var.get())
        fill_a = int(round(max(0.0, min(0.8, alpha)) * 255))

        for idx, R in enumerate(self.rotations):
            color = self.colors[idx % len(self.colors)]
            poly = _outline_for_view(pano_W, pano_H, R, border_samples=180)
            pts = np.asarray(poly, dtype=np.float32)
            # Target center based on measured yaw; mimic cv2.remap's -0.5 pixel origin (wrap)
            p_in, y_in = self.active_pairs[idx]
            p_m, y_m = _measure_center_angles(p_in, y_in)
            u_center = ((1.0 + (y_m / 180.0)) / 2.0 * pano_W - 0.5) % pano_W
            _draw_polygon_with_wrap(overlay, pts, color, fill_a, pano_W, pano_H, target_u=u_center)
            if self.show_labels_var.get():
                # Compute v with same -0.5 shift and clamp
                v_center = ((1.0 - (np.deg2rad(p_m) * 2.0 / np.pi)) / 2.0 * pano_H) - 0.5
                v_center = float(min(max(v_center, 0.0), pano_H - 1.0))
                if getattr(self, 'show_angles_var', None) is not None and self.show_angles_var.get():
                    label = f"{idx}  p={p_m:.1f}  y={y_m:.1f}"
                else:
                    label = f"{idx}"
                sz = 12
                pad = 3
                tw = draw.textlength(label)
                th = sz
                bx0, by0 = u_center - tw/2 - pad, v_center - th/2 - pad
                bx1, by1 = u_center + tw/2 + pad, v_center + th/2 + pad
                draw.rectangle([bx0, by0, bx1, by1], fill=(0, 0, 0, 128))
                draw.text((u_center - tw/2, v_center - th/2), label, fill=(255, 255, 255, 255))

        comp = Image.alpha_composite(pano.convert("RGBA"), overlay)
        return comp.convert("RGB")

    def _on_params_changed(self):
        # Mark overlay dirty and debounce recompute
        self._overlay_dirty = True
        self._debounced_refresh()

    def _debounced_refresh(self, delay_ms: int = 150):
        if self._debounce_job is not None:
            try:
                self.root.after_cancel(self._debounce_job)
            except Exception:
                pass
            self._debounce_job = None
        self._debounce_job = self.root.after(delay_ms, self.refresh)

    def _fit_to_canvas(self, img: Image.Image) -> Image.Image:
        cw = max(100, self.canvas.winfo_width())
        ch = max(100, self.canvas.winfo_height())
        if cw <= 1 or ch <= 1:
            cw, ch = 1280, 640
        iw, ih = img.size
        scale = min(cw / iw, ch / ih)
        nw = max(1, int(iw * scale))
        nh = max(1, int(ih * scale))
        return img.resize((nw, nh), Image.LANCZOS)

    def _on_resize(self):
        # Only rescale cached image on resize
        if self._overlay_comp is not None:
            disp = self._fit_to_canvas(self._overlay_comp)
            self.display_img = disp
            self.tk_img = ImageTk.PhotoImage(disp)
            self.canvas.delete("all")
            self.canvas_img_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
            self.canvas.config(scrollregion=(0, 0, disp.width, disp.height))

    # --- Angle grid helpers ---
    def _fill_grid_from_pairs(self, pairs: List[Tuple[float, float]]):
        n = min(9, len(pairs))
        for i in range(n):
            p, y = pairs[i]
            self.pitch_vars[i].set(str(float(p)))
            self.yaw_vars[i].set(str(float(y)))

    def _pairs_from_grid(self) -> List[Tuple[float, float]]:
        out: List[Tuple[float, float]] = []
        for i in range(9):
            try:
                p = float(self.pitch_vars[i].get())
                y = float(self.yaw_vars[i].get())
            except Exception:
                p, y = 0.0, 0.0
            out.append((p, y))
        return out

    def _load_builtin_into_grid(self):
        self._fill_grid_from_pairs(BUILT_IN_PITCH_YAW_PAIRS)

    def _load_override_into_grid(self):
        if self.src_img_path is None:
            messagebox.showinfo("No image", "Open an image first to locate override.json.")
            return
        pairs, ref_idx, src = self._load_rotation_override_if_any(self.src_img_path)
        if pairs is None:
            messagebox.showinfo("Not found", "No rotation_override.json found next to image, parent, or CWD.")
            return
        self._fill_grid_from_pairs(pairs)

    def _apply_grid_to_preview(self):
        self.active_pairs = self._pairs_from_grid()
        self.rotations = _cams_from_pano_rotations(self.active_pairs)
        self._overlay_dirty = True
        self.refresh()

    def _mirror_horiz(self):
        # yaw -> -yaw, wrap to [-180, 180]
        pairs = self._pairs_from_grid()
        mirrored = [( _wrap_angle_deg(p), _wrap_angle_deg(-y) ) for (p, y) in pairs]
        self._fill_grid_from_pairs(mirrored)
        self._apply_grid_to_preview()

    def _mirror_vert(self):
        # pitch -> -pitch, wrap to [-180, 180]
        pairs = self._pairs_from_grid()
        mirrored = [( _wrap_angle_deg(-p), _wrap_angle_deg(y) ) for (p, y) in pairs]
        self._fill_grid_from_pairs(mirrored)
        self._apply_grid_to_preview()

    def _rotate_180(self):
        # yaw -> yaw + 180 (wrap), pitch unchanged
        pairs = self._pairs_from_grid()
        rotated = [( _wrap_angle_deg(p), _wrap_angle_deg(y + 180.0) ) for (p, y) in pairs]
        self._fill_grid_from_pairs(rotated)
        self._apply_grid_to_preview()

    def _right_to_left_only(self):
        # Flip only views currently on the right half (by measured yaw)
        pairs = self._pairs_from_grid()
        updated: list[tuple[float, float]] = []
        for (p, y) in pairs:
            # Measure current center yaw
            _, y_meas = _measure_center_angles(p, y)
            if y_meas > 0:
                # For seam (±180) choose +179 to land left after sign flip
                if abs(abs(_wrap_angle_deg(y)) - 180.0) < 1.0:
                    y_new = 179.0
                else:
                    y_new = _wrap_angle_deg(-y)
                updated.append((_wrap_angle_deg(p), y_new))
            else:
                updated.append((_wrap_angle_deg(p), _wrap_angle_deg(y)))
        self._fill_grid_from_pairs(updated)
        self._apply_grid_to_preview()

    def _seam_to_left(self):
        # Move any yaw near ±180 to -179 to visually place at left edge
        pairs = self._pairs_from_grid()
        updated: list[tuple[float, float]] = []
        for (p, y) in pairs:
            if abs(abs(_wrap_angle_deg(y)) - 180.0) < 1.0:
                # Use +179 so measured center ends up at left (sign inversion)
                updated.append((_wrap_angle_deg(p), 179.0))
            else:
                updated.append((_wrap_angle_deg(p), _wrap_angle_deg(y)))
        self._fill_grid_from_pairs(updated)
        self._apply_grid_to_preview()

    def _save_override_json(self):
        if self.src_img_path is None:
            messagebox.showinfo("No image", "Open an image first.")
            return
        pairs = self._pairs_from_grid()
        payload = {"pitch_yaw_pairs": pairs, "ref_idx": 0}
        out_path = self.src_img_path.parent / "rotation_override.json"
        try:
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self.angle_src_var.set(f"Angles: override JSON (ref_idx=0) at {out_path}")
            messagebox.showinfo("Saved", f"Wrote {out_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save JSON:\n{e}")


def main():
    root = TK_ROOT()
    app = PanoPreviewApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
