import os
import sys
import json
import cv2
import time
import threading
import subprocess
from pathlib import Path
from typing import NamedTuple
import math

from tkinter import (
    Label, Entry, StringVar, DoubleVar, Frame, Checkbutton, BooleanVar,
    filedialog, messagebox, Button, Text, Scale, Canvas, END, BOTH, DISABLED, NORMAL
)
from tkinterdnd2 import DND_FILES, TkinterDnD
from tkinter import ttk
from PIL import Image, ImageTk
from time_range import normalize_time_range
from scipy.spatial.transform import Rotation

import numpy as np  # NEW: for mask processing
from typing import List, Tuple, Optional
import shutil
import json as _json

# =========================
# Angle profiles & helpers
# =========================

# With masking: your angles (ref at index 0)
MASKING_PITCH_YAW_PAIRS = [
    (0, 90),   # Reference Pose (ref_idx = 0)
    (34, 0),
    (-42, 0),
    (0, 42),
    (0, -42),
    (42, 180),
    (-34, 180),
    (0, 222),
    (0, 138),
]
MASKING_REF_IDX = 0

# Without masking: your angles (ref at index 0)
NO_MASKING_PITCH_YAW_PAIRS = [
    (0, 90),   # Reference Pose (ref_idx = 0)
    (32, 0),
    (-42, 0),
    (0, 42),  # para la R1 usar 33, dejo el default para hacerlo compatible con la X5  
    (0, -25),
    (42, 180),
    (-32, 180),
    (0, 205),
    (0, 138), # para la R1 subirlo (142) para evitar que salga el hombro, dejo el default para hacerlo compatible con la X5
]
NO_MASKING_REF_IDX = 0

VALID_VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv"}

def write_rotation_override(root_dir: Path, pairs, ref_idx: int) -> Path:
    payload = {"pitch_yaw_pairs": pairs, "ref_idx": ref_idx}
    out_path = root_dir / "rotation_override.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path

def open_in_explorer(path: Path) -> None:
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception:
        pass

def list_videos(video_dir: Path):
    return sorted([p for p in video_dir.glob("*") if p.suffix.lower() in VALID_VIDEO_EXT])

# =========================
# UI helpers (set from thread)
# =========================

_root = None
status_var = None
# progress_main = None
progress_sub = None
log_text = None

use_masking = None
frame_interval = None
start_time_var = None
end_time_var = None
drop_zone = None
browse_btn = None
last_btn = None
split_btn = None
align_btn = None

_last_folder = None
_selected_project = None
_split_result = None

# =========================
# Preview globals
# =========================
preview_time_var = None  # StringVar HH:MM:SS
preview_grid = None      # Frame that holds 3x3 previews
_preview_imgs = []       # keep PhotoImage refs
_yaw_vars: List[DoubleVar] | None = None  # per-view yaw
_pitch_vars: List[DoubleVar] | None = None  # per-view pitch

preview_canvas = None    # Canvas wrapper for scrollbars
hscroll = None           # Horizontal scrollbar
vscroll = None           # Vertical scrollbar

# =========================
# Overlay globals (Overlays tab)
# =========================
overlay_canvas = None
_overlay_img_tk = None
_overlay_items = []  # canvas item ids to clear

# =========================
# Masking (seam) globals
# =========================
mask_canvas = None
_mask_img_tk = None
_seam_points_x: list[int] = []  # panorama pixel X positions, up to 4 (2 rectangles)
_mask_preview_items = []  # canvas items to clear


def _mask_clear_canvas():
    global _mask_preview_items
    if mask_canvas is None:
        return
    for item in _mask_preview_items:
        try:
            mask_canvas.delete(item)
        except Exception:
            pass
    _mask_preview_items = []


def _mask_draw_image(img: Image.Image):
    global _mask_img_tk
    if mask_canvas is None:
        return
    W, H = img.size
    max_w = 700
    disp_w = min(W, max_w)
    disp_h = int(H * (disp_w / W))
    im_disp = img.resize((disp_w, disp_h), Image.LANCZOS) if disp_w != W else img
    _mask_img_tk = ImageTk.PhotoImage(im_disp)
    mask_canvas.config(width=disp_w, height=disp_h)
    _mask_clear_canvas()
    bg_id = mask_canvas.create_image(0, 0, anchor="nw", image=_mask_img_tk)
    _mask_preview_items = [bg_id]


def _mask_clear_overlays_only():
    # Keep background image (first item), remove subsequent overlay items
    global _mask_preview_items
    if mask_canvas is None:
        return
    if not _mask_preview_items:
        return
    # Preserve first (bg) id
    bg_id = _mask_preview_items[0]
    for item in _mask_preview_items[1:]:
        try:
            mask_canvas.delete(item)
        except Exception:
            pass
    _mask_preview_items = [bg_id]


def _mask_draw_points_and_rects(preview_path: Path):
    if mask_canvas is None:
        return
    try:
        im = Image.open(preview_path)
        W, H = im.size
        # Determine displayed size from current PhotoImage when available
        try:
            disp_w = int(_mask_img_tk.width()) if _mask_img_tk is not None else W
            disp_h = int(_mask_img_tk.height()) if _mask_img_tk is not None else H
        except Exception:
            disp_w = mask_canvas.winfo_width() or W
            disp_h = mask_canvas.winfo_height() or H
        scale = float(disp_w) / float(W) if W else 1.0
        _mask_clear_overlays_only()
        # Draw vertical lines for each point
        for x in _seam_points_x:
            cx = int(x * scale)
            line_id = mask_canvas.create_line(cx, 0, cx, disp_h, fill="#ff4444", width=2)
            _mask_preview_items.append(line_id)
        # Draw rectangle outline for each completed pair
        for i in range(0, len(_seam_points_x) // 2):
            x0 = _seam_points_x[2 * i]
            x1 = _seam_points_x[2 * i + 1]
            lo, hi = (x0, x1) if x0 <= x1 else (x1, x0)
            rx0 = int(lo * scale)
            rx1 = int(hi * scale)
            rect_id = mask_canvas.create_rectangle(rx0, 0, rx1, disp_h, outline="#ff4444", width=1, dash=(4, 2))
            _mask_preview_items.append(rect_id)
    except Exception:
        pass


def _on_mask_canvas_click(event):
    # Map click x to panorama x using current displayed image size
    try:
        if _selected_project is None:
            ui_log("[ERROR] Please select a folder first.")
            return
        preview_path = _ensure_preview_pano_frame(Path(_selected_project))
        if preview_path is None:
            return
        im = Image.open(preview_path)
        W, H = im.size
        # Determine displayed image width from current PhotoImage
        if _mask_img_tk is None:
            _mask_draw_image(im)
        try:
            disp_w = int(_mask_img_tk.width()) if _mask_img_tk is not None else min(W, 700)
        except Exception:
            disp_w = min(W, 700)
        # Clamp click within displayed image bounds and map to panorama x
        x_disp = max(0, min(int(event.x), int(disp_w) - 1))
        scale = float(W) / float(disp_w) if disp_w else 1.0
        x_pano = int(x_disp * scale)
        if len(_seam_points_x) >= 4:
            ui_log("[WARN] Already have 4 points (2 rectangles). Use Clear to remove.")
            return
        x_pano = max(0, min(W - 1, x_pano))
        # Enforce left->right order within each pair; if second < first, swap
        if len(_seam_points_x) % 2 == 1:
            prev = _seam_points_x[-1]
            if x_pano < prev:
                _seam_points_x[-1], x_pano = x_pano, prev
        _seam_points_x.append(x_pano)
        ui_status(f"Seam points: {_seam_points_x}")
        # Ensure base image is drawn and overlay indicators are updated
        _mask_draw_image(im)
        _mask_draw_points_and_rects(preview_path)
    except Exception as e:
        ui_log(f"[WARN] Failed to record point: {e}")


def _on_clear_last_seam_point():
    if _seam_points_x:
        _seam_points_x.pop()
        ui_status(f"Seam points: {_seam_points_x}")
    else:
        ui_status("No points to clear.")
    # Redraw markers over current preview if available
    try:
        if _selected_project is not None:
            preview_path = _ensure_preview_pano_frame(Path(_selected_project))
            if preview_path is not None:
                _mask_draw_image(Image.open(preview_path))
                _mask_draw_points_and_rects(preview_path)
    except Exception:
        pass


def _on_preview_seam_mask():
    if _selected_project is None:
        ui_log("[ERROR] Please select a folder first.")
        return
    frame_file = _ensure_preview_pano_frame(Path(_selected_project))
    if frame_file is None:
        return
    try:
        im = Image.open(frame_file).convert("RGB")
        W, H = im.size
        # Save binary panorama seam mask for visibility in preview folder
        mask = _build_pano_seam_mask(W, H, _seam_points_x)
        out_bin = Path(_selected_project) / "preview" / "pano_seam_mask.png"
        out_bin.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask).save(out_bin)

        # Build visual overlay (semi-transparent black regions)
        overlay = im.copy()
        import numpy as _np
        arr = _np.array(overlay, dtype=_np.uint8)
        dark = arr.copy()
        dark[..., :] = (dark * 0.4).astype(_np.uint8)
        # Expand mask to 3 channels
        m3 = _np.stack([mask == 0] * 3, axis=-1)
        arr[m3] = dark[m3]
        vis = Image.fromarray(arr)
        _mask_draw_image(vis)
        _mask_draw_points_and_rects(frame_file)
        ui_status("Mask preview updated.")
    except Exception as e:
        ui_log(f"[ERROR] Failed to preview mask: {e}")


def _on_create_seam_masks_now():
    if _selected_project is None:
        ui_log("[ERROR] Please select a folder first.")
        return
    try:
        project_root = Path(_selected_project)
        # If no points, guide user explicitly
        if len(_seam_points_x) < 2:
            ui_log("[INFO] Define at least two points (left, right) before creating masks.")
            return

        # If outputs are missing, still write the panorama seam mask so it is saved
        out_images_root = project_root / "output" / "images"
        if not out_images_root.exists():
            # Ensure a preview exists and write pano mask for later
            frame_file = _ensure_preview_pano_frame(project_root)
            if frame_file is not None:
                try:
                    im = Image.open(frame_file)
                    W, H = im.size
                except Exception:
                    W = H = 0
                if W > 0 and H > 0:
                    mask = _build_pano_seam_mask(W, H, _seam_points_x)
                    out_bin = project_root / "preview" / "pano_seam_mask.png"
                    out_bin.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(mask).save(out_bin)
            ui_log("[INFO] Rendered images not found yet. Run START SPLITTING to generate per-image RS seam masks. Panorama mask saved to preview.")
            return

        count = _generate_and_write_rs_seam_masks(project_root)
        if count == 0:
            ui_log("[INFO] No RS seam masks created. Ensure images exist under output/images and try again.")
    except Exception as e:
        ui_log(f"[ERROR] Failed to create RS seam masks: {e}")

def ui_status(msg: str):
    status_var.set(msg)
    _root.update_idletasks()

def ui_log(msg: str):
    log_text.configure(state=NORMAL)
    log_text.insert(END, msg.rstrip() + "\n")
    log_text.see(END)
    log_text.configure(state=DISABLED)
    _root.update_idletasks()

# def ui_main_progress(value: float | None = None, indeterminate: bool = False):
#     try:
#         progress_main.stop()
#     except Exception:
#         pass
#     if indeterminate:
#         progress_main["mode"] = "indeterminate"
#         progress_main.start(12)
#     else:
#         progress_main["mode"] = "determinate"
#         progress_main["value"] = 0 if value is None else value
#     _root.update_idletasks()

def ui_sub_progress(value: float | None = None, indeterminate: bool = False):
    try:
        progress_sub.stop()
    except Exception:
        pass
    if indeterminate:
        progress_sub["mode"] = "indeterminate"
        progress_sub.start(12)
    else:
        progress_sub["mode"] = "determinate"
        progress_sub["value"] = 0 if value is None else value
    _root.update_idletasks()

# =========================
# Preview helpers
# =========================

def _default_pairs(masking_enabled: bool) -> List[Tuple[float, float]]:
    if masking_enabled:
        return [(float(a), float(b)) for a, b in MASKING_PITCH_YAW_PAIRS]
    else:
        return [(float(a), float(b)) for a, b in NO_MASKING_PITCH_YAW_PAIRS]


def _ensure_preview_vars(reset_with_defaults: bool = False):
    global preview_time_var, _yaw_vars, _pitch_vars
    if preview_time_var is None:
        preview_time_var = StringVar(value="00:00:10")
    if _yaw_vars is None or _pitch_vars is None or reset_with_defaults:
        pairs = _default_pairs(bool(use_masking.get())) if use_masking is not None else _default_pairs(False)
        _yaw_vars = [DoubleVar(value=p[1]) for p in pairs]
        _pitch_vars = [DoubleVar(value=p[0]) for p in pairs]


def _current_pairs() -> List[Tuple[float, float]]:
    # Return list[(pitch, yaw)] from sliders
    global _yaw_vars, _pitch_vars
    if _yaw_vars is None or _pitch_vars is None:
        _ensure_preview_vars(reset_with_defaults=True)
    return [(float(_pitch_vars[i].get()), float(_yaw_vars[i].get())) for i in range(9)]


def _write_rotation_override_for_dir(root_dir: Path):
    pairs = _current_pairs()
    ref_idx = MASKING_REF_IDX if bool(use_masking.get()) else NO_MASKING_REF_IDX
    write_rotation_override(root_dir, pairs, ref_idx)


def _extract_preview_frame(video_path: Path, time_hms: str, out_file: Path) -> bool:
    try:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        # Try ffmpeg first
        cmd = [
            "ffmpeg", "-y",
            "-ss", time_hms,
            "-i", str(video_path),
            "-frames:v", "1",
            "-q:v", "2",
            str(out_file),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode == 0 and out_file.exists():
            return True
    except Exception:
        pass
    # Fallback to OpenCV
    try:
        h, m, s = [int(x) for x in time_hms.split(":")]
        seconds = h * 3600 + m * 60 + s
    except Exception:
        seconds = 10
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    frame_no = int(max(0, seconds * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()
    cap.release()
    if ok and frame is not None:
        cv2.imwrite(str(out_file), frame)
        return True
    return False


def _collect_preview_images(preview_out: Path) -> List[Path]:
    images_root = preview_out / "images"
    found: List[Path] = []
    if not images_root.exists():
        return found
    # Expect images under pano_camera*/ subfolders; pick first image per view index 0..8
    for idx in range(9):
        sub = images_root / f"pano_camera{idx}"
        if sub.exists():
            imgs = sorted([p for p in sub.rglob("*.jpg")])
            if imgs:
                found.append(imgs[0])
            else:
                found.append(Path())
        else:
            found.append(Path())
    return found


def _refresh_preview_grid(preview_root: Path):
    global preview_grid, _preview_imgs
    if preview_grid is None:
        return
    # Clear previous widgets
    for w in list(preview_grid.children.values()):
        try:
            w.destroy()
        except Exception:
            pass
    _preview_imgs.clear()

    out_dir = preview_root / "output"
    imgs = _collect_preview_images(out_dir)
    # Build 1x9 horizontal strip
    for i, img_path in enumerate(imgs):
        cell = Frame(preview_grid, bg="black")
        cell.grid(row=0, column=i, padx=4, pady=4, sticky="n")
        if img_path and img_path.exists():
            try:
                im = Image.open(img_path)
                im.thumbnail((175, 175))
                ph = ImageTk.PhotoImage(im)
                lbl = Label(cell, image=ph, bg="black")
                lbl.image = ph
                _preview_imgs.append(ph)
                lbl.pack()
            except Exception:
                Label(cell, text="(image error)", bg="black", fg="white").pack()
        else:
            Label(cell, text="(no image)", bg="black", fg="white").pack()
        # Sliders
        # Label(cell, text=f"View {i}", bg="black", fg="#ccc").pack()
        # yaw = _yaw_vars[i]
        # pitch = _pitch_vars[i]
        # Scale(cell, from_=-180, to=180, orient="horizontal", length=150, label="Yaw", variable=yaw).pack()
        # Scale(cell, from_=-90, to=90, orient="horizontal", length=150, label="Pitch", variable=pitch).pack()
        Label(cell, text=f"View {i}", bg="black", fg="#ccc").pack()
        yaw = _yaw_vars[i]
        pitch = _pitch_vars[i]
        # Yaw row: slider + numeric entry
        yaw_row = Frame(cell, bg="black"); yaw_row.pack()
        Scale(yaw_row, from_=-180, to=180, orient="horizontal",
              length=130, label="Yaw", variable=yaw).pack(side="left")
        Entry(yaw_row, textvariable=yaw, width=6, justify="right").pack(side="left", padx=(4,0))
        # Pitch row: slider + numeric entry
        pitch_row = Frame(cell, bg="black"); pitch_row.pack()
        Scale(pitch_row, from_=-90, to=90, orient="horizontal",
              length=130, label="Pitch", variable=pitch).pack(side="left")
        Entry(pitch_row, textvariable=pitch, width=6, justify="right").pack(side="left", padx=(4,0))


def _on_compute_views():
    if _selected_project is None:
        ui_log("[ERROR] Please select a folder with a video first.")
        return
    project_root = Path(_selected_project)
    videos = list_videos(project_root)
    if not videos:
        ui_log("[ERROR] No videos found in the selected folder.")
        return
    if len(videos) > 1:
        # Let user pick which video
        chosen = filedialog.askopenfilename(title="Select a video for preview", initialdir=str(project_root))
        if not chosen:
            return
        video_path = Path(chosen)
    else:
        video_path = videos[0]

    _ensure_preview_vars()
    preview_root = project_root / "preview"
    frame_file = preview_root / "frames" / "preview.jpg"

    ui_status("Extracting preview frame…")
    if not _extract_preview_frame(video_path, preview_time_var.get(), frame_file):
        ui_log("[ERROR] Failed to extract preview frame. Ensure ffmpeg is installed or try another time.")
        return

    # Write override using current sliders
    _write_rotation_override_for_dir(preview_root)

    # Render-only in preview directory
    ui_status("Rendering preview views…")
    ui_sub_progress(indeterminate=True)
    try:
        run_ok = run_panorama_sfm(preview_root, render_only=True)
    finally:
        ui_sub_progress(0, indeterminate=False)
    if not run_ok:
        ui_log("[ERROR] Preview render failed.")
        return

    _refresh_preview_grid(preview_root)
    ui_status("Preview updated.")


# =========================
# Overlay helpers (project 9 views onto panorama)
# =========================

def _dir_from_yaw_pitch(pitch_deg: float, yaw_deg: float) -> np.ndarray:
    # 3D direction for given pitch (down positive) and yaw (right positive)
    phi = math.radians(yaw_deg)
    theta = math.radians(pitch_deg)
    x = math.sin(phi) * math.cos(theta)
    y = -math.sin(theta)
    z = math.cos(phi) * math.cos(theta)
    v = np.array([x, y, z], dtype=float)
    n = np.linalg.norm(v) or 1.0
    return v / n


def _basis_from_center(pitch_deg: float, yaw_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Returns (forward d, right r, up u)
    d = _dir_from_yaw_pitch(pitch_deg, yaw_deg)
    up_world = np.array([0.0, 1.0, 0.0], dtype=float)
    r = np.cross(up_world, d)
    rn = np.linalg.norm(r)
    if rn < 1e-6:
        # Degenerate at poles; pick arbitrary right
        r = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        r = r / rn
    u = np.cross(d, r)
    un = np.linalg.norm(u) or 1.0
    u = u / un
    return d, r, u


def _yaw_pitch_from_vec(v: np.ndarray) -> tuple[float, float]:
    # Returns (yaw_rad, pitch_rad)
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    yaw = math.atan2(x, z)
    pitch = -math.atan2(y, math.hypot(x, z))
    return yaw, pitch


def _virtual_cam_from_pano_height(pano_h: int, fov_deg: float) -> tuple[int, float, float, float]:
    # Mirror panorama_sfm.create_virtual_camera
    image_size = int(pano_h * fov_deg / 180.0)
    focal = image_size / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    cx = image_size / 2.0
    cy = image_size / 2.0
    return image_size, focal, cx, cy


def _project_xy_list_to_pano_uv(
    yaw_deg: float,
    pitch_deg: float,
    fov_deg: float,
    pano_W: int,
    pano_H: int,
    x_list: list[float],
    y_list: list[float],
) -> list[tuple[float, float]]:
    # Build virtual cam intrinsics
    img_size, f, cx, cy = _virtual_cam_from_pano_height(pano_H, fov_deg)

    # Normalized camera rays
    x_arr = np.array(x_list, dtype=np.float64)
    y_arr = np.array(y_list, dtype=np.float64)
    x_norm = (x_arr - cx) / f
    y_norm = (y_arr - cy) / f
    rays_cam = np.stack([x_norm, y_norm, np.ones_like(x_norm)], axis=1)
    rays_cam /= np.linalg.norm(rays_cam, axis=1, keepdims=True)

    # Rotation: exact match to panorama_sfm (Rotation.from_euler("YX", [yaw, pitch]))
    R = Rotation.from_euler("YX", [yaw_deg, pitch_deg], degrees=True).as_matrix()
    rays_pano = rays_cam @ R

    # Convert to equirect uv
    x, y, z = rays_pano[:, 0], rays_pano[:, 1], rays_pano[:, 2]
    yaw = np.arctan2(x, z)
    pitch = -np.arctan2(y, np.hypot(x, z))
    u = (1.0 + yaw / math.pi) / 2.0 * pano_W
    v = (1.0 - (pitch * 2.0 / math.pi)) / 2.0 * pano_H
    return list(zip(u.tolist(), v.tolist()))


def _clear_overlay_canvas():
    global _overlay_items
    if overlay_canvas is None:
        return
    for item in _overlay_items:
        try:
            overlay_canvas.delete(item)
        except Exception:
            pass
    _overlay_items = []


def _draw_polyline_with_seam(canvas: Canvas, pts: list[tuple[float, float]], W: int, scale: float, color: str, width: int):
    # Draw polyline segments with seam handling; no closing edge
    def draw_seg(a, b):
        ax, ay = a; bx, by = b
        canvas_id = canvas.create_line(ax * scale, ay * scale, bx * scale, by * scale, fill=color, width=width)
        _overlay_items.append(canvas_id)

    n = len(pts)
    for i in range(n - 1):
        u0, v0 = pts[i]
        u1, v1 = pts[i + 1]
        du = u1 - u0
        if abs(du) <= W / 2:
            draw_seg((u0, v0), (u1, v1))
        else:
            # Seam crossing; split into two segments using the correct border
            if du > 0:
                # Cross left border at u=0 (wrap from left to right)
                u1_un = u1 - W
                t = (0 - u0) / (u1_un - u0)
                vc = v0 + t * (v1 - v0)
                draw_seg((u0, v0), (0, vc))
                draw_seg((W, vc), (u1, v1))
            else:
                # Cross right border at u=W (wrap from right to left)
                u1_un = u1 + W
                t = (W - u0) / (u1_un - u0)
                vc = v0 + t * (v1 - v0)
                draw_seg((u0, v0), (W, vc))
                draw_seg((0, vc), (u1, v1))


def _draw_overlays_on_canvas(image_path: Path, pairs: List[Tuple[float, float]], ref_idx: int):
    global _overlay_img_tk
    try:
        im = Image.open(image_path)
    except Exception as e:
        ui_log(f"[ERROR] Failed to load panorama: {e}")
        return

    W, H = im.size
    # Scale image for display
    max_w = 700
    disp_w = min(W, max_w)
    disp_h = int(H * (disp_w / W))
    im_disp = im.resize((disp_w, disp_h), Image.LANCZOS) if disp_w != W else im
    _overlay_img_tk = ImageTk.PhotoImage(im_disp)

    if overlay_canvas is None:
        return
    overlay_canvas.config(width=disp_w, height=disp_h)
    _clear_overlay_canvas()
    # Set background
    bg_id = overlay_canvas.create_image(0, 0, anchor="nw", image=_overlay_img_tk)
    _overlay_items.append(bg_id)

    # Colors and widths
    colors = ["#ff4d4d", "#4dff4d", "#4d4dff", "#ffd24d", "#4dd2ff", "#d24dff", "#ff884d", "#8cff4d", "#4dffd2"]
    for i, (pitch_deg, yaw_deg) in enumerate(pairs):
        color = colors[i % len(colors)]
        width_px = 3 if i == ref_idx else 2

        # Build dense edge samples to match panorama_sfm remap (curved in equirect)
        img_size, _, _, _ = _virtual_cam_from_pano_height(H, 90.0)
        s = max(16, int(img_size / 24))  # number of samples per edge, proportional to view size
        # corners in pixel center coordinates
        x0, x1 = 0.5, img_size - 0.5
        y0, y1 = 0.5, img_size - 0.5
        xs = np.linspace(x0, x1, s).tolist()
        ys = np.linspace(y0, y1, s).tolist()

        # Top edge
        pts_top = _project_xy_list_to_pano_uv(yaw_deg, pitch_deg, 90.0, W, H, xs, [y0] * s)
        # Right edge
        pts_right = _project_xy_list_to_pano_uv(yaw_deg, pitch_deg, 90.0, W, H, [x1] * s, ys)
        # Bottom edge (reverse x to keep path order)
        pts_bottom = _project_xy_list_to_pano_uv(yaw_deg, pitch_deg, 90.0, W, H, xs[::-1], [y1] * s)
        # Left edge (reverse y)
        pts_left = _project_xy_list_to_pano_uv(yaw_deg, pitch_deg, 90.0, W, H, [x0] * s, ys[::-1])

        scale = float(disp_w) / float(W)
        _draw_polyline_with_seam(overlay_canvas, pts_top, W, scale, color, width_px)
        _draw_polyline_with_seam(overlay_canvas, pts_right, W, scale, color, width_px)
        _draw_polyline_with_seam(overlay_canvas, pts_bottom, W, scale, color, width_px)
        _draw_polyline_with_seam(overlay_canvas, pts_left, W, scale, color, width_px)

        # Label at approximate center (midpoint of top edge)
        u_lbl, v_lbl = pts_top[len(pts_top) // 2]
        txt_id = overlay_canvas.create_text(u_lbl * scale + 6, v_lbl * scale + 6, text=str(i), fill=color, anchor="nw")
    _overlay_items.append(txt_id)


def _on_refresh_overlays():
    if _selected_project is None:
        ui_log("[ERROR] Please select a folder with a video first.")
        return
    project_root = Path(_selected_project)
    videos = list_videos(project_root)
    if not videos:
        ui_log("[ERROR] No videos found in the selected folder.")
        return
    if len(videos) > 1:
        chosen = filedialog.askopenfilename(title="Select a video for overlay", initialdir=str(project_root))
        if not chosen:
            return
        video_path = Path(chosen)
    else:
        video_path = videos[0]

    preview_root = project_root / "preview"
    frame_file = preview_root / "frames" / "preview.jpg"
    ui_status("Extracting panorama frame for overlays…")
    if not _extract_preview_frame(video_path, preview_time_var.get(), frame_file):
        ui_log("[ERROR] Failed to extract panorama frame for overlays.")
        return

    # Prefer using the same rotations used by the preview (if present)
    # to ensure overlays match the preview exactly.
    pairs: List[Tuple[float, float]]
    ref_idx: int
    override_path = preview_root / "rotation_override.json"
    if override_path.exists():
        try:
            data = _json.loads(override_path.read_text(encoding="utf-8"))
            raw_pairs = data.get("pitch_yaw_pairs", None)
            if isinstance(raw_pairs, list) and len(raw_pairs) == 9:
                pairs = [(float(a), float(b)) for (a, b) in raw_pairs]
                ref_idx = int(data.get("ref_idx", 0))
            else:
                # Fallback to current sliders if file malformed
                pairs = _current_pairs()
                ref_idx = MASKING_REF_IDX if bool(use_masking.get()) else NO_MASKING_REF_IDX
        except Exception:
            pairs = _current_pairs()
            ref_idx = MASKING_REF_IDX if bool(use_masking.get()) else NO_MASKING_REF_IDX
    else:
        # Fallback to current sliders
        pairs = _current_pairs()
        ref_idx = MASKING_REF_IDX if bool(use_masking.get()) else NO_MASKING_REF_IDX
    _draw_overlays_on_canvas(frame_file, pairs, ref_idx)


def refresh_action_buttons():
    project_exists = _selected_project is not None and Path(_selected_project).exists()
    if split_btn is not None:
        split_btn.configure(state=NORMAL if project_exists else DISABLED)

    align_ready = (_split_result is not None and
                   Path(_split_result.project_root).exists())
    if align_btn is not None:
        align_btn.configure(state=NORMAL if align_ready else DISABLED)

    last_ready = _last_folder is not None and Path(_last_folder).exists()
    if last_btn is not None:
        last_btn.configure(state=NORMAL if last_ready else DISABLED)


def ui_disable_inputs(disabled=True):
    if disabled:
        if drop_zone is not None:
            drop_zone.configure(state="disabled")
        for btn in (browse_btn, last_btn, split_btn, align_btn):
            if btn is not None:
                btn.configure(state=DISABLED)
    else:
        if drop_zone is not None:
            drop_zone.configure(state="normal")
        if browse_btn is not None:
            browse_btn.configure(state=NORMAL)
        refresh_action_buttons()
    # removed: loop over _yolo_widgets


# =========================
# Frame extraction with progress
# =========================

def extract_frames_with_progress(video_dir: Path, interval_seconds: float, start_seconds: int, end_seconds: Optional[int]) -> int:
    """
    Extract frames from all videos directly inside video_dir into:
        video_dir/frames/
    (All frames in ONE folder; filenames are prefixed by video name.)

    Returns: number of videos processed.
    """
    output_base_dir = video_dir / "frames"
    output_base_dir.mkdir(parents=True, exist_ok=True)

    vids = list_videos(video_dir)
    if not vids:
        ui_log(f"[WARN] No videos found in: {video_dir}")
        return 0

    total_vids = len(vids)
    for vid_idx, video_file in enumerate(vids, start=1):
        video_name = video_file.stem

        ui_status(f"Extracting frames: {video_file.name}")
        ui_log(f"[EXTRACT] {video_file.name}")

        cap = cv2.VideoCapture(str(video_file))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if not fps or fps <= 0:
            ui_log(f"[ERROR] Could not read FPS from {video_file.name}; skipping.")
            cap.release()
            continue

        # Compute start/end frames for the requested range
        max_frame_index = max(0, frame_count - 1)
        start_frame = int(round(start_seconds * fps))
        if start_frame > max_frame_index:
            duration = frame_count / fps if fps else 0.0
            ui_log(f"[WARN] Start time exceeds video length ({duration:.2f}s); skipping {video_file.name}.")
            cap.release()
            continue

        end_frame = frame_count if end_seconds is None else min(int(round(end_seconds * fps)), frame_count)
        if end_frame <= start_frame:
            ui_log(f"[WARN] Provided time range produced no frames; skipping {video_file.name}.")
            cap.release()
            continue

        # Seek to start
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        step_frames = max(1, int(round(fps * float(interval_seconds))))
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_idx < start_frame:
            frame_idx = start_frame
        saved_idx = 0

        ui_sub_progress(0, indeterminate=False)
        last_update = time.time()
        total_range_frames = max(1, end_frame - start_frame)

        while cap.isOpened():
            if frame_idx >= end_frame:
                break
            ret, frame = cap.read()
            if not ret:
                break
            if (frame_idx - start_frame) % step_frames == 0:
                frame_path = output_base_dir / f"{video_name}_frame_{saved_idx:05d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                saved_idx += 1

            frame_idx += 1

            if (time.time() - last_update) > 0.05:
                ui_sub_progress(min(100.0, 100.0 * (frame_idx - start_frame) / total_range_frames), indeterminate=False)
                last_update = time.time()

        cap.release()
        ui_sub_progress(100.0, indeterminate=False)
        ui_log(f"[OK] Saved {saved_idx} frames into {output_base_dir}")

        # ui_main_progress(min(100.0, 100.0 * vid_idx / total_vids), indeterminate=False)

    return total_vids

# =========================
# NEW: YOLO segmentation-based masking
# =========================

def run_yolo_masking(frames_root: Path) -> int:
    """
    Generate YOLO person masks without modifying original frames:
      - model: yolov8x-seg.pt, conf: 0.35, iou: 0.45, imgsz: 1024, classes: [0]
      - Input:  frames_root/*.jpg
      - Output: frames_root.parent/masks_YOLO/<image>.mask.png
        Person = black (0), background = white (255)
    Returns number of masks written.
    """
    try:
        from ultralytics import YOLO
        import numpy as np
        import cv2
        import torch
        import time
    except Exception as e:
        ui_log("[ERROR] YOLO/torch/OpenCV not available.")
        ui_log("       pip install ultralytics opencv-python torch torchvision torchaudio")
        return 0

    # --- Fixed parameters (mirroring mask_yolo.py) ---
    model_name = "yolov8x-seg.pt"
    conf = 0.35
    iou = 0.45
    imgsz = 1024
    classes = [0]           # person
    remove_person = True
    grow_person_px = 30     # expand the person region before removal

    def load_bgr(path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        return img

    def morph_expand(mask_bool: np.ndarray, grow_px: int) -> np.ndarray:
        """Grow (dilate) or shrink (erode) a boolean mask in pixel units."""
        if grow_px == 0:
            return mask_bool
        mask = (mask_bool.astype(np.uint8) * 255)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # ~3px per iteration; mirror your scriptâ€™s iterative growth behavior
        step = 3
        iters = max(1, int(abs(grow_px) / step))
        out = cv2.dilate(mask, k, iterations=iters) if grow_px > 0 else cv2.erode(mask, k, iterations=iters)
        return (out > 0)

    img_paths = sorted(frames_root.glob("*.jpg"))
    if not img_paths:
        ui_log(f"[WARN] No .jpg frames found in {frames_root}")
        return 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ui_log(f"[YOLO] Loading {model_name} on {device} â€¦")
    model = YOLO(model_name)  # auto-download if missing

    total = len(img_paths)
    processed = 0
    ui_status("Masking User")
    ui_sub_progress(0, indeterminate=False)

    last_update = time.time()

    for i, img_path in enumerate(img_paths, 1):
        try:
            bgr = load_bgr(img_path)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]

            # Predict with fixed settings (avoids imgsz=None error)
            results = model.predict(
                source=rgb,       # ndarray -> avoids loader path handling
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                verbose=False,
                classes=classes,
                device=None       # let ultralytics choose (cuda if available)
            )

            # Default: if no person found, keep image as-is
            keep_mask = np.ones((h, w), dtype=bool)

            if results and results[0].masks is not None and len(results[0].masks) > 0:
                masks_tensor = results[0].masks.data.cpu().numpy()   # (N, Hm, Wm) float [0,1]
                person_small = (masks_tensor.max(axis=0) > 0.5).astype(np.uint8)
                person = cv2.resize(person_small, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

                # Grow person region to be safe
                person = morph_expand(person, grow_person_px)

                # We remove the person â†’ keep everything else
                if remove_person:
                    keep_mask = ~person
                else:
                    keep_mask = person

            # Apply in-place: black-out pixels where keep_mask is False
            # (JPEG canâ€™t store alpha; this mimics transparency by zeroing)
            out_rgb = rgb.copy()
            out_rgb[~keep_mask] = 0  # black-out removed area
            out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(img_path), out_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            processed += 1

        except Exception as e:
            ui_log(f"[ERROR] {img_path.name}: {e}")

        if (time.time() - last_update) > 0.05:
            ui_sub_progress(min(100.0, 100.0 * i / total), indeterminate=False)
            last_update = time.time()

    ui_sub_progress(100.0, indeterminate=False)
    ui_log(f"[OK] In-place masking complete: {processed}/{total} (overwrote JPGs in {frames_root})")
    return processed


# =========================
# Override: YOLO masks to separate folder
# =========================

def run_yolo_masking(frames_root: Path) -> int:  # type: ignore[override]
    """
    Generate YOLO person masks without modifying original frames:
      - model: yolov8x-seg.pt, conf: 0.35, iou: 0.45, imgsz: 1024, classes: [0]
      - Input:  frames_root/*.jpg
      - Output: frames_root.parent/masks_YOLO/<image>.mask.png
        Person = black (0), background = white (255)
    Returns number of masks written.
    """
    try:
        from ultralytics import YOLO
        import numpy as np
        import cv2
        import torch
        import time
    except Exception:
        ui_log("[ERROR] YOLO/torch/OpenCV not available.")
        ui_log("       pip install ultralytics opencv-python torch torchvision torchaudio")
        return 0

    # --- Fixed parameters ---
    model_name = "yolov8x-seg.pt"
    conf = 0.35
    iou = 0.45
    imgsz = 1024
    classes = [0]           # person
    grow_person_px = 30

    def load_bgr(path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        return img

    def morph_expand(mask_bool: np.ndarray, grow_px: int) -> np.ndarray:
        if grow_px == 0:
            return mask_bool
        mask = (mask_bool.astype(np.uint8) * 255)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        step = 3
        iters = max(1, int(abs(grow_px) / step))
        out = cv2.dilate(mask, k, iterations=iters) if grow_px > 0 else cv2.erode(mask, k, iterations=iters)
        return (out > 0)

    img_paths = sorted(frames_root.glob("*.jpg"))
    if not img_paths:
        ui_log(f"[WARN] No .jpg frames found in {frames_root}")
        return 0

    masks_dir = frames_root.parent / "masks_YOLO"
    masks_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ui_log(f"[YOLO] Loading {model_name} on {device} â€¦")
    model = YOLO(model_name)

    total = len(img_paths)
    processed = 0
    ui_status("Generating YOLO masksâ€¦")
    ui_sub_progress(0, indeterminate=False)

    last_update = time.time()

    for i, img_path in enumerate(img_paths, 1):
        try:
            bgr = load_bgr(img_path)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]

            results = model.predict(
                source=rgb,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                verbose=False,
                classes=classes,
                device=None
            )

            person_bool = np.zeros((h, w), dtype=bool)  # default no person
            if results and results[0].masks is not None and len(results[0].masks) > 0:
                masks_tensor = results[0].masks.data.cpu().numpy()
                person_small = (masks_tensor.max(axis=0) > 0.5).astype(np.uint8)
                person = cv2.resize(person_small, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                person = morph_expand(person, grow_person_px)
                person_bool = person

            # Person black (0), background white (255)
            mask_u8 = np.where(person_bool, 0, 255).astype(np.uint8)
            out_name = f"{Path(img_path).stem}.mask.png"
            out_path = masks_dir / out_name
            cv2.imwrite(str(out_path), mask_u8)

            processed += 1

        except Exception as e:
            ui_log(f"[ERROR] {img_path.name}: {e}")

        if (time.time() - last_update) > 0.05:
            ui_sub_progress(min(100.0, 100.0 * i / total), indeterminate=False)
            last_update = time.time()

    ui_sub_progress(100.0, indeterminate=False)
    ui_log(f"[OK] Wrote {processed}/{total} masks to {masks_dir}")
    return processed


# =========================
# Seam masking helpers (panorama -> per-view RS masks)
# =========================

def _ensure_preview_pano_frame(project_root: Path) -> Path | None:
    videos = list_videos(project_root)
    if not videos:
        ui_log("[ERROR] No videos found in the selected folder.")
        return None
    if len(videos) > 1:
        chosen = filedialog.askopenfilename(title="Select a video for mask preview", initialdir=str(project_root))
        if not chosen:
            return None
        video_path = Path(chosen)
    else:
        video_path = videos[0]

    preview_root = project_root / "preview"
    frame_file = preview_root / "frames" / "preview.jpg"
    ui_status("Extracting panorama frame for masking preview.")
    ok = _extract_preview_frame(video_path, preview_time_var.get(), frame_file)
    if not ok:
        ui_log("[ERROR] Failed to extract panorama frame for masking preview.")
        return None
    return frame_file


def _build_pano_seam_mask(width: int, height: int, points_x: list[int]) -> np.ndarray:
    mask = np.full((height, width), 255, dtype=np.uint8)
    pts = points_x[:4]
    for i in range(0, len(pts) // 2):
        x0 = int(max(0, min(width - 1, pts[2 * i])))
        x1 = int(max(0, min(width - 1, pts[2 * i + 1])))
        lo, hi = (x0, x1) if x0 <= x1 else (x1, x0)
        mask[:, lo:hi + 1] = 0
    return mask


def _remap_pano_mask_to_view(pano_mask: np.ndarray, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    H = int(pano_mask.shape[0])
    W = int(pano_mask.shape[1])
    img_size, _, _, _ = _virtual_cam_from_pano_height(H, 90.0)
    cam_w = cam_h = img_size

    R = Rotation.from_euler("YX", [yaw_deg, pitch_deg], degrees=True).as_matrix()
    # Build rays grid in camera
    y, x = np.indices((cam_h, cam_w)).astype(np.float32)
    x = x + 0.5
    y = y + 0.5
    focal = _virtual_cam_from_pano_height(H, 90.0)[1]
    cx = cam_w / 2.0
    cy = cam_h / 2.0
    x_norm = (x - cx) / focal
    y_norm = (y - cy) / focal
    rays_cam = np.stack([x_norm, y_norm, np.ones_like(x_norm)], axis=-1)
    rays_cam /= np.linalg.norm(rays_cam, axis=-1, keepdims=True)
    rays_pano = np.tensordot(rays_cam, R, axes=([2],[1]))
    xr, yr, zr = rays_pano[..., 0], rays_pano[..., 1], rays_pano[..., 2]
    yaw = np.arctan2(xr, zr)
    pitch = -np.arctan2(yr, np.hypot(xr, zr))
    u = ((1.0 + yaw / np.pi) / 2.0) * W
    v = ((1.0 - (pitch * 2.0 / np.pi)) / 2.0) * H
    map_x = (u - 0.5).astype(np.float32)
    map_y = (v - 0.5).astype(np.float32)
    view_mask = cv2.remap(pano_mask, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP)
    return view_mask


def _generate_and_write_rs_seam_masks(project_root: Path) -> int:
    # If no seam points, skip
    if len(_seam_points_x) < 2:
        return 0
    # Determine panorama size from frames or preview fallback
    frames_root = project_root / "frames"
    sample = next(iter(frames_root.glob("*.jpg")), None)
    pano_img = None
    if sample is not None and sample.exists():
        pano_img = cv2.imread(str(sample), cv2.IMREAD_GRAYSCALE)
    if pano_img is None:
        # Fallback to preview panorama frame
        preview_frame = project_root / "preview" / "frames" / "preview.jpg"
        if preview_frame.exists():
            pano_img = cv2.imread(str(preview_frame), cv2.IMREAD_GRAYSCALE)
    if pano_img is None:
        ui_log("[WARN] Could not determine panorama size (no frames or preview). Skipping RS seam masks.")
        return 0
    H, W = pano_img.shape[:2]

    pano_mask = _build_pano_seam_mask(W, H, _seam_points_x)

    pairs = _current_pairs()
    total_written = 0
    for idx, (pitch_deg, yaw_deg) in enumerate(pairs):
        view_mask = _remap_pano_mask_to_view(pano_mask, pitch_deg, yaw_deg)
        # Replicate mask per image in corresponding view folder
        view_dir = project_root / "output" / "images" / f"pano_camera{idx}"
        if not view_dir.exists():
            continue
        for img_path in sorted(view_dir.glob("*.jpg")):
            out_name = view_dir / f"{img_path.stem}.mask.png"
            cv2.imwrite(str(out_name), view_mask)
            total_written += 1
    if total_written > 0:
        ui_log(f"[OK] Wrote {total_written} RS seam masks next to images under output/")
    else:
        ui_log("[WARN] No output images found to write RS seam masks. Run START SPLITTING first.")
    return total_written
# External steps (threaded helpers)
# =========================

def run_panorama_sfm(project_root: Path, render_only: bool = False) -> bool:
    wrapper = Path(__file__).parent / "run_panorama_sfm.py"
    if not wrapper.exists():
        ui_log(f"[ERROR] Missing run_panorama_sfm.py next to the GUI: {wrapper}")
        return False

    # Build wrapper command and pass optional XMP export
    cmd = [sys.executable, str(wrapper), str(project_root)]
    if render_only:
        cmd.append("--render_only")
    try:
        if export_rc_xmp.get():
            cmd.append("--export_rc_xmp")
    except Exception:
        pass
    ui_log(f"[RUN] {' '.join(cmd)}")
    ui_status("Rendering images (render_only)" if render_only else "Running COLMAP pipeline…")

    # Spin sub-progress as an activity indicator
    ui_sub_progress(indeterminate=True)

    try:
        import subprocess  # <-- keep only subprocess here

        # Stream logs live into the GUI
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        ) as proc:
            if proc.stdout is not None:
                for line in proc.stdout:
                    ui_log(line.rstrip())

            ret = proc.wait()

        ui_sub_progress(100, indeterminate=False)

        if ret == 0:
            ui_log("[OK] panorama_sfm finished.")
            return True
        else:
            ui_log(f"[ERROR] panorama_sfm exited with code {ret}.")
            return False

    except Exception as e:
        ui_sub_progress(0, indeterminate=False)
        ui_log(f"[ERROR] panorama_sfm failed: {e}")
        return False



def delete_pano_camera0(project_root: Path) -> None:
    deleter = Path(__file__).parent / "delete_pano0.py"
    if not deleter.exists():
        ui_log(f"[WARN] Missing delete_pano0.py (skipping): {deleter}")
        return

    cmd = [sys.executable, str(deleter), str(project_root)]
    ui_log(f"[RUN] {' '.join(cmd)}")
    ui_status("Deleting pano_camera0 foldersâ€¦")
    ui_sub_progress(indeterminate=True)
    try:
        subprocess.run(cmd, check=True)
        ui_sub_progress(100, indeterminate=False)
        ui_log("[OK] Deleted all pano_camera0 folders.")
    except subprocess.CalledProcessError as e:
        ui_sub_progress(0, indeterminate=False)
        ui_log(f"[WARN] delete_pano0.py reported an error: {e}")

def run_segment_images(project_root: Path) -> bool:
    """Run segment_images.py at the end, only needed when multiple input videos were used."""
    seg = Path(__file__).parent / "segment_images.py"
    if not seg.exists():
        ui_log(f"[WARN] Missing segment_images.py (skipping): {seg}")
        return False

    cmd = [sys.executable, str(seg)]
    ui_log(f"[RUN] {' '.join(cmd)} (cwd={project_root})")
    ui_status("Segmenting COLMAP images by clip prefixâ€¦")
    ui_sub_progress(indeterminate=True)
    try:
        subprocess.run(cmd, check=True, cwd=project_root)
        ui_sub_progress(100, indeterminate=False)
        ui_log("[OK] Segmented images (see Segment_images/).")
        return True
    except subprocess.CalledProcessError as e:
        ui_sub_progress(0, indeterminate=False)
        ui_log(f"[WARN] segment_images.py reported an error: {e}")
        return False

# =========================
# End-to-end pipeline (threaded)
# =========================


class SplitResult(NamedTuple):
    project_root: Path
    seconds_per_frame: float
    masking_enabled: bool
    video_count: int
    start_seconds: int
    end_seconds: Optional[int]



def run_split_stage(project_root: Path, seconds_per_frame: float, masking_enabled: bool, start_seconds: int, end_seconds: Optional[int]) -> SplitResult:
    # ui_main_progress(0, indeterminate=False)
    ui_status("Preparing extraction...")
    video_count = extract_frames_with_progress(project_root, seconds_per_frame, start_seconds, end_seconds)
    frames_root = project_root / "frames"

    if masking_enabled:
        ui_status("Writing rotation override (masking)...")
        write_rotation_override(project_root, _current_pairs(), MASKING_REF_IDX)
        ui_log(f"[OK] Wrote rotation_override.json (masking) in {project_root}")

        ui_status("Masking frames (fixed YOLO settings)...")
        processed = run_yolo_masking(frames_root=frames_root)
        if processed == 0:
            ui_log("[WARN] No frames were masked (or masking skipped due to error). Continuing...")
    else:
        ui_status("Writing rotation override (no masking)...")
        write_rotation_override(project_root, _current_pairs(), NO_MASKING_REF_IDX)
        ui_log(f"[OK] Wrote rotation_override.json (no masking) in {project_root}")

    # ui_main_progress(33, indeterminate=False)
    # Render perspective images/masks (pre-indexing) via COLMAP wrapper
    ui_status("Rendering perspective images from panoramas…")
    ok = run_panorama_sfm(project_root, render_only=True)
    if not ok:
        ui_log("[ERROR] Rendering step failed. See log.")
        return SplitResult(project_root, seconds_per_frame, masking_enabled, video_count)
    # Generate RS seam masks if seam points were defined
    try:
        _generate_and_write_rs_seam_masks(project_root)
    except Exception as e:
        ui_log(f"[WARN] RS seam mask generation failed: {e}")
    return SplitResult(
        project_root=project_root,
        seconds_per_frame=seconds_per_frame,
        masking_enabled=masking_enabled,
        video_count=video_count,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )



def run_align_stage(project_root: Path, video_count: int) -> bool:
    #ui_main_progress(33, indeterminate=False)

    if not run_panorama_sfm(project_root):
        ui_status("COLMAP failed. See log.")
        return False

    #ui_main_progress(66, indeterminate=False)

    delete_pano_camera0(project_root)
    #ui_main_progress(90, indeterminate=False)

    if video_count > 1:
        ui_log(f"[INFO] Multiple videos detected ({video_count}). Running segment_images...")
        run_segment_images(project_root)

    #ui_main_progress(100, indeterminate=False)
    ui_status("All done.")
    ui_log("[DONE] Pipeline complete.")
    return True



def _stop_progress_bars():
    try:
        # progress_main.stop()
        progress_sub.stop()
    except Exception:
        pass



def _split_thread(project_root: Path, seconds_per_frame: float, masking_enabled: bool, start_seconds: int, end_seconds: Optional[int]) -> None:
    global _split_result
    try:
    # remove preview folder before splitting
        preview_dir = project_root / "preview"
        if preview_dir.exists():
            try:
                shutil.rmtree(preview_dir)
                ui_log(f"[OK] Removed preview folder: {preview_dir}")
            except Exception as e:
                ui_log(f"[WARN] Could not remove preview folder {preview_dir}: {e}")

        ui_disable_inputs(True)
        ui_sub_progress(0, indeterminate=False)
        _split_result = None
        ui_log(f"[INFO] Splitting time range: {start_seconds:02d}s to {('end' if end_seconds is None else f'{end_seconds:02d}s')}")
        result = run_split_stage(project_root, seconds_per_frame, masking_enabled, start_seconds, end_seconds)
        _split_result = result
        ui_status("Splitting complete. Ready for alignment.")
        ui_log("[OK] Splitting finished. Press START ALIGNING to continue.")
    except Exception as exc:
        _split_result = None
        ui_sub_progress(0, indeterminate=False)
        ui_log(f"[ERROR] Splitting failed: {exc}")
        ui_status("Splitting failed. See log.")
    finally:
        _stop_progress_bars()
        ui_disable_inputs(False)
        refresh_action_buttons()



def _align_thread(split_result: SplitResult) -> None:
    try:
        ui_disable_inputs(True)
        ui_sub_progress(0, indeterminate=False)
        success = run_align_stage(split_result.project_root, split_result.video_count)
        if not success:
            ui_log("[WARN] Alignment stage did not complete successfully.")
    except Exception as exc:
        ui_sub_progress(0, indeterminate=False)
        ui_log(f"[ERROR] Alignment failed: {exc}")
        ui_status("Alignment failed. See log.")
    finally:
        _stop_progress_bars()
        ui_disable_inputs(False)
        refresh_action_buttons()



# =========================
# GUI
# =========================

def _read_seconds_per_frame() -> float:
    try:
        seconds = float(frame_interval.get())
        if seconds <= 0:
            raise ValueError
    except Exception:
        seconds = 1.0
        frame_interval.set("1")
    return seconds


def _read_masking_flag() -> bool:
    try:
        return bool(use_masking.get())
    except Exception:
        return False


def start_pipeline_with_path(folder_path: Path):
    global _last_folder, _selected_project, _split_result
    folder_path = Path(folder_path)
    if not folder_path.exists():
        ui_log(f"[ERROR] Folder does not exist: {folder_path}")
        return

    _selected_project = folder_path
    _last_folder = folder_path
    _split_result = None

    ui_status(f"Selected folder: {folder_path}")
    ui_log(f"[INFO] Selected folder: {folder_path}")

    refresh_action_buttons()


def on_drop(event):
    folder_path = Path(event.data.strip("{}"))
    if folder_path.is_dir():
        start_pipeline_with_path(folder_path)
    else:
        ui_log("[ERROR] Please drop a valid folder.")


def browse_folder():
    chosen = filedialog.askdirectory(title="Select folder containing 360 video(s)")
    if chosen:
        start_pipeline_with_path(Path(chosen))


def run_last():
    if _last_folder and Path(_last_folder).exists():
        start_pipeline_with_path(Path(_last_folder))
    else:
        ui_log("[ERROR] No valid last folder. Please Browse or Drop a folder.")


def on_start_split():
    global _split_result, _selected_project
    if _selected_project is None:
        ui_log("[ERROR] Please select a folder before starting the split stage.")
        return

    project_root = Path(_selected_project)
    if not project_root.exists():
        ui_log(f"[ERROR] Selected folder is missing: {project_root}")
        _selected_project = None
        refresh_action_buttons()
        return

    seconds = _read_seconds_per_frame()
    masking = _read_masking_flag()

    # Parse and validate time range from UI
    try:
        s_text = start_time_var.get() if start_time_var is not None else ""
        e_text = end_time_var.get() if end_time_var is not None else ""
        start_seconds, end_seconds = normalize_time_range(s_text, e_text)
    except Exception as ex:
        try:
            messagebox.showwarning("Invalid time range", str(ex))
        except Exception:
            pass
        ui_log(f"[ERROR] Invalid time range: {ex}")
        return

    _split_result = None
    refresh_action_buttons()

    if split_btn is not None:
        split_btn.configure(state=DISABLED)
    if align_btn is not None:
        align_btn.configure(state=DISABLED)

    threading.Thread(
        target=_split_thread,
        args=(project_root, seconds, masking, start_seconds, end_seconds),
        daemon=True
    ).start()


def on_start_align():
    if _split_result is None:
        ui_log("[ERROR] Run START spltting before alignment.")
        return

    project_root = Path(_split_result.project_root)
    if not project_root.exists():
        ui_log(f"[ERROR] Project folder is missing: {project_root}")
        ui_status("Alignment aborted. Folder missing.")
        refresh_action_buttons()
        return

    if align_btn is not None:
        align_btn.configure(state=DISABLED)

    threading.Thread(
        target=_align_thread,
        args=(_split_result,),
        daemon=True
    ).start()


def on_masking_toggle():
    try:
        _ensure_preview_vars(reset_with_defaults=True)
    except Exception:
        pass


def main():
    global _root, status_var, progress_sub, log_text
    global use_masking, frame_interval, start_time_var, end_time_var, drop_zone, browse_btn, last_btn, split_btn, align_btn
    global export_rc_xmp
    global yolo_model_path, yolo_conf, yolo_dilate_px, yolo_invert_mask, yolo_apply_to_rgb, _yolo_widgets

    _root = TkinterDnD.Tk()
    _root.title("360 Video Dataset Preparation")
    _root.geometry("1280x900")
    _root.configure(bg="black")
    _root.resizable(True, True)

    # ---------- Header ----------
    header = Frame(_root, bg="black")
    header.pack(pady=(14, 8))

    icon_path = Path(__file__).parent / "folder_icon.png"
    if icon_path.exists():
        try:
            img = Image.open(icon_path).resize((84, 84))
            icon = ImageTk.PhotoImage(img)
            icon_label = Label(header, image=icon, bg="black")
            icon_label.image = icon
            icon_label.pack(side="left", padx=(0, 12))
        except Exception:
            Label(header, text="📁", font=("Arial", 44), bg="black", fg="white").pack(side="left", padx=(0, 12))
    else:
        Label(header, text="📁", font=("Arial", 44), bg="black", fg="white").pack(side="left", padx=(0, 12))

    Label(header,
          text="Insta360 Video(s) To Training Format Pipeline",
          bg="black", fg="white", font=("Helvetica", 16, "bold")).pack(side="left")

    # ---------- Controls ----------
    ctrl = Frame(_root, bg="black")
    ctrl.pack(pady=(8, 2))

    Label(ctrl, text="Extract 1 frame per", bg="black", fg="white").pack(side="left")

    frame_interval = StringVar(value="1")
    Entry(ctrl, textvariable=frame_interval, width=6).pack(side="left", padx=(6, 6))

    Label(ctrl, text="seconds", bg="black", fg="white").pack(side="left", padx=(0, 16))

    use_masking = BooleanVar(value=False)
    mcb = Checkbutton(
        ctrl,
        text="Enable Masking (removes user from frames)",
        variable=use_masking,
        onvalue=True, offvalue=False,
        bg="black", fg="white", activebackground="black",
        selectcolor="black",
    )
    mcb.pack(side="left")

    # XMP export toggle
    export_rc_xmp = BooleanVar(value=False)
    xmp_cb = Checkbutton(
        ctrl,
        text="Export XMP for RealityCapture",
        variable=export_rc_xmp,
        onvalue=True, offvalue=False,
        bg="black", fg="white", activebackground="black",
        selectcolor="black",
    )
    xmp_cb.pack(side="left", padx=(12,0))

    # ---------- Drop zone ----------
    drop_zone = Label(
        _root, text="Drop Folder With Insta360 Videos Here",
        bg="#222", fg="white", width=70, height=6,
        relief="ridge", bd=2
    )
    drop_zone.pack(pady=10)
    drop_zone.drop_target_register(DND_FILES)
    drop_zone.dnd_bind('<<Drop>>', on_drop)

    # ---------- Buttons ----------
    btn_style = {"bg":"#333","fg":"white","activebackground":"#444","activeforeground":"white","disabledforeground":"#888"}
    btns = Frame(_root, bg="black")
    btns.pack()
    browse_btn = Button(btns, text="Browse", command=browse_folder, **btn_style)
    browse_btn.pack(side="left", padx=6)
    last_btn = Button(btns, text="Run Last Chosen Folder", state=DISABLED, command=run_last, **btn_style)
    last_btn.pack(side="left", padx=6)

    # ---------- Progress ----------
    prog = Frame(_root, bg="black")
    prog.pack(fill=BOTH, padx=16, pady=(12, 4))

    # Label(prog, text="Overall Progress", bg="black", fg="#ccc").pack(anchor="w")
    # progress_main = ttk.Progressbar(prog, orient="horizontal", mode="determinate", length=680)
    # progress_main.pack(pady=(2, 8))

    Label(prog, text="Current Task", bg="black", fg="#ccc").pack(anchor="w")
    progress_sub = ttk.Progressbar(prog, orient="horizontal", mode="determinate", length=680)
    progress_sub.pack(pady=(2, 8))

    status_var = StringVar(value="Idle.")
    Label(_root, textvariable=status_var, bg="black", fg="white").pack(pady=(0, 6))

    # ---------- Time Range ----------
    range_frame = Frame(_root, bg="black")
    range_frame.pack(pady=(6, 2))

    start_lbl = Label(range_frame, text="Start (hh:mm:ss)", bg="black", fg="white")
    start_lbl.pack(side="left", padx=(0, 6))
    start_time_var = StringVar(value="")
    Entry(range_frame, textvariable=start_time_var, width=10).pack(side="left", padx=(0, 16))

    end_lbl = Label(range_frame, text="End (hh:mm:ss)", bg="black", fg="white")
    end_lbl.pack(side="left", padx=(0, 6))
    end_time_var = StringVar(value="")
    Entry(range_frame, textvariable=end_time_var, width=10).pack(side="left")

    # ---------- Stage Controls ----------
    stage_btns = Frame(_root, bg="black")
    stage_btns.pack(pady=(0, 12))
    split_btn = Button(stage_btns, text="START SPLITTING", state=DISABLED, command=on_start_split, **btn_style)
    split_btn.pack(side="left", padx=6)
    align_btn = Button(stage_btns, text="START ALIGNING", state=DISABLED, command=on_start_align, **btn_style)
    align_btn.pack(side="left", padx=6)

    # ---------- Log ----------
    log_frame = Frame(_root, bg="black")
    log_frame.pack(fill=BOTH, expand=True, padx=16, pady=(0, 12))
    log_text = Text(log_frame, height=6, bg="#111", fg="#ddd", insertbackground="white")
    log_text.pack(fill=BOTH)
    log_text.configure(state=DISABLED)

    # ---------- Preview / Overlays Tabs ----------
    _ensure_preview_vars(reset_with_defaults=True)
    tabs = ttk.Notebook(_root)
    tabs.pack(fill=BOTH, expand=True, padx=16, pady=(8, 6))

    # Preview tab
    preview_tab = Frame(tabs, bg="black")
    tabs.add(preview_tab, text="Preview")

    prev_ctrl = Frame(preview_tab, bg="black")
    prev_ctrl.pack(fill=BOTH, pady=(4, 6))
    Label(prev_ctrl, text="Preview time (HH:MM:SS)", bg="black", fg="white").pack(side="left")
    Entry(prev_ctrl, textvariable=preview_time_var, width=10).pack(side="left", padx=(6, 12))
    Button(prev_ctrl, text="Compute views", command=_on_compute_views).pack(side="left")

    # 3x3 preview grid
    global preview_grid
    preview_wrap = Frame(preview_tab, bg="black")
    preview_wrap.pack(fill=BOTH, pady=(6, 12))
    global preview_canvas
    preview_canvas = Canvas(preview_wrap, bg="black", highlightthickness=0, height=500)
    preview_canvas.pack(side="left", fill=BOTH, expand=True)
    global preview_grid, hscroll, vscroll
    preview_grid = Frame(preview_canvas, bg="black")
    preview_canvas.create_window((0, 0), window=preview_grid, anchor="nw")

    # Overlays tab
    overlays_tab = Frame(tabs, bg="black")
    tabs.add(overlays_tab, text="Overlays")

    ov_ctrl = Frame(overlays_tab, bg="black")
    ov_ctrl.pack(fill=BOTH, pady=(4, 6))
    Label(ov_ctrl, text="Uses Preview time", bg="black", fg="#ccc").pack(side="left", padx=(0, 12))
    Button(ov_ctrl, text="Refresh Overlays", command=_on_refresh_overlays).pack(side="left")

    global overlay_canvas
    overlay_canvas = Canvas(overlays_tab, bg="black", highlightthickness=0, height=520)
    overlay_canvas.pack(fill=BOTH, expand=True, pady=(6, 12))

    # Masking tab
    masking_tab = Frame(tabs, bg="black")
    tabs.add(masking_tab, text="Masking")

    mk_ctrl = Frame(masking_tab, bg="black")
    mk_ctrl.pack(fill=BOTH, pady=(4, 6))
    Button(mk_ctrl, text="Preview Mask", command=lambda: _on_preview_seam_mask()).pack(side="left", padx=(0, 8))
    Button(mk_ctrl, text="Create Masks", command=lambda: _on_create_seam_masks_now()).pack(side="left", padx=(0, 8))
    Button(mk_ctrl, text="Clear Last Point", command=lambda: _on_clear_last_seam_point()).pack(side="left")

    global mask_canvas
    mask_canvas = Canvas(masking_tab, bg="black", highlightthickness=0, height=520)
    mask_canvas.pack(fill=BOTH, expand=True, pady=(6, 12))
    try:
        mask_canvas.bind('<Button-1>', _on_mask_canvas_click)
    except Exception:
        pass

    refresh_action_buttons()

    # initialize YOLO controls state
    on_masking_toggle()

    _root.mainloop()

if __name__ == "__main__":
    main()







