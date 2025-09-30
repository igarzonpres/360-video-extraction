# -*- coding: utf-8 -*-
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
    Tk, Label, Entry, StringVar, DoubleVar, Frame, Checkbutton, BooleanVar,
    filedialog, messagebox, Button, Text, Scale, Canvas, END, BOTH, DISABLED, NORMAL
)
# Drag-and-drop dependency removed
from tkinter import ttk
from PIL import Image, ImageTk
from time_range import normalize_time_range
from scipy.spatial.transform import Rotation

import numpy as np  # NEW: for mask processing
from typing import List, Tuple, Optional
import shutil
import json as _json
import shutil as _shutil

# =========================
# UI palette (centralized)
# =========================
# Light theme palette used across the GUI. To tweak the theme,
# change values here instead of per-widget.
PALETTE = {
    "bg": "white",              # main window and frames
    "fg": "#222222",           # primary text
    "muted_fg": "#555555",     # secondary text
    "panel_bg": "#f7f7f7",     # panel/sections background (if needed)
    "drop_bg": "#efefef",      # drop zone background
    "btn_bg": "#e6e6e6",       # button background
    "btn_fg": "#222222",       # button text
    "btn_active_bg": "#d9d9d9",# button active background
    "btn_active_fg": "#000000",# button active text
    "btn_disabled_fg": "#999999", # button disabled fg
    "canvas_bg": "white",      # canvas background
    "log_bg": "#fafafa",       # log box background
    "log_fg": "#333333",       # log box text
    "insert_bg": "#000000",    # caret color in text widgets
}

def _btn_style():
    return {
        "bg": PALETTE["btn_bg"],
        "fg": PALETTE["btn_fg"],
        "activebackground": PALETTE["btn_active_bg"],
        "activeforeground": PALETTE["btn_active_fg"],
        "disabledforeground": PALETTE["btn_disabled_fg"],
    }

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
# NO_MASKING_PITCH_YAW_PAIRS = [
#     (0, 90),   # Reference Pose (ref_idx = 0)
#     (32, 0),
#     (-42, 0),
#     (0, 42),  # para la R1 usar 33, dejo el default para hacerlo compatible con la X5  
#     (0, -25),
#     (42, 180),
#     (-32, 180),
#     (0, 205),
#     (0, 138), # para la R1 subirlo (142) para evitar que salga el hombro, dejo el default para hacerlo compatible con la X5
# ]
NO_MASKING_PITCH_YAW_PAIRS = [
    (0, 90),   # Reference Pose (ref_idx = 0)
    (25, 10),
    (-30, 0),
    (0, 42),  # para la R1 usar 33, dejo el default para hacerlo compatible con la X5  
    (0, -25),
    (30, 170),
    (-25, 170),
    (0, -170),
    (0, 138), # para la R1 subirlo (142) para evitar que salga el hombro, dejo el default para hacerlo compatible con la X5
]
NO_MASKING_REF_IDX = 0

# Inverted preset (no masking): user-requested alternative angles
# Format: list of (pitch, yaw) in degrees
INVERTED_NO_MASKING_PITCH_YAW_PAIRS = [
    (0, -90),
    (32, -180),
    (-42, -180),
    (0, -138),
    (0, 155),
    (42, 0),
    (-32, 0),
    (0, 0),
    (0, -42),
]

# Toggle flag controlled by the GUI button
inverted_preset_active = False

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


def _is_two_to_one_image(path: Path, tol: float = 0.01) -> bool:
    try:
        with Image.open(path) as im:
            w, h = im.size
        if h == 0:
            return False
        ratio = float(w) / float(h)
        return abs(ratio - 2.0) <= tol
    except Exception:
        return False


def detect_panorama_frames_dir(project_root: Path) -> tuple[bool, list[Path]]:
    """Return (is_valid, images) for project_root/frames treated as panorama images.
    Valid when the folder exists, has >=1 images (.jpg/.jpeg/.png), and all are 2:1.
    Checks only in the selected folder (no recursion beyond 'frames').
    """
    frames_dir = project_root / "frames"
    if not frames_dir.exists() or not frames_dir.is_dir():
        return False, []
    exts = {".jpg", ".jpeg", ".png"}
    imgs = sorted([p for p in frames_dir.glob("*") if p.suffix.lower() in exts and p.is_file()])
    if not imgs:
        return False, []
    for p in imgs:
        if not _is_two_to_one_image(p):
            return False, imgs
    return True, imgs


def _list_frames_images_sorted(project_root: Path) -> list[Path]:
    """List images in <project_root>/frames sorted by filename (case-insensitive).
    Considers .jpg/.jpeg/.png. Returns an empty list if folder missing.
    """
    frames_dir = project_root / "frames"
    if not frames_dir.exists() or not frames_dir.is_dir():
        return []
    exts = {".jpg", ".jpeg", ".png"}
    imgs = [p for p in frames_dir.glob("*") if p.is_file() and p.suffix.lower() in exts]
    return sorted(imgs, key=lambda p: p.name.lower())
# =========================
# UI helpers (set from thread)
# =========================

_root = None
status_var = None
# progress_main = None
progress_sub = None
log_text = None
invert_panos_var = None  # GUI checkbox for 180 deg panorama inversion
panorama_mode_var = None  # GUI checkbox to prefer frames over video
video_mode_var = None     # GUI checkbox to force video mode (mutually exclusive)

# Paths configured via Settings tab
ffmpeg_path_var = None
jpegtran_path_var = None
# Panorama mode globals
panorama_mode_active = False
_panorama_images: list[Path] | None = None
preview_pano_index_var = None  # StringVar with 1-based panorama index for preview/overlays
preview_time_entry = None  # Entry widget to allow disabling when in panorama mode


use_masking = None
frame_interval = None
start_time_var = None
end_time_var = None
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
preview_canvas_window_id = None  # Canvas window id for centering

# =========================
# Overlay globals (Overlays tab)
# =========================
overlay_canvas = None
_overlay_img_tk = None
_overlay_items = []  # canvas item ids to clear

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

def _frames_available() -> bool:
    try:
        return bool(_panorama_images) and len(_panorama_images or []) > 0
    except Exception:
        return False

def _refresh_controls_for_panorama_mode():
    # Disable/enable controls that are irrelevant in panorama mode
    global preview_time_entry
    try:
        if preview_time_entry is not None:
            preview_time_entry.configure(state=DISABLED if panorama_mode_active else NORMAL)
    except Exception:
        pass

def _update_panorama_mode_active_from_ui(log: bool = True):
    """Sync internal panorama_mode_active with UI checkbox and frames availability."""
    global panorama_mode_active
    want = False
    force_video = False
    try:
        want = bool(panorama_mode_var.get()) if panorama_mode_var is not None else False
    except Exception:
        want = False
    try:
        force_video = bool(video_mode_var.get()) if video_mode_var is not None else False
    except Exception:
        force_video = False
    if force_video:
        panorama_mode_active = False
        if log:
            ui_log("[INFO] Video Mode is ON. Using video-based workflow.")
        _refresh_controls_for_panorama_mode()
        return
    if want and not _frames_available():
        # Cannot enable without frames; revert checkbox
        try:
            if panorama_mode_var is not None:
                panorama_mode_var.set(False)
            if video_mode_var is not None:
                video_mode_var.set(True)
        except Exception:
            pass
        panorama_mode_active = False
        if log:
            ui_log("[WARN] Panorama Mode requested but no valid panoramas in 'frames/'.")
    else:
        panorama_mode_active = want
        if log:
            ui_log("[INFO] Panorama Mode is ON. Frames take priority; preview time ignored." if panorama_mode_active else "[INFO] Panorama Mode is OFF. Using video-based workflow.")
    _refresh_controls_for_panorama_mode()

def _select_and_copy_panorama(project_root: Path, index_1based: int, dst_file: Path, context: str) -> bool:
    """Select a panorama by 1-based index (sorted by name) and copy to dst_file.
    Returns True on success.
    """
    imgs = _list_frames_images_sorted(project_root)
    if not imgs:
        ui_log("[ERROR] No panoramas found in frames/ while in panorama mode.")
        return False
    try:
        idx = int(index_1based)
    except Exception:
        idx = 1
    idx = max(1, min(idx, len(imgs)))
    src = imgs[idx - 1]
    try:
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst_file)
        ui_status(f"Using panorama #{idx} for {context}.")
        ui_log(f"[PANORAMA] {context}: #{idx} -> {dst_file}")
        return True
    except Exception as e:
        ui_log(f"[ERROR] Failed to prepare panorama for {context}: {e}")
        return False

def _default_pairs(masking_enabled: bool) -> List[Tuple[float, float]]:
    # Returns the initial preset used to seed sliders
    # Respect the inverted toggle only for the non-masking profile
    if masking_enabled:
        return [(float(a), float(b)) for a, b in MASKING_PITCH_YAW_PAIRS]
    else:
        pairs = INVERTED_NO_MASKING_PITCH_YAW_PAIRS if inverted_preset_active else NO_MASKING_PITCH_YAW_PAIRS
        return [(float(a), float(b)) for a, b in pairs]


def _apply_pairs_to_vars(pairs: List[Tuple[float, float]]):
    # Apply given (pitch, yaw) pairs to existing slider variables
    global _yaw_vars, _pitch_vars
    _ensure_preview_vars(reset_with_defaults=False)
    for i in range(9):
        p, y = pairs[i]
        _pitch_vars[i].set(float(p))
        _yaw_vars[i].set(float(y))


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
        if panorama_mode_active and _panorama_images:
            return True

        # Try ffmpeg first, resolving from settings or PATH
        exe = None
        try:
            cfg = ffmpeg_path_var.get() if ffmpeg_path_var is not None else ""
            exe = cfg.strip() or None
        except Exception:
            exe = None
        if not exe:
            exe = _shutil.which("ffmpeg")
        if exe:
            cmd = [
                exe, "-y",
                "-ss", time_hms,
                "-i", str(video_path),
                "-frames:v", "1",
                "-q:v", "2",
                str(out_file),
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if proc.returncode == 0 and out_file.exists():
                return True
        else:
            ui_log("[ERROR] ffmpeg not configured and not found in PATH. Falling back to OpenCV.")
    except Exception as e:
        ui_log(f"[WARN] ffmpeg preview extraction failed ({e}). Falling back to OpenCV.")
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


def _resolve_jpegtran_exe() -> Optional[str]:
    """Return jpegtran executable path if configured and exists; else try PATH; else None."""
    try:
        cfg = jpegtran_path_var.get() if jpegtran_path_var is not None else ""
        exe = cfg.strip()
        if exe:
            p = Path(exe)
            if p.exists():
                return str(p)
    except Exception:
        pass
    found = _shutil.which("jpegtran")
    return found


def _rotate_panoramas_in_dir(frames_root: Path) -> bool:
    """Rotate all .jpg in frames_root by 180 deg using jpegtran, preserving metadata.
    Returns True if at least one rotation succeeded or if there were no images; False if jpegtran missing or all failed.
    """
    if not frames_root.exists():
        return True
    imgs = sorted(frames_root.glob("*.jpg"))
    if not imgs:
        return True
    exe = _resolve_jpegtran_exe()
    if not exe:
        ui_log("[WARN] jpegtran not configured or not found. Set it under Settings > jpegtran.exe path.")
        try:
            messagebox.showwarning("jpegtran missing", "jpegtran.exe not found. Set it in the Settings tab or add to PATH. Continuing without inversion.")
        except Exception:
            pass
        return False
    ok_count = 0
    for src in imgs:
        try:
            tmp = src.with_suffix(".rot180.jpg")
            cmd = _jpegtran_cmd(exe, src, tmp)
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if proc.returncode == 0 and tmp.exists():
                try:
                    tmp.replace(src)
                except Exception:
                    _shutil.move(str(tmp), str(src))
                ok_count += 1
            else:
                try:
                    err = proc.stderr.decode(errors='ignore') if proc.stderr is not None else ''
                except Exception:
                    err = ''
                ui_log(f"[WARN] jpegtran failed for {src.name}: {err[:200]}")
                if tmp.exists():
                    try:
                        tmp.unlink()
                    except Exception:
                        pass
        except Exception as e:
            ui_log(f"[WARN] jpegtran error for {src.name}: {e}")
    if ok_count > 0:
        ui_log(f"[OK] Inverted {ok_count}/{len(imgs)} panorama(s) with jpegtran.")
        return True
    return False


def _jpegtran_cmd(exe: str, src: Path, dst: Path) -> list[str]:
    # IJG jpegtran expects output via -outfile <dst> then <src>
    return [str(exe), "-rotate", "180", "-copy", "all", "-perfect", "-outfile", str(dst), str(src)]


def _collect_preview_images(preview_out: Path) -> List[Path]:
    images_root = preview_out / "images"
    found: List[Path] = []
    if not images_root.exists():
        return found
    # Expect images under pano_camera*/ subfolders; pick first image per view index 0..8
    for idx in range(9):
        sub = images_root / f"pano_camera{idx}"
        if sub.exists():
            imgs = [p for p in sub.rglob("*.jpg")]
            if imgs:
                try:
                    latest = max(imgs, key=lambda p: p.stat().st_mtime)
                except Exception:
                    latest = sorted(imgs)[-1]
                found.append(latest)
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
    # Build 2-row layout: first 5 on top row, remaining 4 on bottom row
    for i, img_path in enumerate(imgs):
        row = 0 if i < 5 else 1
        col = i if i < 5 else (i - 5)
        cell = Frame(preview_grid, bg=PALETTE["bg"])
        cell.grid(row=row, column=col, padx=6, pady=6, sticky="n")
        if img_path and img_path.exists():
            try:
                im = Image.open(img_path)
                im.thumbnail((185, 185))
                ph = ImageTk.PhotoImage(im)
                lbl = Label(cell, image=ph, bg=PALETTE["bg"]) 
                lbl.image = ph
                _preview_imgs.append(ph)
                lbl.pack()
            except Exception:
                Label(cell, text="(image error)", bg=PALETTE["bg"], fg=PALETTE["muted_fg"]).pack()
        else:
            Label(cell, text="(no image)", bg=PALETTE["bg"], fg=PALETTE["muted_fg"]).pack()
        # Sliders
        # Label(cell, text=f"View {i}", bg=PALETTE["bg"], fg=PALETTE["muted_fg"]).pack()
        # yaw = _yaw_vars[i]
        # pitch = _pitch_vars[i]
        # Scale(cell, from_=-180, to=180, orient="horizontal", length=150, label="Yaw", variable=yaw).pack()
        # Scale(cell, from_=-90, to=90, orient="horizontal", length=150, label="Pitch", variable=pitch).pack()
        Label(cell, text=f"View {i}", bg=PALETTE["bg"], fg=PALETTE["muted_fg"]).pack()
        yaw = _yaw_vars[i]
        pitch = _pitch_vars[i]
        # Yaw row: slider + numeric entry
        yaw_row = Frame(cell, bg=PALETTE["bg"]); yaw_row.pack()
        Scale(yaw_row, from_=-180, to=180, orient="horizontal",
              length=130, label="Yaw", variable=yaw).pack(side="left")
        Entry(yaw_row, textvariable=yaw, width=6, justify="right").pack(side="left", padx=(4,0))
        # Pitch row: slider + numeric entry
        pitch_row = Frame(cell, bg=PALETTE["bg"]); pitch_row.pack()
        Scale(pitch_row, from_=-90, to=90, orient="horizontal",
              length=130, label="Pitch", variable=pitch).pack(side="left")
        Entry(pitch_row, textvariable=pitch, width=6, justify="right").pack(side="left", padx=(4,0))


def _on_compute_views():
    if _selected_project is None:
        ui_log("[ERROR] Please select a folder with a video first.")
        return
    project_root = Path(_selected_project)
    # When in panorama mode, skip video selection entirely
    if not (panorama_mode_active and _panorama_images):
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
    # Ensure a clean preview workspace to avoid stale results from previous runs
    try:
        if preview_root.exists():
            shutil.rmtree(preview_root)
            ui_log(f"[INFO] Cleared previous preview folder: {preview_root}")
    except Exception:
        pass
    frame_file = preview_root / "frames" / "preview.jpg"
    # Panorama-mode: copy selected panorama into preview frame file
    if panorama_mode_active:
        idx = 1
        if preview_pano_index_var is not None:
            try:
                idx = int(preview_pano_index_var.get() or "1")
            except Exception:
                idx = 1
        ok = _select_and_copy_panorama(project_root, idx, frame_file, context="preview")
        if not ok:
            return

    if not panorama_mode_active:
        ui_status("Extracting preview frame...")
        if not _extract_preview_frame(video_path, preview_time_var.get(), frame_file):
            ui_log("[ERROR] Failed to extract preview frame. Ensure ffmpeg is installed or try another time.")
            return

    # Optional panorama inversion before preview split
    try:
        inv = bool(invert_panos_var.get()) if invert_panos_var is not None else False
    except Exception:
        inv = False
    if inv:
        ui_status("Inverting panorama (preview) with jpegtran...")
        _rotate_panoramas_in_dir(frame_file.parent)

    # Write override using current sliders
    _write_rotation_override_for_dir(preview_root)

    # Render-only in preview directory
    ui_status("Rendering preview views...")
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


def _draw_polyline_with_seam(canvas: Canvas, pts: list[tuple[float, float]], W: int, scale: float, color: str, width: int, dx: float = 0.0, dy: float = 0.0):
    # Draw polyline segments with seam handling; no closing edge
    def draw_seg(a, b):
        ax, ay = a; bx, by = b
        canvas_id = canvas.create_line(ax * scale + dx, ay * scale + dy, bx * scale + dx, by * scale + dy, fill=color, width=width)
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
    max_w = 1300
    disp_w = min(W, max_w)
    disp_h = int(H * (disp_w / W))
    im_disp = im.resize((disp_w, disp_h), Image.LANCZOS) if disp_w != W else im
    _overlay_img_tk = ImageTk.PhotoImage(im_disp)

    if overlay_canvas is None:
        return
    # Center within current canvas size
    try:
        _root.update_idletasks()
    except Exception:
        pass
    try:
        cw = max(int(overlay_canvas.winfo_width()), disp_w)
        ch = max(int(overlay_canvas.winfo_height()), disp_h)
    except Exception:
        cw, ch = disp_w, disp_h
    offx = max(0, (cw - disp_w) // 2)
    offy = max(0, (ch - disp_h) // 2)
    _clear_overlay_canvas()
    # Set background centered
    bg_id = overlay_canvas.create_image(offx, offy, anchor="nw", image=_overlay_img_tk)
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
        _draw_polyline_with_seam(overlay_canvas, pts_top, W, scale, color, width_px, dx=offx, dy=offy)
        _draw_polyline_with_seam(overlay_canvas, pts_right, W, scale, color, width_px, dx=offx, dy=offy)
        _draw_polyline_with_seam(overlay_canvas, pts_bottom, W, scale, color, width_px, dx=offx, dy=offy)
        _draw_polyline_with_seam(overlay_canvas, pts_left, W, scale, color, width_px, dx=offx, dy=offy)

        # Label at approximate center (midpoint of top edge)
        u_lbl, v_lbl = pts_top[len(pts_top) // 2]
        txt_id = overlay_canvas.create_text(u_lbl * scale + offx + 6, v_lbl * scale + offy + 6, text=str(i), fill=color, anchor="nw")
        _overlay_items.append(txt_id)


def _on_refresh_overlays():
    if _selected_project is None:
        ui_log("[ERROR] Please select a folder with a video first.")
        return
    project_root = Path(_selected_project)
    if not (panorama_mode_active and _panorama_images):
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
    # Panorama-mode: copy selected panorama into overlays preview frame
    if panorama_mode_active:
        idx = 1
        if preview_pano_index_var is not None:
            try:
                idx = int(preview_pano_index_var.get() or "1")
            except Exception:
                idx = 1
        ok = _select_and_copy_panorama(project_root, idx, frame_file, context="overlays")
        if not ok:
            return
    if not panorama_mode_active:
        ui_status("Extracting panorama frame for overlays...")
        if not _extract_preview_frame(video_path, preview_time_var.get(), frame_file):
            ui_log("[ERROR] Failed to extract panorama frame for overlays.")
            return

    # Optional panorama inversion for overlays background image
    try:
        inv = bool(invert_panos_var.get()) if invert_panos_var is not None else False
    except Exception:
        inv = False
    if inv:
        ui_status("Inverting panorama (overlays) with jpegtran...")
        _rotate_panoramas_in_dir(frame_file.parent)

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
        for btn in (browse_btn, last_btn, split_btn, align_btn):
            if btn is not None:
                btn.configure(state=DISABLED)
    else:
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

                # We remove the person — keep everything else
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
    ui_status("Rendering images (render_only)" if render_only else "Running COLMAP pipeline...")

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
    if panorama_mode_active:
        ui_log("[INFO] Panorama Mode active: skipping video extraction and using frames/.")
        video_count = 0
    else:
        video_count = extract_frames_with_progress(project_root, seconds_per_frame, start_seconds, end_seconds)
    frames_root = project_root / "frames"

    # Optional panorama inversion before any split/render/mask
    try:
        inv = bool(invert_panos_var.get()) if invert_panos_var is not None else False
    except Exception:
        inv = False
    if inv:
        ui_status("Inverting panoramas with jpegtran...")
        ok_inv = _rotate_panoramas_in_dir(frames_root)
        if not ok_inv:
            ui_log("[INFO] Proceeding without inversion (jpegtran missing or failed).")

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
    ui_status("Rendering perspective images from panoramas...")
    ok = run_panorama_sfm(project_root, render_only=True)
    if not ok:
        ui_log("[ERROR] Rendering step failed. See log.")
        return SplitResult(project_root, seconds_per_frame, masking_enabled, video_count)
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
    # Detect panorama frames folder and cache list (do not auto-enable mode)
    global panorama_mode_active, _panorama_images
    ok_frames, imgs = detect_panorama_frames_dir(folder_path)
    _panorama_images = imgs if ok_frames else None
    if ok_frames:
        ui_log(f"[INFO] Detected panorama 'frames' folder with {len(imgs)} image(s).")
    else:
        ui_log("[INFO] No valid panorama 'frames' folder detected.")
    # Apply checkbox policy: enable active mode only if user opted-in and frames exist
    _update_panorama_mode_active_from_ui(log=False)
    if panorama_mode_active:
        ui_log("[INFO] Panorama Mode is ON for this folder. Video extraction will be skipped.")
    else:
        ui_log("[INFO] Panorama Mode is OFF for this folder. Using video-based workflow.")

    ui_status(f"Selected folder: {folder_path}")
    ui_log(f"[INFO] Selected folder: {folder_path}")

    refresh_action_buttons()


## Drag-and-drop handler removed


def browse_folder():
    chosen = filedialog.askdirectory(title="Select folder containing 360 video(s)")
    if chosen:
        start_pipeline_with_path(Path(chosen))


def run_last():
    if _last_folder and Path(_last_folder).exists():
        start_pipeline_with_path(Path(_last_folder))
    else:
        ui_log("[ERROR] No valid last folder. Please use Browse.")


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

def _on_panorama_mode_toggle():
    # Called when the Panorama Mode checkbox toggles
    # Enforce mutual exclusivity with video_mode_var
    try:
        if panorama_mode_var is not None and bool(panorama_mode_var.get()):
            if video_mode_var is not None:
                video_mode_var.set(False)
    except Exception:
        pass
    _update_panorama_mode_active_from_ui(log=True)

def _on_video_mode_toggle():
    # Called when the Video Mode checkbox toggles
    try:
        if video_mode_var is not None and bool(video_mode_var.get()):
            if panorama_mode_var is not None:
                panorama_mode_var.set(False)
    except Exception:
        pass
    _update_panorama_mode_active_from_ui(log=True)


def _on_toggle_inverted_preset():
    global inverted_preset_active
    try:
        inverted_preset_active = not inverted_preset_active
        # Compute which pairs to apply based on masking flag and toggle
        if bool(use_masking.get()):
            pairs = _default_pairs(masking_enabled=True)
        else:
            pairs = _default_pairs(masking_enabled=False)
        _apply_pairs_to_vars(pairs)
        ui_status("Using inverted yaw preset." if inverted_preset_active else "Using default yaw preset.")
    except Exception:
        pass


def main():
    global _root, status_var, progress_sub, log_text
    global use_masking, frame_interval, start_time_var, end_time_var, browse_btn, last_btn, split_btn, align_btn
    global export_rc_xmp
    global invert_panos_var
    global panorama_mode_var
    global video_mode_var
    global ffmpeg_path_var, jpegtran_path_var
    global preview_pano_index_var
    global yolo_model_path, yolo_conf, yolo_dilate_px, yolo_invert_mask, yolo_apply_to_rgb, _yolo_widgets

    _root = Tk()
    _root.title("360 Video Dataset Preparation")
    _root.geometry("1280x900")
    try:
        _root.state("zoomed")
    except Exception:
        pass
    _root.configure(bg=PALETTE["bg"])
    _root.resizable(True, True)

    # ---------- Header ----------
    header = Frame(_root, bg=PALETTE["bg"])
    header.pack(pady=(14, 8))

    # icon_path = Path(__file__).parent / "folder_icon.png"
    # if icon_path.exists():
    #     try:
    #         img = Image.open(icon_path).resize((84, 84))
    #         icon = ImageTk.PhotoImage(img)
    #         icon_label = Label(header, image=icon, bg=PALETTE["bg"])
    #         icon_label.image = icon
    #         icon_label.pack(side="left", padx=(0, 12))
    #     except Exception:
    #         Label(header, text="??", font=("Arial", 44), bg="black", fg="white").pack(side="left", padx=(0, 12))
    # else:
    #     Label(header, text="??", font=("Arial", 44), bg="black", fg="white").pack(side="left", padx=(0, 12))

    Label(header,
          text="360 VIDEO PREPARATION TOOL",
          bg=PALETTE["bg"], fg=PALETTE["fg"], font=("Arial", 12, "bold")).pack(side="left")

    # Ensure any fallback icon labels inherited palette (handles inline-pack case)
    try:
        for _child in header.winfo_children():
            if isinstance(_child, Label) and _child.cget("text") == "??":
                _child.configure(bg=PALETTE["bg"], fg=PALETTE["fg"])
    except Exception:
        pass

    # ---------- Buttons ----------
    btn_style = _btn_style()
    btns = Frame(_root, bg=PALETTE["bg"])
    btns.pack()
    browse_btn = Button(btns, text="Browse", command=browse_folder, **btn_style)
    browse_btn.pack(side="left", padx=6)
    last_btn = Button(btns, text="Run Last Chosen Folder", state=DISABLED, command=run_last, **btn_style)
    last_btn.pack(side="left", padx=6)

    # Current Task progress just below browse buttons
    prog_top = Frame(_root, bg=PALETTE["bg"])
    prog_top.pack(fill=BOTH, padx=16, pady=(8, 4))
    Label(prog_top, text="Current Task", bg=PALETTE["bg"], fg=PALETTE["muted_fg"]).pack(anchor="w")
    progress_sub = ttk.Progressbar(prog_top, orient="horizontal", mode="determinate", length=680)
    progress_sub.pack(pady=(2, 8))

    # ---------- Controls ----------
    # Moved to tabs per user request


    # ---------- Time Range / Stage Controls moved to Splitting tab ----------
    
    # ---------- Progress moved into Splitting tab ----------

    status_var = StringVar(value="Idle.")
    Label(_root, textvariable=status_var, bg=PALETTE["bg"], fg=PALETTE["fg"]).pack(pady=(0, 6))

    # Preset toggle moved to Settings tab

    # ---------- Tabs ----------
    _ensure_preview_vars(reset_with_defaults=True)
    tabs = ttk.Notebook(_root)
    tabs.pack(fill=BOTH, expand=True, padx=16, pady=(8, 6))

    # Settings tab (first) with grouped sections
    settings_tab = Frame(tabs, bg=PALETTE["bg"])
    tabs.add(settings_tab, text="Settings")

    # --- General group ---
    general_group = Frame(settings_tab, bg=PALETTE["bg"], highlightthickness=0)
    Label(general_group, text="General", bg=PALETTE["bg"], fg=PALETTE["fg"], font=("Arial", 11, "bold")).pack(anchor="w", padx=8, pady=(10,2))
    general_group.pack(fill=BOTH)

    # Person masking
    use_masking = BooleanVar(value=False)
    Checkbutton(general_group, text="Enable Person Masking", variable=use_masking,
                onvalue=True, offvalue=False, bg=PALETTE["bg"], fg=PALETTE["fg"],
                activebackground=PALETTE["bg"], selectcolor=PALETTE["bg"],
                command=on_masking_toggle).pack(anchor="w", padx=16, pady=(2,2))

    # XMP export
    export_rc_xmp = BooleanVar(value=False)
    Checkbutton(general_group, text="RC XMP Export XMP", variable=export_rc_xmp,
                onvalue=True, offvalue=False, bg=PALETTE["bg"], fg=PALETTE["fg"],
                activebackground=PALETTE["bg"], selectcolor=PALETTE["bg"]).pack(anchor="w", padx=16, pady=2)

    # Invert panoramas
    invert_panos_var = BooleanVar(value=False)
    Checkbutton(general_group, text="Invert panoramas", variable=invert_panos_var,
                onvalue=True, offvalue=False, bg=PALETTE["bg"], fg=PALETTE["fg"],
                activebackground=PALETTE["bg"], selectcolor=PALETTE["bg"]).pack(anchor="w", padx=16, pady=2)

    # Invert yaw values preset
    def _on_invert_yaw_var_toggle():
        # Sync global flag and update sliders
        global inverted_preset_active
        try:
            inverted_preset_active = bool(invert_yaw_var.get())
            pairs = _default_pairs(masking_enabled=bool(use_masking.get()))
            _apply_pairs_to_vars(pairs)
            ui_status("Using inverted yaw preset." if inverted_preset_active else "Using default yaw preset.")
        except Exception:
            pass

    invert_yaw_var = BooleanVar(value=False)
    Checkbutton(general_group, text="Invert yaw values", variable=invert_yaw_var,
                onvalue=True, offvalue=False, bg=PALETTE["bg"], fg=PALETTE["fg"],
                activebackground=PALETTE["bg"], selectcolor=PALETTE["bg"],
                command=_on_invert_yaw_var_toggle).pack(anchor="w", padx=16, pady=2)

    # --- Photo/Video group ---
    pv_group = Frame(settings_tab, bg=PALETTE["bg"], highlightthickness=0)
    Label(pv_group, text="Photo/Video", bg=PALETTE["bg"], fg=PALETTE["fg"], font=("Arial", 11, "bold")).pack(anchor="w", padx=8, pady=(12,2))
    pv_group.pack(fill=BOTH)

    modes_row = Frame(pv_group, bg=PALETTE["bg"]) ; modes_row.pack(anchor="w", fill=BOTH)
    panorama_mode_var = BooleanVar(value=False)
    Checkbutton(modes_row, text="Panorama Mode (use frames/)", variable=panorama_mode_var,
                onvalue=True, offvalue=False, bg=PALETTE["bg"], fg=PALETTE["fg"],
                activebackground=PALETTE["bg"], selectcolor=PALETTE["bg"],
                command=_on_panorama_mode_toggle).pack(side="left", padx=(16,12), pady=2)
    global video_mode_var
    video_mode_var = BooleanVar(value=True)
    Checkbutton(modes_row, text="Video Mode", variable=video_mode_var,
                onvalue=True, offvalue=False, bg=PALETTE["bg"], fg=PALETTE["fg"],
                activebackground=PALETTE["bg"], selectcolor=PALETTE["bg"],
                command=_on_video_mode_toggle).pack(side="left", padx=(0,12), pady=2)

    # Frame extraction controls
    fv_row = Frame(pv_group, bg=PALETTE["bg"]) ; fv_row.pack(anchor="w", fill=BOTH, pady=(2,2))
    Label(fv_row, text="Extract 1 frame per", bg=PALETTE["bg"], fg=PALETTE["fg"]).pack(side="left", padx=(16,6))
    frame_interval = StringVar(value="1")
    Entry(fv_row, textvariable=frame_interval, width=6).pack(side="left")
    Label(fv_row, text="seconds", bg=PALETTE["bg"], fg=PALETTE["fg"]).pack(side="left", padx=(6, 16))

    range_row = Frame(pv_group, bg=PALETTE["bg"]) ; range_row.pack(anchor="w", fill=BOTH, pady=(2,8))
    Label(range_row, text="Start (hh:mm:ss)", bg=PALETTE["bg"], fg=PALETTE["fg"]).pack(side="left", padx=(16,6))
    start_time_var = StringVar(value="")
    Entry(range_row, textvariable=start_time_var, width=10).pack(side="left", padx=(0, 16))
    Label(range_row, text="End (hh:mm:ss)", bg=PALETTE["bg"], fg=PALETTE["fg"]).pack(side="left", padx=(0,6))
    end_time_var = StringVar(value="")
    Entry(range_row, textvariable=end_time_var, width=10).pack(side="left")

    # --- Software group ---
    sw_group = Frame(settings_tab, bg=PALETTE["bg"], highlightthickness=0)
    Label(sw_group, text="Software", bg=PALETTE["bg"], fg=PALETTE["fg"], font=("Arial", 11, "bold")).pack(anchor="w", padx=8, pady=(12,2))
    sw_group.pack(fill=BOTH)

    paths_frame = Frame(sw_group, bg=PALETTE["bg"]) ; paths_frame.pack(fill=BOTH, pady=(2, 8))
    # jpegtran path selector
    Label(paths_frame, text="jpegtran.exe path", bg=PALETTE["bg"], fg=PALETTE["fg"]).grid(row=0, column=0, sticky="w", padx=(16,8), pady=4)
    default_jpegtran = r"C:Users/test/Downloads/jpegtran.exe"
    jpegtran_path_var = StringVar(value=default_jpegtran)
    jt_entry = Entry(paths_frame, textvariable=jpegtran_path_var, width=72)
    jt_entry.grid(row=0, column=1, sticky="we", pady=4)
    def _browse_jpegtran():
        p = filedialog.askopenfilename(title="Select jpegtran.exe", filetypes=[["Executable", "*.exe"], ["All files", "*.*"]])
        if p:
            jpegtran_path_var.set(p)
    Button(paths_frame, text="Browse", command=_browse_jpegtran, **_btn_style()).grid(row=0, column=2, padx=(8,0))

    # ffmpeg path selector
    Label(paths_frame, text="ffmpeg.exe path", bg=PALETTE["bg"], fg=PALETTE["fg"]).grid(row=1, column=0, sticky="w", padx=(16,8), pady=4)
    ffmpeg_path_var = StringVar(value="")
    ff_entry = Entry(paths_frame, textvariable=ffmpeg_path_var, width=72)
    ff_entry.grid(row=1, column=1, sticky="we", pady=4)
    def _browse_ffmpeg():
        p = filedialog.askopenfilename(title="Select ffmpeg.exe", filetypes=[["Executable", "*.exe"], ["All files", "*.*"]])
        if p:
            ffmpeg_path_var.set(p)
    Button(paths_frame, text="Browse", command=_browse_ffmpeg, **_btn_style()).grid(row=1, column=2, padx=(8,0))
    # Make grid columns stretch
    try:
        paths_frame.grid_columnconfigure(1, weight=1)
    except Exception:
        pass

    # Splitting tab (only stage buttons; settings moved to Settings > Photo/Video)
    splitting_tab = Frame(tabs, bg=PALETTE["bg"])
    tabs.add(splitting_tab, text="Splitting")

    stage_btns = Frame(splitting_tab, bg=PALETTE["bg"])
    stage_btns.pack(pady=(0, 12))
    split_btn = Button(stage_btns, text="START SPLITTING", state=DISABLED, command=on_start_split, **btn_style)
    split_btn.pack(side="left", padx=6)
    align_btn = Button(stage_btns, text="START ALIGNING", state=DISABLED, command=on_start_align, **btn_style)
    align_btn.pack(side="left", padx=6)

    # (Progress bar is displayed at the top area)

    # Preview tab
    preview_tab = Frame(tabs, bg=PALETTE["bg"])
    tabs.add(preview_tab, text="Preview")

    prev_ctrl = Frame(preview_tab, bg=PALETTE["bg"])
    prev_ctrl.pack(fill=BOTH, pady=(4, 6))
    Label(prev_ctrl, text="Preview time (HH:MM:SS)", bg=PALETTE["bg"], fg=PALETTE["fg"]).pack(side="left")
    global preview_time_entry
    preview_time_entry = Entry(prev_ctrl, textvariable=preview_time_var, width=10)
    preview_time_entry.pack(side="left", padx=(6, 12))
    Label(prev_ctrl, text="Panorama #", bg=PALETTE["bg"], fg=PALETTE["fg"]).pack(side="left", padx=(12, 6))
    preview_pano_index_var = StringVar(value="1")
    Entry(prev_ctrl, textvariable=preview_pano_index_var, width=6).pack(side="left")
    Button(prev_ctrl, text="Compute views", command=_on_compute_views, **_btn_style()).pack(side="left")

    # 3x3 preview grid
    global preview_grid
    preview_wrap = Frame(preview_tab, bg=PALETTE["bg"])
    preview_wrap.pack(fill=BOTH, pady=(6, 12))
    global preview_canvas
    preview_canvas = Canvas(preview_wrap, bg=PALETTE["canvas_bg"], highlightthickness=0, height=700)
    preview_canvas.pack(side="left", fill=BOTH, expand=True)
    # Vertical scrollbar for preview grid
    global preview_grid, hscroll, vscroll
    vscroll = ttk.Scrollbar(preview_wrap, orient="vertical", command=preview_canvas.yview)
    vscroll.pack(side="right", fill="y")
    preview_canvas.configure(yscrollcommand=vscroll.set)
    preview_grid = Frame(preview_canvas, bg=PALETTE["bg"])
    global preview_canvas_window_id
    preview_canvas_window_id = preview_canvas.create_window((0, 0), window=preview_grid, anchor="nw")
    # Update scrollregion whenever content changes
    def _on_preview_inner_configure(event=None):
        try:
            preview_canvas.configure(scrollregion=preview_canvas.bbox("all"))
        except Exception:
            pass
        _center_preview_grid()
    preview_grid.bind("<Configure>", _on_preview_inner_configure)
    # Re-center when canvas size changes
    def _on_preview_canvas_configure(event=None):
        _center_preview_grid()
    preview_canvas.bind("<Configure>", _on_preview_canvas_configure)

    def _center_preview_grid():
        try:
            cw = preview_canvas.winfo_width()
            iw = preview_grid.winfo_reqwidth()
            x = max(0, int((cw - iw) / 2))
            if preview_canvas_window_id is not None:
                preview_canvas.coords(preview_canvas_window_id, x, 0)
        except Exception:
            pass

    # Overlays tab
    overlays_tab = Frame(tabs, bg=PALETTE["bg"])
    tabs.add(overlays_tab, text="Overlays")

    ov_ctrl = Frame(overlays_tab, bg=PALETTE["bg"])
    ov_ctrl.pack(fill=BOTH, pady=(4, 6))
    Label(ov_ctrl, text="Uses Preview time", bg=PALETTE["bg"], fg=PALETTE["muted_fg"]).pack(side="left", padx=(0, 12))
    Button(ov_ctrl, text="Refresh Overlays", command=_on_refresh_overlays, **_btn_style()).pack(side="left")

    global overlay_canvas
    overlay_canvas = Canvas(overlays_tab, bg=PALETTE["canvas_bg"], highlightthickness=0, height=520)
    overlay_canvas.pack(fill=BOTH, expand=True, pady=(6, 12))

    # ---------- Log (moved below tabs to free space) ----------
    log_frame = Frame(_root, bg=PALETTE["bg"])
    log_frame.pack(fill=BOTH, expand=False, padx=16, pady=(0, 12))
    log_text = Text(log_frame, height=6, bg=PALETTE["log_bg"], fg=PALETTE["log_fg"], insertbackground=PALETTE["insert_bg"])
    log_text.pack(fill=BOTH)
    log_text.configure(state=DISABLED)

    refresh_action_buttons()
    # Ensure controls reflect current panorama mode state
    _refresh_controls_for_panorama_mode()

    # initialize YOLO controls state
    on_masking_toggle()

    _root.mainloop()

if __name__ == "__main__":
    main()


























