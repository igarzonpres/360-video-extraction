import os
import sys
import json
import cv2
import time
import threading
import subprocess
from pathlib import Path
from typing import NamedTuple

from tkinter import (
    Label, Entry, StringVar, DoubleVar, Frame, Checkbutton, BooleanVar,
    filedialog, Button, Text, Scale, END, BOTH, DISABLED, NORMAL
)
from tkinterdnd2 import DND_FILES, TkinterDnD
from tkinter import ttk
from PIL import Image, ImageTk

import numpy as np  # NEW: for mask processing
from typing import List, Tuple

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
    (0, 33),  # hay que bajarlo de 42 
    (0, -25),
    (42, 180),
    (-32, 180),
    (0, 205),
    (0, 142), # hay que subirlo de 138
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
progress_main = None
progress_sub = None
log_text = None

use_masking = None
frame_interval = None
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

def ui_status(msg: str):
    status_var.set(msg)
    _root.update_idletasks()

def ui_log(msg: str):
    log_text.configure(state=NORMAL)
    log_text.insert(END, msg.rstrip() + "\n")
    log_text.see(END)
    log_text.configure(state=DISABLED)
    _root.update_idletasks()

def ui_main_progress(value: float | None = None, indeterminate: bool = False):
    try:
        progress_main.stop()
    except Exception:
        pass
    if indeterminate:
        progress_main["mode"] = "indeterminate"
        progress_main.start(12)
    else:
        progress_main["mode"] = "determinate"
        progress_main["value"] = 0 if value is None else value
    _root.update_idletasks()

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
    # Build 3x3 grid
    for i, img_path in enumerate(imgs):
        r, c = divmod(i, 3)
        cell = Frame(preview_grid, bg="black")
        cell.grid(row=r, column=c, padx=4, pady=4, sticky="n")
        if img_path and img_path.exists():
            try:
                im = Image.open(img_path)
                im.thumbnail((220, 220))
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
        Label(cell, text=f"View {i}", bg="black", fg="#ccc").pack()
        yaw = _yaw_vars[i]
        pitch = _pitch_vars[i]
        Scale(cell, from_=-180, to=180, orient="horizontal", length=200, label="Yaw", variable=yaw).pack()
        Scale(cell, from_=-90, to=90, orient="horizontal", length=200, label="Pitch", variable=pitch).pack()


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

def extract_frames_with_progress(video_dir: Path, interval_seconds: float) -> int:
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

        step_frames = max(1, int(round(fps * float(interval_seconds))))
        frame_idx = 0
        saved_idx = 0

        ui_sub_progress(0, indeterminate=False)
        last_update = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step_frames == 0:
                frame_path = output_base_dir / f"{video_name}_frame_{saved_idx:05d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                saved_idx += 1

            frame_idx += 1

            if frame_count > 0 and (time.time() - last_update) > 0.05:
                ui_sub_progress(min(100.0, 100.0 * frame_idx / max(1, frame_count)), indeterminate=False)
                last_update = time.time()

        cap.release()
        ui_sub_progress(100.0, indeterminate=False)
        ui_log(f"[OK] Saved {saved_idx} frames into {output_base_dir}")

        ui_main_progress(min(100.0, 100.0 * vid_idx / total_vids), indeterminate=False)

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
# External steps (threaded helpers)
# =========================

def run_panorama_sfm(project_root: Path, render_only: bool = False) -> bool:
    wrapper = Path(__file__).parent / "run_panorama_sfm.py"
    if not wrapper.exists():
        ui_log(f"[ERROR] Missing run_panorama_sfm.py next to the GUI: {wrapper}")
        return False

    # Build wrapper command and pass optional XMP export
    cmd = [sys.executable, str(wrapper), str(project_root)]
    try:
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



def run_split_stage(project_root: Path, seconds_per_frame: float, masking_enabled: bool) -> SplitResult:
    ui_main_progress(0, indeterminate=False)
    ui_status("Preparing extraction...")
    video_count = extract_frames_with_progress(project_root, seconds_per_frame)
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

    ui_main_progress(33, indeterminate=False)
    # Render perspective images/masks (pre-indexing) via COLMAP wrapper
    ui_status("Rendering perspective images from panoramas…")
    ok = run_panorama_sfm(project_root, render_only=True)
    if not ok:
        ui_log("[ERROR] Rendering step failed. See log.")
        return SplitResult(project_root, seconds_per_frame, masking_enabled, video_count)
    return SplitResult(
        project_root=project_root,
        seconds_per_frame=seconds_per_frame,
        masking_enabled=masking_enabled,
        video_count=video_count,
    )



def run_align_stage(project_root: Path, video_count: int) -> bool:
    ui_main_progress(33, indeterminate=False)

    if not run_panorama_sfm(project_root):
        ui_status("COLMAP failed. See log.")
        return False

    ui_main_progress(66, indeterminate=False)

    delete_pano_camera0(project_root)
    ui_main_progress(90, indeterminate=False)

    if video_count > 1:
        ui_log(f"[INFO] Multiple videos detected ({video_count}). Running segment_images...")
        run_segment_images(project_root)

    ui_main_progress(100, indeterminate=False)
    ui_status("All done.")
    ui_log("[DONE] Pipeline complete.")
    return True



def _stop_progress_bars():
    try:
        progress_main.stop()
        progress_sub.stop()
    except Exception:
        pass



def _split_thread(project_root: Path, seconds_per_frame: float, masking_enabled: bool) -> None:
    global _split_result
    try:
        ui_disable_inputs(True)
        ui_sub_progress(0, indeterminate=False)
        _split_result = None
        result = run_split_stage(project_root, seconds_per_frame, masking_enabled)
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

    _split_result = None
    refresh_action_buttons()

    if split_btn is not None:
        split_btn.configure(state=DISABLED)
    if align_btn is not None:
        align_btn.configure(state=DISABLED)

    threading.Thread(
        target=_split_thread,
        args=(project_root, seconds, masking),
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
    global _root, status_var, progress_main, progress_sub, log_text
    global use_masking, frame_interval, drop_zone, browse_btn, last_btn, split_btn, align_btn
    global export_rc_xmp
    global yolo_model_path, yolo_conf, yolo_dilate_px, yolo_invert_mask, yolo_apply_to_rgb, _yolo_widgets

    _root = TkinterDnD.Tk()
    _root.title("360 Video Dataset Preparation")
    _root.geometry("780x720")
    _root.configure(bg="black")
    _root.resizable(False, False)

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
            Label(header, text="ðŸ“", font=("Arial", 44), bg="black", fg="white").pack(side="left", padx=(0, 12))
    else:
        Label(header, text="ðŸ“", font=("Arial", 44), bg="black", fg="white").pack(side="left", padx=(0, 12))

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
    btns = Frame(_root, bg="black")
    btns.pack()
    browse_btn = Button(btns, text="Browseâ€¦", command=browse_folder)
    browse_btn.pack(side="left", padx=6)
    last_btn = Button(btns, text="Run Last Chosen Folder", state=DISABLED, command=run_last)
    last_btn.pack(side="left", padx=6)

    # ---------- Progress ----------
    prog = Frame(_root, bg="black")
    prog.pack(fill=BOTH, padx=16, pady=(12, 4))

    Label(prog, text="Overall Progress", bg="black", fg="#ccc").pack(anchor="w")
    progress_main = ttk.Progressbar(prog, orient="horizontal", mode="determinate", length=680)
    progress_main.pack(pady=(2, 8))

    Label(prog, text="Current Task", bg="black", fg="#ccc").pack(anchor="w")
    progress_sub = ttk.Progressbar(prog, orient="horizontal", mode="determinate", length=680)
    progress_sub.pack(pady=(2, 8))

    status_var = StringVar(value="Idle.")
    Label(_root, textvariable=status_var, bg="black", fg="white").pack(pady=(0, 6))

    # ---------- Log ----------
    log_frame = Frame(_root, bg="black")
    log_frame.pack(fill=BOTH, expand=True, padx=16, pady=(0, 12))
    log_text = Text(log_frame, height=12, bg="#111", fg="#ddd", insertbackground="white")
    log_text.pack(fill=BOTH, expand=True)
    log_text.configure(state=DISABLED)

    # ---------- Preview Controls ----------
    _ensure_preview_vars(reset_with_defaults=True)
    prev_ctrl = Frame(_root, bg="black")
    prev_ctrl.pack(fill=BOTH, padx=16, pady=(8, 6))
    Label(prev_ctrl, text="Preview time (HH:MM:SS)", bg="black", fg="white").pack(side="left")
    Entry(prev_ctrl, textvariable=preview_time_var, width=10).pack(side="left", padx=(6, 12))
    Button(prev_ctrl, text="Compute views", command=_on_compute_views).pack(side="left")

    # 3x3 preview grid
    global preview_grid
    preview_grid = Frame(_root, bg="black")
    preview_grid.pack(fill=BOTH, padx=16, pady=(6, 12))

    # ---------- Stage Controls ----------
    stage_btns = Frame(_root, bg="black")
    stage_btns.pack(pady=(0, 12))
    split_btn = Button(stage_btns, text="START spltting", state=DISABLED, command=on_start_split)
    split_btn.pack(side="left", padx=6)
    align_btn = Button(stage_btns, text="START ALIGNING", state=DISABLED, command=on_start_align)
    align_btn.pack(side="left", padx=6)

    refresh_action_buttons()

    # initialize YOLO controls state
    on_masking_toggle()

    _root.mainloop()

if __name__ == "__main__":
    main()
