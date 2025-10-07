"""
SfM pipeline for already pre-projected panoramic datasets.

This is a copy of panorama_sfm_descargada that skips panoramic projection.
Instead, it consumes a prepared root folder with subfolders like:

  prepared_root/
    images/pano_camera0/*.jpg
    images/pano_camera1/*.jpg
    ...
    masks/pano_camera0/*.jpg.png  (optional, same naming as original pipeline)

It keeps the rotations override JSON approach intact, and applies the same
rig configuration to the database, so the rest of the metadata generation and
SfM steps remain unchanged.
"""

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import PIL.Image
from scipy.spatial.transform import Rotation

import pycolmap
from pycolmap import logging
import json


def load_rotation_override_if_any(root: Path | None):
    """Search for rotation_override.json near the prepared dataset.

    Search order:
      1) root (e.g., prepared_root or prepared_root/images)
      2) parent of root
      3) current working directory
    Returns (pairs, ref_idx) or (None, None)
    """
    candidates = []
    if root:
        candidates.append(Path(root) / "rotation_override.json")
        candidates.append(Path(root).parent / "rotation_override.json")
    candidates.append(Path.cwd() / "rotation_override.json")

    for p in candidates:
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                pairs = data.get("pitch_yaw_pairs", None)
                ref_idx = data.get("ref_idx", 0)
                if isinstance(pairs, list) and len(pairs) > 0:
                    pairs = [(float(a), float(b)) for (a, b) in pairs]
                    ref_idx = int(ref_idx)
                    logging.info(f"Using rotation_override.json at: {p}")
                    return pairs, ref_idx
            except Exception as e:
                logging.warning(f"Failed to read {p}: {e}")
    return None, None


def get_virtual_rotations_from_pairs(pitch_yaw_pairs: Sequence[tuple[float, float]]) -> list[np.ndarray]:
    cams_from_pano_rotation = []
    for pitch_deg, yaw_deg in pitch_yaw_pairs:
        R = Rotation.from_euler("YX", [yaw_deg, pitch_deg], degrees=True).as_matrix()
        cams_from_pano_rotation.append(R)
    return cams_from_pano_rotation


def get_default_virtual_rotations() -> list[np.ndarray]:
    # Same defaults as panorama_sfm_descargada
    default_pairs = [
        (0, 90),  # Reference Pose
        (33, 0),
        (-42, 0),
        (0, 42),
        (0, -27),
        (42, 180),
        (-33, 180),
        (0, 207),
        (0, 138),
    ]
    return get_virtual_rotations_from_pairs(default_pairs)


def create_pano_rig_config(
    cams_from_pano_rotation: Sequence[np.ndarray], ref_idx: int = 0
) -> pycolmap.RigConfig:
    """Create a RigConfig with proper stereo-style outward Z-offsets.

    Mirrors panorama_sfm_descargada to keep metadata identical.
    """
    rig_cameras = []
    baseline = 0.065  # 6.5cm stereo separation

    for idx, cam_from_pano_rotation in enumerate(cams_from_pano_rotation):
        if idx == ref_idx:
            cam_from_rig = None
        else:
            cam_from_ref_rotation = (
                cam_from_pano_rotation @ cams_from_pano_rotation[ref_idx].T
            )

            # Views 1–5 = right lens, 6–10 = left lens
            side = 1 if idx <= 4 else -1
            local_offset = np.array([-baseline * side, 0, 0])
            translation = cam_from_ref_rotation @ local_offset

            cam_from_rig = pycolmap.Rigid3d(
                pycolmap.Rotation3d(cam_from_ref_rotation),
                translation
            )

        rig_cameras.append(
            pycolmap.RigConfigCamera(
                ref_sensor=(idx == ref_idx),
                image_prefix=f"pano_camera{idx}/",
                cam_from_rig=cam_from_rig,
            )
        )
    return pycolmap.RigConfig(cameras=rig_cameras)




def run(args):
    prepared_root: Path = args.prepared_path
    image_dir = prepared_root / "images"
    mask_dir = prepared_root / "masks"

    if not image_dir.exists():
        raise FileNotFoundError(f"Images folder not found: {image_dir}")
    if not any(image_dir.rglob("*.jpg")) and not any(image_dir.rglob("*.png")):
        logging.warning(f"No images found under {image_dir}.")

    database_path = args.output_path / "database.db"
    if database_path.exists():
        database_path.unlink()
    rec_path = args.output_path / "sparse"
    rec_path.mkdir(exist_ok=True, parents=True)

    # Load rotations override JSON if present
    override_pairs, override_ref_idx = load_rotation_override_if_any(image_dir)
    if override_pairs is not None:
        cams_from_pano_rotation = get_virtual_rotations_from_pairs(override_pairs)
        ref_idx = 0 if override_ref_idx is None else override_ref_idx
        logging.info(f"Loaded {len(cams_from_pano_rotation)} rotations from override (ref_idx={ref_idx}).")
    else:
        cams_from_pano_rotation = get_default_virtual_rotations()
        ref_idx = 0
        logging.info("No rotation_override.json found; using built-in defaults.")

    rig_config = create_pano_rig_config(cams_from_pano_rotation, ref_idx=ref_idx)

    # Mirror original intrinsics: create a virtual camera from prepared image size
    # Assumes prepared images are square (W=H) as produced by the original projector.
    # If sizes differ, we can adjust, but default to strictness for consistency.
    sample = None
    for ext in ("*.jpg", "*.png", "*.jpeg"):
        files = list(image_dir.rglob(ext))
        if files:
            sample = files[0]
            break
    if sample is None:
        raise FileNotFoundError(f"No images found under {image_dir} (jpg/png/jpeg).")
    with PIL.Image.open(sample) as im:
        width, height = im.size
    if width != height:
        logging.warning(f"Prepared images are not square ({width}x{height}); using min dimension for intrinsics.")
    image_size = min(width, height)
    fov_deg = 90.0
    focal = image_size / (2 * np.tan(np.deg2rad(fov_deg) / 2))
    camera = pycolmap.Camera.create(0, "PINHOLE", focal, image_size, image_size)
    for rc in rig_config.cameras:
        rc.camera = camera

    pycolmap.set_random_seed(0)

    extraction_options = pycolmap.SiftExtractionOptions()
    extraction_options.use_gpu = True
    extraction_options.gpu_index = "0"

    reader_options = {}
    if mask_dir.exists():
        reader_options["mask_path"] = mask_dir

    pycolmap.extract_features(
        database_path,
        image_dir,
        reader_options=reader_options,
        sift_options=extraction_options,
        camera_mode=pycolmap.CameraMode.PER_FOLDER,
    )

    with pycolmap.Database(database_path) as db:
        pycolmap.apply_rig_config([rig_config], db)

    matching_options = pycolmap.SiftMatchingOptions()
    matching_options.use_gpu = True
    matching_options.gpu_index = "0"

    if args.matcher == "sequential":
        seq_opts = pycolmap.SequentialMatchingOptions(loop_detection=True)
        pycolmap.match_sequential(database_path, matching_options=seq_opts, sift_options=matching_options)
    elif args.matcher == "exhaustive":
        pycolmap.match_exhaustive(database_path, sift_options=matching_options)
    elif args.matcher == "vocabtree":
        pycolmap.match_vocabtree(database_path, sift_options=matching_options)
    elif args.matcher == "spatial":
        pycolmap.match_spatial(database_path, sift_options=matching_options)
    else:
        logging.fatal(f"Unknown matcher: {args.matcher}")

    opts = pycolmap.IncrementalPipelineOptions(
        ba_refine_sensor_from_rig=False,
        ba_refine_focal_length=False,
        ba_refine_principal_point=False,
        ba_refine_extra_params=False,
    )
    recs = pycolmap.incremental_mapping(database_path, image_dir, rec_path, opts)
    for idx, rec in recs.items():
        logging.info(f"#{idx} {rec.summary()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared_path", type=Path, required=True, help="Root with pre-projected images/masks subfolders.")
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--matcher", default="sequential", choices=["sequential", "exhaustive", "vocabtree", "spatial"])
    run(parser.parse_args())
