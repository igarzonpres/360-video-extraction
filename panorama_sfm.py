"""
An example for running incremental SfM on 360 spherical panorama images.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

import cv2
import re
import shutil
import numpy as np
import PIL.ExifTags
import PIL.Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import pycolmap
from pycolmap import logging
import json

def load_rotation_override_if_any(input_image_path: Path | None):
    """
    Look for rotation_override.json written by the GUI.
    Search order:
      1) input_image_path (e.g. .../frames)
      2) parent of input_image_path (the folder you dropped into the GUI)
      3) current working directory
    Returns (pairs, ref_idx) or (None, None)
    """
    candidates = []
    if input_image_path:
        candidates.append(Path(input_image_path) / "rotation_override.json")
        candidates.append(Path(input_image_path).parent / "rotation_override.json")
    candidates.append(Path.cwd() / "rotation_override.json")

    for p in candidates:
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                pairs = data.get("pitch_yaw_pairs", None)
                ref_idx = data.get("ref_idx", 0)
                if isinstance(pairs, list) and len(pairs) > 0:
                    # Coerce to floats & ints defensively
                    pairs = [(float(a), float(b)) for (a, b) in pairs]
                    ref_idx = int(ref_idx)
                    logging.info(f"Using rotation_override.json at: {p}")
                    return pairs, ref_idx
            except Exception as e:
                logging.warning(f"Failed to read {p}: {e}")
    return None, None



def create_virtual_camera(
    pano_height: int, fov_deg: float = 90
) -> pycolmap.Camera:
    """Create a virtual perspective camera (SIMPLE_PINHOLE).

    This pycolmap build expects a single focal parameter for SIMPLE_PINHOLE
    and centers the principal point by default.
    """
    image_size = int(pano_height * fov_deg / 180)
    focal = image_size / (2 * np.tan(np.deg2rad(fov_deg) / 2))
    return pycolmap.Camera.create(0, "SIMPLE_PINHOLE", float(focal), image_size, image_size)


def get_virtual_camera_rays(camera: pycolmap.Camera) -> np.ndarray:
    size = (camera.width, camera.height)
    y, x = np.indices(size).astype(np.float32)
    xy = np.column_stack([x.ravel(), y.ravel()])
    # The center of the upper left most pixel has coordinate (0.5, 0.5)
    xy += 0.5
    xy_norm = camera.cam_from_img(xy)
    rays = np.concatenate([xy_norm, np.ones_like(xy_norm[:, :1])], -1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays


def spherical_img_from_cam(image_size, rays_in_cam: np.ndarray) -> np.ndarray:
    """Project rays into a 360 panorama (spherical) image."""
    if image_size[0] != image_size[1] * 2:
        raise ValueError("Only 360° panoramas are supported.")
    if rays_in_cam.ndim != 2 or rays_in_cam.shape[1] != 3:
        raise ValueError(f"{rays_in_cam.shape=} but expected (N,3).")
    r = rays_in_cam.T
    yaw = np.arctan2(r[0], r[2])
    pitch = -np.arctan2(r[1], np.linalg.norm(r[[0, 2]], axis=0))
    u = (1 + yaw / np.pi) / 2
    v = (1 - pitch * 2 / np.pi) / 2
    return np.stack([u, v], -1) * image_size


def get_virtual_rotations() -> Sequence[np.ndarray]:
    """Custom virtual camera rotations defined by exact pitch/yaw angles."""
    pitch_yaw_pairs = [
        (0, 90), #Reference Pose
        (42, 0),
        (-42, 0),
        (0, 42),
        (0, -42),
        (42, 180),
        (-42, 180),
        (0, 222),
        (0, 138),
        
    ]
    cams_from_pano_r = []
    for pitch_deg, yaw_deg in pitch_yaw_pairs:
        cam_from_pano_r = Rotation.from_euler(
            "YX", [yaw_deg, pitch_deg], degrees=True
        ).as_matrix()
        cams_from_pano_r.append(cam_from_pano_r)
    return cams_from_pano_r


def create_pano_rig_config(
    cams_from_pano_rotation: Sequence[np.ndarray], ref_idx: int = 0
) -> pycolmap.RigConfig:
    """Create a RigConfig with proper stereo-style outward Z-offsets."""
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



def _wrap_angle_deg(a: float) -> float:
    return ((a + 180.0) % 360.0) - 180.0


def render_perspective_images(
    pano_image_names: Sequence[str],
    pano_image_dir: Path,
    output_image_dir: Path,
    mask_dir: Path,
    override_pairs: list[tuple[float, float]] | None = None,
    override_ref_idx: int | None = None,
    export_xmp: bool = False,
) -> pycolmap.RigConfig:
    # Build camera rotations
    if override_pairs is not None:
        cams_from_pano_rotation = []
        used_pairs: list[tuple[float, float]] = []
        for pitch_deg, yaw_deg in override_pairs:
            R = Rotation.from_euler("YX", [yaw_deg, pitch_deg], degrees=True).as_matrix()
            cams_from_pano_rotation.append(R)
            used_pairs.append((float(pitch_deg), float(yaw_deg)))
        ref_idx = 0 if override_ref_idx is None else override_ref_idx
        logging.info(f"Loaded {len(cams_from_pano_rotation)} rotations from override (ref_idx={ref_idx}).")
    else:
        # Use built-in rotations and record their source angles
        built_in_pairs = [
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
        cams_from_pano_rotation = [
            Rotation.from_euler("YX", [yaw_deg, pitch_deg], degrees=True).as_matrix()
            for (pitch_deg, yaw_deg) in built_in_pairs
        ]
        used_pairs = built_in_pairs
        ref_idx = 0  # your current default
        logging.info(f"Using built-in get_virtual_rotations() (ref_idx={ref_idx}).")

    rig_config = create_pano_rig_config(cams_from_pano_rotation, ref_idx=ref_idx)

    def write_xmp_sidecar(image_path: Path, pitch_deg: float, yaw_deg: float, camera: pycolmap.Camera):
        """Write a minimal XMP sidecar with yaw/pitch/roll and intrinsics for RealityCapture.
        - Uses GPano Pose* degrees as generic orientation hints
        - Adds aux:FocalLength in pixels
        """
        # Derive pose directly from the known render angles
        heading = _wrap_angle_deg(-float(yaw_deg))
        pitch_out = float(pitch_deg)
        roll_out = 0.0

        # Robustly get focal in pixels from camera params across models
        try:
            params = np.array(camera.params, dtype=float)
        except Exception:
            params = None
        if params is not None and params.size >= 2:
            fx, fy = float(params[0]), float(params[1])
            focal = float((fx + fy) / 2.0)
        elif params is not None and params.size >= 1:
            focal = float(params[0])
        else:
            # Fallback: try attribute if available, else set to 0
            focal = float(getattr(camera, "focal_length", 0.0) or 0.0)

        width = int(camera.width)
        height = int(camera.height)

        xmp = f"""<?xpacket begin='﻿' id='W5M0MpCehiHzreSzNTczkc9d'?>
<x:xmpmeta xmlns:x='adobe:ns:meta/'>
 <rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>
  <rdf:Description xmlns:GPano='http://ns.google.com/photos/1.0/panorama/'>
   <GPano:PoseHeadingDegrees>{heading:.6f}</GPano:PoseHeadingDegrees>
   <GPano:PosePitchDegrees>{pitch_out:.6f}</GPano:PosePitchDegrees>
   <GPano:PoseRollDegrees>{roll_out:.6f}</GPano:PoseRollDegrees>
  </rdf:Description>
  <rdf:Description xmlns:aux='http://ns.adobe.com/exif/1.0/aux/'>
   <aux:FocalLength>{focal:.6f}</aux:FocalLength>
  </rdf:Description>
  <rdf:Description xmlns:tiff='http://ns.adobe.com/tiff/1.0/'>
   <tiff:ImageWidth>{width}</tiff:ImageWidth>
   <tiff:ImageLength>{height}</tiff:ImageLength>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end='w'?>
"""
        try:
            image_path.with_suffix('.xmp').write_text(xmp, encoding='utf-8')
        except Exception:
            pass


    # We assign each pano pixel to the virtual camera with the closest center.
    cam_centers_in_pano = np.einsum(
        "nij,i->nj", cams_from_pano_rotation, [0, 0, 1]
    )

    camera = pano_size = rays_in_cam = None
    mask_mappings: list[tuple[str, str]] = []  # (image_name, absolute mask path)
    for pano_name in tqdm(pano_image_names):
        pano_path = pano_image_dir / pano_name
        try:
            pano_image = PIL.Image.open(pano_path)
        except PIL.Image.UnidentifiedImageError:
            logging.info(f"Skipping file {pano_path} as it cannot be read.")
            continue
        pano_exif = pano_image.getexif()
        pano_image = np.asarray(pano_image)
        gpsonly_exif = PIL.Image.Exif()
        gpsonly_exif[PIL.ExifTags.IFD.GPSInfo] = pano_exif.get_ifd(
            PIL.ExifTags.IFD.GPSInfo
        )

        pano_height, pano_width, *_ = pano_image.shape
        if pano_width != pano_height * 2:
            raise ValueError("Only 360° panoramas are supported.")

        if camera is None:  # First image.
            camera = create_virtual_camera(pano_height)
            for rig_camera in rig_config.cameras:
                rig_camera.camera = camera
            pano_size = (pano_width, pano_height)
            rays_in_cam = get_virtual_camera_rays(camera)  # Precompute.
        else:
            if (pano_width, pano_height) != pano_size:
                raise ValueError(
                    "Panoramas of different sizes are not supported."
                )

        for cam_idx, cam_from_pano_r in enumerate(cams_from_pano_rotation):
            rays_in_pano = rays_in_cam @ cam_from_pano_r
            xy_in_pano = spherical_img_from_cam(pano_size, rays_in_pano)
            xy_in_pano = xy_in_pano.reshape(
                camera.width, camera.height, 2
            ).astype(np.float32)
            xy_in_pano -= 0.5  # COLMAP to OpenCV pixel origin.
            image = cv2.remap(
                pano_image,
                *np.moveaxis(xy_in_pano, -1, 0),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_WRAP,
            )
            # We define a mask such that each pixel of the panorama has its
            # features extracted only in a single virtual camera.
            closest_camera = np.argmax(rays_in_pano @ cam_centers_in_pano.T, -1)
            mask = (
                ((closest_camera == cam_idx) * 255)
                .astype(np.uint8)
                .reshape(camera.width, camera.height)
            )

            image_name = rig_config.cameras[cam_idx].image_prefix + pano_name
            # Prefix file name with numeric subfolder (or folder name) to ensure uniqueness.
            _img_rel = Path(image_name)
            folder = _img_rel.parent.name
            m = re.search(r"(\d+)$", folder)
            folder_prefix = m.group(1) if m else folder
            prefixed_name = f"{folder_prefix}_{_img_rel.name}"
            image_name = str(_img_rel.parent / prefixed_name)

            # Build mask name as <image_base>.mask.png, avoiding double extensions like .jpg.png
            _img_rel = Path(image_name)
            mask_rel = _img_rel.parent / f"{_img_rel.stem}.mask.png"

            image_path = output_image_dir / image_name
            image_path.parent.mkdir(exist_ok=True, parents=True)
            PIL.Image.fromarray(image).save(image_path, exif=gpsonly_exif)
            if export_xmp:
                pitch_deg, yaw_deg = used_pairs[cam_idx]
                write_xmp_sidecar(image_path, float(pitch_deg), float(yaw_deg), camera)

            mask_path = mask_dir / mask_rel
            mask_path.parent.mkdir(exist_ok=True, parents=True)
            if not pycolmap.Bitmap.from_array(mask).write(mask_path):
                raise RuntimeError(f"Cannot write {mask_path}")
            # Also write legacy COLMAP mask naming into output/colmap_masks/<image_name>.png
            legacy_mask_root = mask_dir.parent / "colmap_masks"
            legacy_mask_path = legacy_mask_root / f"{image_name}.png"
            legacy_mask_path.parent.mkdir(exist_ok=True, parents=True)
            if not pycolmap.Bitmap.from_array(mask).write(legacy_mask_path):
                raise RuntimeError(f"Cannot write {legacy_mask_path}")

            # Record mapping: image relative name (as used by COLMAP) -> mask absolute path (new-style only)
            mask_mappings.append((image_name, str(mask_path.resolve())))

    # Write mask list file for optional consumers (not used by our pycolmap build)
    mask_list_path = mask_dir / "mask_list.txt"
    try:
        with open(mask_list_path, "w", encoding="utf-8") as f:
            for img_name, mpath in mask_mappings:
                f.write(f"{img_name} {mpath}\n")
    except Exception:
        pass

    return rig_config


def run(args):
    image_dir = args.output_path / "images"
    mask_dir = args.output_path / "masks"
    image_dir.mkdir(exist_ok=True, parents=True)
    mask_dir.mkdir(exist_ok=True, parents=True)

    database_path = args.output_path / "database.db"
    if database_path.exists():
        database_path.unlink()
    rec_path = args.output_path / "sparse"
    rec_path.mkdir(exist_ok=True, parents=True)

    # --- gather input panos ---
    pano_image_dir = args.input_image_path
    pano_image_names = sorted(
        p.relative_to(pano_image_dir).as_posix()
        for p in pano_image_dir.rglob("*")
        if not p.is_dir()
    )
    logging.info(f"Found {len(pano_image_names)} images in {pano_image_dir}.")

    # --- NEW: try rotation_override.json (from GUI) ---
    override_pairs, override_ref_idx = load_rotation_override_if_any(pano_image_dir)
    if override_pairs is not None:
        logging.info(f"Using rotation_override: {len(override_pairs)} views (ref_idx={override_ref_idx}).")
    else:
        logging.info("No rotation_override.json found; using built-in get_virtual_rotations().")

    # --- pass override into renderer ---
    rig_config = render_perspective_images(
        pano_image_names,
        pano_image_dir,
        image_dir,
        mask_dir,
        override_pairs=override_pairs,
        override_ref_idx=override_ref_idx,
        export_xmp=bool(getattr(args, 'export_rc_xmp', False)),
    )

    # --- Flatten images and masks for RealityCapture ---
    try:
        rc_dir = args.output_path / "RC"
        rc_dir.mkdir(exist_ok=True, parents=True)
        img_count = 0
        mask_count = 0
        # Copy images (flat)
        for p in image_dir.rglob("*"):
            if p.is_file():
                shutil.copy2(p, rc_dir / p.name)
                img_count += 1
        # Copy masks (flat)
        for p in mask_dir.rglob("*"):
            if p.is_file():
                shutil.copy2(p, rc_dir / p.name)
                mask_count += 1
        logging.info(f"RC export: copied {img_count} images and {mask_count} masks into {rc_dir}")
    except Exception as e:
        logging.warning(f"RC export step failed: {e}")

    pycolmap.set_random_seed(0)


    extraction_options = pycolmap.SiftExtractionOptions()
    extraction_options.use_gpu = True
    extraction_options.gpu_index = "0"

    # Use legacy mask_path pointing to output/colmap_masks for compatibility
    pycolmap.extract_features(
        database_path,
        image_dir,
        reader_options={"mask_path": args.output_path / "colmap_masks"},
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
    parser.add_argument("--input_image_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--matcher", default="sequential", choices=["sequential", "exhaustive", "vocabtree", "spatial"])
    parser.add_argument("--export_rc_xmp", action="store_true", help="Write XMP sidecars with yaw/pitch/roll and intrinsics")
    run(parser.parse_args())
