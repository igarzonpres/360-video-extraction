# delete_pano0.py
import shutil
import sys
from pathlib import Path

def delete_all_pano_camera0(source_dir: Path):
    deleted_folders = 0
    for cam0_folder in source_dir.glob("**/pano_camera0"):
        if cam0_folder.is_dir():
            print(f"🗑️  Deleting folder: {cam0_folder}")
            shutil.rmtree(cam0_folder)
            deleted_folders += 1
    if deleted_folders == 0:
        print("ℹ️  No 'pano_camera0' folders found.")
    else:
        print(f"✅ Deleted {deleted_folders} 'pano_camera0' folder(s).")

if __name__ == "__main__":
    # If an argument is provided, try to interpret it as an output root
    # Preferred: <output_root>/images
    # Fallback: treat as project_root and use <project_root>/output/images
    if len(sys.argv) >= 2:
        root = Path(sys.argv[1]).resolve()
        if (root / "images").exists():
            source = root / "images"
        else:
            source = root / "output" / "images"
    else:
        # Fallback: current-dir/output/images (original behavior)
        source = Path("output/images").resolve()
    delete_all_pano_camera0(source)
