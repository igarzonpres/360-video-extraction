from pathlib import Path
import shutil

import run_gui as gui


def test_list_frames_images_sorted_by_name(tmp_path: Path):
    project_root = tmp_path
    frames = project_root / "frames"
    frames.mkdir(parents=True, exist_ok=True)

    # Create mixed-case names and extensions; non-image should be ignored
    names = [
        "b.jpg",
        "A.JPG",  # uppercase ext should be accepted (suffix.lower() check)
        "c.PNG",
        "zz.txt",
    ]
    for n in names:
        (frames / n).write_bytes(b"test")

    imgs = gui._list_frames_images_sorted(project_root)
    assert [p.name for p in imgs] == ["A.JPG", "b.jpg", "c.PNG"]

    # Selection by 1-based index should clamp and pick the sorted item
    def pick(index_1based: int) -> str:
        idx = max(1, min(index_1based, len(imgs)))
        return imgs[idx - 1].name

    assert pick(1) == "A.JPG"
    assert pick(2) == "b.jpg"
    assert pick(3) == "c.PNG"
    assert pick(99) == "c.PNG"
