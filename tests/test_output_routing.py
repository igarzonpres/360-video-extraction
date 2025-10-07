from pathlib import Path

import run_gui as gui


class _DummyVar:
    def __init__(self, value: str):
        self._v = value
    def get(self):
        return self._v


def test_list_subfolders_with_videos(tmp_path: Path):
    parent = tmp_path / "parent"
    (parent / "A").mkdir(parents=True)
    (parent / "B").mkdir(parents=True)
    # A has a video
    (parent / "A" / "clip.mp4").write_bytes(b"vid")
    # B has a non-video
    (parent / "B" / "note.txt").write_text("x")

    subs = gui.list_subfolders_with_videos(parent)
    names = [p.name for p in subs]
    assert names == ["A"]


def test_get_output_base_logic(tmp_path: Path, monkeypatch):
    project = tmp_path / "proj"
    project.mkdir()
    common = tmp_path / "common"
    common.mkdir()

    # No common output configured -> project
    monkeypatch.setattr(gui, "common_output_path_var", None)
    assert gui.get_output_base(project) == project

    # Common + merge
    monkeypatch.setattr(gui, "common_output_path_var", _DummyVar(str(common)))
    monkeypatch.setattr(gui, "output_mode_var", _DummyVar("merge"))
    assert gui.get_output_base(project) == common

    # Common + separate
    monkeypatch.setattr(gui, "output_mode_var", _DummyVar("separate"))
    assert gui.get_output_base(project) == common / project.name

