import sys, types
import numpy as np


def _stub_module(name, attrs=None):
    mod = types.ModuleType(name)
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def setup_module(module):
    # Stub GUI-heavy deps
    _stub_module('tkinter', {
        'Label': object(), 'Entry': object(), 'StringVar': object(), 'DoubleVar': object(),
        'Frame': object(), 'Checkbutton': object(), 'BooleanVar': object(),
        'filedialog': types.SimpleNamespace(askopenfilename=lambda **k: None, askdirectory=lambda **k: None),
        'messagebox': types.SimpleNamespace(showwarning=lambda *a, **k: None),
        'Button': object(), 'Text': object(), 'Scale': object(), 'Canvas': object(),
        'END': 'end', 'BOTH': 'both', 'DISABLED': 'disabled', 'NORMAL': 'normal',
        'ttk': types.SimpleNamespace(Progressbar=object(), Notebook=object()),
    })
    _stub_module('tkinterdnd2', {'DND_FILES': object(), 'TkinterDnD': types.SimpleNamespace(Tk=object())})
    _stub_module('PIL', {})
    _stub_module('PIL.Image', {'open': lambda p: None, 'LANCZOS': 1, 'Image': type('Image', (), {})})
    _stub_module('PIL.ImageTk', {'PhotoImage': object()})
    class _DummyRot:
        @staticmethod
        def from_euler(*args, **kwargs):
            class _R:
                def as_matrix(self):
                    return [[1,0,0],[0,1,0],[0,0,1]]
            return _R()
    _stub_module('scipy', {})
    _stub_module('scipy.spatial', {})
    _stub_module('scipy.spatial.transform', {'Rotation': _DummyRot})
    _stub_module('cv2', {})


def test_build_pano_seam_mask_two_points():
    import run_gui as gui
    W, H = 100, 10
    mask = gui._build_pano_seam_mask(W, H, [10, 20])
    assert mask.shape == (H, W)
    # Columns 10..20 inclusive are black
    assert np.all(mask[:, :10] == 255)
    assert np.all(mask[:, 10:21] == 0)
    assert np.all(mask[:, 21:] == 255)


def test_build_pano_seam_mask_two_rectangles_unordered():
    import run_gui as gui
    W, H = 50, 5
    # Unordered pairs (x2 < x1) should still produce correct ranges
    mask = gui._build_pano_seam_mask(W, H, [30, 20, 5, 7])
    # First rectangle: 20..30
    # Second rectangle: 5..7
    # Verify spot checks
    assert np.all(mask[:, 0:5] == 255)
    assert np.all(mask[:, 5:8] == 0)
    assert np.all(mask[:, 8:20] == 255)
    assert np.all(mask[:, 20:31] == 0)
    assert np.all(mask[:, 31:] == 255)
