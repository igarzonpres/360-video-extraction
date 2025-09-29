from pathlib import Path

import run_gui as gui


def test_jpegtran_cmd_building():
    exe = r"C:\\Tools\\jpegtran.exe"
    src = Path(r"C:\\data\\frame.jpg")
    dst = Path(r"C:\\data\\frame.rot180.jpg")
    cmd = gui._jpegtran_cmd(exe, src, dst)
    assert cmd[:2] == [exe, "-rotate"], "exe and -rotate should be first"
    assert cmd[2] == "180", "rotation must be 180 degrees"
    assert "-copy" in cmd and "all" in cmd, "should preserve all metadata"
    # Ensure -outfile dst precedes input src
    assert "-outfile" in cmd, "-outfile must be present"
    oi = cmd.index("-outfile")
    assert cmd[oi+1] == str(dst)
    assert cmd[-1] == str(src)
