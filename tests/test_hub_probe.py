from PIL import Image

from edge_vlm.cli import build_parser
from edge_vlm.hub import missing_files
from edge_vlm.probe import make_sign


def test_missing_files_checks_presence_and_size(tmp_path):
    (tmp_path / "a.json").write_text("{}")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.bin").write_bytes(b"123")
    expected = {"a.json": 2, "sub/b.bin": 4, "c.txt": None}
    assert missing_files(tmp_path, expected) == ["sub/b.bin", "c.txt"]
    (tmp_path / "sub" / "b.bin").write_bytes(b"1234")
    (tmp_path / "c.txt").write_text("x")
    assert missing_files(tmp_path, expected) == []


def test_sign_image(tmp_path):
    path = make_sign(tmp_path / "x" / "sign.png")
    img = Image.open(path)
    assert img.size == (640, 480)
    assert img.getpixel((320, 120)) == (250, 200, 0)


def test_probe_defaults():
    args = build_parser().parse_args(["probe", "models/m/fp16"])
    assert args.devices == ["CPU", "GPU", "NPU"]
    assert args.image is None


def test_matrix_status_and_table():
    from edge_vlm import matrix

    good = {"model": "m", "format": "fp16", "device": "NPU", "ok": True, "text": " Road works 300 m.", "load_s": 1.0}
    wrong = {**good, "device": "CPU", "text": "ROAD WORK"}
    failed = {"model": "m", "format": "fp16", "device": "GPU", "ok": False, "error": "RuntimeError: boom\nmore"}
    assert [matrix.status(r) for r in (good, wrong, failed)] == ["ok", "wrong", "fail"]
    lines = matrix.markdown([good, failed, wrong]).splitlines()
    assert [line.split(" | ")[2] for line in lines[2:]] == ["CPU", "GPU", "NPU"]
    assert "RuntimeError: boom" in lines[3] and "more" not in lines[3]


def test_matrix_answer_stays_on_one_line():
    from edge_vlm import matrix

    rec = {"model": "m", "format": "int4", "device": "NPU", "ok": True, "text": "2 1\n 3 | 300 m\n\n"}
    assert matrix.row(rec)[-1] == "2 1 3 / 300 m"
