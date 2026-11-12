import json

import pytest

from edge_vlm import export


def snapshot(tmp_path, model_type):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    return tmp_path


def test_family_pin_and_flags(tmp_path, monkeypatch):
    monkeypatch.setenv("EVB_UV", "uv")
    cmd = export.export_cmd(snapshot(tmp_path, "qwen3_vl"), tmp_path / "out", "int4-sym-cw")
    assert cmd[:4] == ["uv", "run", "--isolated", "--no-project"]
    assert "transformers==5.0.0" in cmd
    tail = cmd[cmd.index("--task") :]
    assert tail[1] == "image-text-to-text"
    assert tail[2:-1] == ["--weight-format", "int4", "--sym", "--group-size", "-1", "--ratio", "1.0"]
    assert tail[-1] == str(tmp_path / "out")


def test_gemma4_gets_newer_transformers(tmp_path, monkeypatch):
    monkeypatch.setenv("EVB_UV", "uv")
    cmd = export.export_cmd(snapshot(tmp_path, "gemma4"), tmp_path / "out", "fp16")
    assert "transformers==5.5.4" in cmd


def test_unknown_family_or_format(tmp_path, monkeypatch):
    monkeypatch.setenv("EVB_UV", "uv")
    with pytest.raises(SystemExit, match="model_type"):
        export.export_cmd(snapshot(tmp_path, "internvl_chat"), tmp_path, "fp16")
    with pytest.raises(SystemExit, match="format"):
        export.export_cmd(snapshot(tmp_path, "qwen3_vl"), tmp_path, "int3")


def test_int4_formats_quantize_every_layer():
    for name, flags in export.FORMATS.items():
        if name.startswith("int4"):
            assert flags[flags.index("--ratio") + 1] == "1.0"
