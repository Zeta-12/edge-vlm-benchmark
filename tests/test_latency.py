from pathlib import Path

from edge_vlm.cli import run_child
from edge_vlm.data import MAX_PIXELS
from edge_vlm.latency import SIZES, markdown, summary


def test_sizes_fit_both_patch_grids():
    for w, h in SIZES:
        assert w % 28 == h % 28 == w % 32 == h % 32 == 0
    assert max(w * h for w, h in SIZES) == MAX_PIXELS


def test_summary_takes_medians():
    runs = [{"input_tokens": 600, "generated_tokens": 1, "ttft_ms": t} for t in (900.0, 1500.0, 1000.0)]
    s = summary(runs, ("ttft_ms",))
    assert s["ttft_ms"] == 1000.0
    assert s["ttft_ms_all"] == [900.0, 1500.0, 1000.0]
    assert s["input_tokens"] == 600


def test_markdown_rows():
    ok = {
        "model": "m",
        "format": "int4",
        "device": "GPU",
        "ok": True,
        "load_s": 15.2,
        "prefill": [{"ttft_ms": 500.0 * (i + 1), "input_tokens": 200 * (i + 1)} for i in range(len(SIZES))],
        "decode": {"tok_per_s": 19.84},
        "peak_rss_mb": 4096,
    }
    bad = {"model": "m", "format": "int4", "device": "NPU", "ok": False, "error": "RuntimeError: boom\nmore"}
    lines = markdown([ok, bad]).splitlines()
    assert lines[2] == "| m | int4 | GPU | 15 | 500 (200 tok) | 1000 (400 tok) | 1500 (600 tok) | 19.8 | 4.0 |"
    assert lines[3].startswith("| m | int4 | NPU | failed: RuntimeError: boom |")
    assert lines[3].count("|") == lines[2].count("|")


def test_run_child_records_a_crash(tmp_path):
    out = tmp_path / "x.json"
    rec = run_child("edge_vlm.no_such_module", Path("models/m/int4"), "NPU", [], out, timeout=60)
    assert rec["ok"] is False
    assert (rec["model"], rec["format"], rec["device"]) == ("m", "int4", "NPU")
    assert "No module named" in rec["error"]
    assert out.is_file()
