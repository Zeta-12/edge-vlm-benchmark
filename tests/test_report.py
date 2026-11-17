import json

import pytest

from edge_vlm.report import collect, markdown, paired, per_sample


def rec(i, pred, answer, cat="random"):
    return {"id": str(i), "pred": pred, "answers": [answer], "meta": {"category": cat}}


def test_per_sample_scores():
    assert per_sample("pope", [rec(1, "Yes.", "yes"), rec(2, "no", "yes")]) == {"1": 1.0, "2": 0.0}
    assert per_sample("mmstar", [rec(1, "(B)", "B"), rec(2, "C", "B")]) == {"1": 1.0, "2": 0.0}
    tv = per_sample("textvqa", [{"id": "1", "pred": "Stop", "answers": ["stop"] * 10}])
    assert tv == {"1": 1.0}


def test_paired_uses_common_samples_only():
    a = [rec(1, "yes", "yes"), rec(2, "yes", "no"), rec(3, "no", "no"), rec(4, "yes", "yes")]
    b = [rec(1, "yes", "yes"), rec(2, "no", "no"), rec(3, "no", "no")]
    d = paired("pope", a, b)
    assert d["n"] == 3
    assert d["delta"] == pytest.approx(-1 / 3)
    assert d["changed"] == 1


def test_collect_keeps_the_biggest_run(tmp_path):
    def write(path, n):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(rec(i, "yes", "yes")) + "\n" for i in range(n)))

    write(tmp_path / "m" / "int4" / "GPU" / "pope_n100_s0.jsonl", 100)
    write(tmp_path / "m" / "int4" / "GPU" / "pope_n500_s0.jsonl", 500)
    write(tmp_path / "m" / "int4" / "NPU" / "pope_n100_s0.jsonl", 100)
    write(tmp_path / "m" / "int8" / "GPU" / "pope_n500_s0.jsonl", 500)
    runs = collect(tmp_path)
    assert len(runs[("m", "int4", "GPU", "pope")]) == 500
    text = markdown(tmp_path)
    assert "| m | int4 vs int8 | pope | 500 | +0.0 +/- 0.0 | 0.0 |" in text
    assert "| m | int4 | NPU vs GPU | pope | 100 | +0.0 +/- 0.0 | 0.0 |" in text
