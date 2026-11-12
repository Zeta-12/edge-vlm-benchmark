"""Support matrix from the probe records: what loads and answers on which device."""

from __future__ import annotations

import json
from pathlib import Path

from edge_vlm.probe import SIGN_TEXT

COLUMNS = ("model", "format", "device", "status", "load s", "ttft ms", "vision ms", "tok/s", "peak MB", "answer")


def load(probe_dir: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(probe_dir.glob("*.json"))]


def status(rec: dict, expected: str = SIGN_TEXT) -> str:
    if not rec["ok"]:
        return "fail"
    norm = " ".join(rec.get("text", "").split()).strip(" .\"'").lower()
    return "ok" if norm == expected.lower() else "wrong"


def row(rec: dict) -> list[str]:
    def num(key, fmt):
        v = rec.get(key)
        return format(v, fmt) if isinstance(v, float | int) and rec["ok"] else "-"

    answer = " ".join(rec.get("text", "").split())[:80] if rec["ok"] else rec.get("error", "").splitlines()[0][:60]
    return [
        rec["model"],
        rec["format"],
        rec["device"],
        status(rec),
        num("load_s", ".1f"),
        num("ttft_ms", ".0f"),
        num("embeddings_ms", ".0f"),
        num("tok_per_s", ".1f"),
        num("peak_rss_mb", ".0f"),
        answer.replace("|", "/"),
    ]


def markdown(records: list[dict]) -> str:
    order = {"CPU": 0, "GPU": 1, "NPU": 2}
    records = sorted(records, key=lambda r: (r["model"], r["format"], order.get(r["device"], 9)))
    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    lines += ["| " + " | ".join(row(r)) + " |" for r in records]
    return "\n".join(lines)
