"""Compare two eval runs of the same subset sample by sample."""

from __future__ import annotations

from pathlib import Path

from edge_vlm.evaluate import read_records
from edge_vlm.metrics import normalize_vqa


def agreement(a: Path, b: Path) -> dict:
    ra = {r["id"]: r for r in read_records(a)}
    rb = {r["id"]: r for r in read_records(b)}
    common = sorted(set(ra) & set(rb), key=lambda x: (len(x), x))
    same_text = [i for i in common if ra[i]["pred"].strip() == rb[i]["pred"].strip()]
    same_norm = [i for i in common if normalize_vqa(ra[i]["pred"]) == normalize_vqa(rb[i]["pred"])]
    same_tokens = [i for i in common if ra[i]["input_tokens"] == rb[i]["input_tokens"]]
    diff = [(i, ra[i]["pred"], rb[i]["pred"]) for i in common if i not in same_norm]
    return {
        "n": len(common),
        "same_text": len(same_text),
        "same_normalized": len(same_norm),
        "same_input_tokens": len(same_tokens),
        "differences": diff,
    }
