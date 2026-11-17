"""Accuracy tables from the eval runs, and paired differences between runs.

Two runs of the same subset are compared question by question: the mean of
the per-sample score differences has a much tighter interval than the gap
between two independent accuracies.
"""

from __future__ import annotations

import math
import statistics
from pathlib import Path

from edge_vlm import metrics
from edge_vlm.evaluate import read_records

TASKS = ("textvqa", "mmstar", "pope")
DEVICES = ("GPU", "CPU", "NPU")


def parsed(task: str, pred: str) -> str | None:
    """The answer as the metric sees it."""
    if task == "textvqa":
        return metrics.normalize_vqa(pred)
    if task == "pope":
        return metrics.yes_no(pred)
    return metrics.option_letter(pred)


def per_sample(task: str, records: list[dict]) -> dict[str, float]:
    """VQA accuracy of each sample, or 1 / 0 for a right / wrong yes-no or option letter."""
    if task == "textvqa":
        return {r["id"]: metrics.vqa_accuracy(r["pred"], r["answers"]) for r in records}
    return {r["id"]: float(parsed(task, r["pred"]) == r["answers"][0]) for r in records}


def ci95(values: list[float]) -> float:
    """Half width of the normal 95% interval of the mean."""
    if len(values) < 2:
        return float("nan")
    return 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def paired(task: str, a: list[dict], b: list[dict]) -> dict:
    """a minus b on the samples both runs have."""
    sa, sb = per_sample(task, a), per_sample(task, b)
    common = sorted(set(sa) & set(sb))
    d = [sa[i] - sb[i] for i in common]
    pa = {r["id"]: parsed(task, r["pred"]) for r in a}
    pb = {r["id"]: parsed(task, r["pred"]) for r in b}
    return {
        "n": len(common),
        "delta": statistics.fmean(d) if d else float("nan"),
        "ci": ci95(d),
        "changed": sum(pa[i] != pb[i] for i in common),
    }


def collect(root: Path, seed: int = 0) -> dict[tuple[str, str, str, str], list[dict]]:
    """(model, format, device, task) -> records of the run with the most samples."""
    best: dict[tuple[str, str, str, str], list[dict]] = {}
    for p in sorted(root.glob(f"*/*/*/*_s{seed}.jsonl")):
        key = (p.parts[-4], p.parts[-3], p.parts[-2], p.name.split("_")[0])
        recs = read_records(p)
        if len(recs) > len(best.get(key, [])):
            best[key] = recs
    return best


def _table(head: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _pct(x: float, ci: float | None = None) -> str:
    return f"{100 * x:.1f}" if ci is None else f"{100 * x:.1f} +/- {100 * ci:.1f}"


def accuracy_table(runs: dict) -> str:
    configs = sorted({k[:3] for k in runs})
    rows = []
    for cfg in configs:
        row = list(cfg)
        for task in TASKS:
            recs = runs.get((*cfg, task))
            if recs is None:
                row.append("")
                continue
            s = list(per_sample(task, recs).values())
            row.append(f"{_pct(statistics.fmean(s), ci95(s))} ({len(s)})")
        recs = runs.get((*cfg, "mmstar"))
        # the model explains instead of giving a letter, and runs out of tokens
        row.append(_pct(statistics.fmean(parsed("mmstar", r["pred"]) is None for r in recs)) if recs else "")
        recs = runs.get((*cfg, "pope"))
        if recs:
            p = metrics.pope([r["pred"] for r in recs], [r["answers"][0] for r in recs])
            row += [_pct(p["f1"]), _pct(p["yes_ratio"])]
        else:
            row += ["", ""]
        rows.append(row)
    head = ["model", "format", "device", "TextVQA", "MMStar", "POPE acc", "MMStar no letter %", "POPE F1", "POPE yes %"]
    return _table(head, rows)


def _delta_row(task: str, a: list[dict], b: list[dict]) -> list[str]:
    d = paired(task, a, b)
    sign = "+" if d["delta"] >= 0 else ""
    return [task, str(d["n"]), sign + _pct(d["delta"], d["ci"]), f"{100 * d['changed'] / d['n']:.1f}"]


def format_table(runs: dict, refs: tuple[str, ...] = ("fp16", "int8")) -> str:
    """Each format against the reference format of the same model, on the GPU."""
    rows = []
    for model in sorted({k[0] for k in runs}):
        for task in TASKS:
            ref = next((f for f in refs if (model, f, "GPU", task) in runs), None)
            if ref is None:
                continue
            for fmt in sorted({k[1] for k in runs if k[0] == model and k[1] != ref}):
                if (model, fmt, "GPU", task) in runs:
                    a, b = runs[(model, fmt, "GPU", task)], runs[(model, ref, "GPU", task)]
                    rows.append([model, f"{fmt} vs {ref}", *_delta_row(task, a, b)])
    head = ["model", "formats", "task", "n", "delta (points)", "answers changed %"]
    return _table(head, rows)


def device_table(runs: dict) -> str:
    """Same model and format on another device, against the GPU run."""
    rows = []
    for model, fmt, device, task in sorted(runs):
        if device != "GPU" and (model, fmt, "GPU", task) in runs:
            a, b = runs[(model, fmt, device, task)], runs[(model, fmt, "GPU", task)]
            rows.append([model, fmt, f"{device} vs GPU", *_delta_row(task, a, b)])
    head = ["model", "format", "devices", "task", "n", "delta (points)", "answers changed %"]
    return _table(head, rows)


def markdown(root: Path) -> str:
    runs = collect(root)
    parts = [
        "## Accuracy (%, 95% interval, n)",
        accuracy_table(runs),
        "## Formats, paired on the same questions",
        format_table(runs),
        "## Devices, paired on the same questions",
        device_table(runs),
    ]
    return "\n\n".join(parts)
