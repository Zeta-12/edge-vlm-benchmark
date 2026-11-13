"""Run one (model, format, device) on a task subset, one jsonl line per sample.

The jsonl is appended as we go, so a crashed or interrupted run resumes where
it stopped.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from edge_vlm.data import Sample

log = logging.getLogger(__name__)

MAX_NEW_TOKENS = {"textvqa": 16, "pope": 8, "mmstar": 8}


def done_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def to_tensor(sample: Sample):
    import numpy as np
    import openvino as ov

    return ov.Tensor(np.asarray(sample.pil(), dtype=np.uint8)[None])


def _mean(v):
    return float(v.mean) if hasattr(v, "mean") else float(v)


def keep_awake(on: bool) -> None:
    """Stop Windows from idle-sleeping while this process runs (no system setting is changed)."""
    import ctypes
    import sys

    if sys.platform != "win32":
        return
    es_continuous, es_system_required = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | (es_system_required if on else 0))


def memory_mb() -> dict:
    import psutil

    return {
        "rss_mb": round(psutil.Process().memory_info().rss / 2**20),
        "free_mb": round(psutil.virtual_memory().available / 2**20),
    }


def run(model_dir: Path, device: str, task: str, samples: list[Sample], out: Path, config: dict | None = None) -> Path:
    import openvino_genai as ov_genai

    todo = [s for s in samples if s.id not in done_ids(out)]
    if not todo:
        log.info("%s: nothing to do", out)
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    pipe = ov_genai.VLMPipeline(str(model_dir), device, **(config or {}))
    log.info("%s on %s loaded in %.1f s, %d samples to go", model_dir, device, time.perf_counter() - t0, len(todo))

    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = MAX_NEW_TOKENS[task]
    cfg.do_sample = False

    keep_awake(True)
    t_start = time.perf_counter()
    try:
        _loop(pipe, cfg, todo, out, t_start)
    finally:
        keep_awake(False)
    return out


def _loop(pipe, cfg, todo: list[Sample], out: Path, t_start: float) -> None:
    with open(out, "a", encoding="utf-8") as f:
        for i, s in enumerate(todo, 1):
            res = pipe.generate(s.prompt, images=[to_tensor(s)], generation_config=cfg)
            pm = res.perf_metrics
            rec = {
                "id": s.id,
                "pred": res.texts[0],
                "answers": s.answers,
                "meta": s.meta,
                "input_tokens": int(_mean(pm.get_num_input_tokens())),
                "generated_tokens": int(_mean(pm.get_num_generated_tokens())),
                "ttft_ms": _mean(pm.get_ttft()),
                "generate_ms": _mean(pm.get_generate_duration()),
                "embeddings_ms": _mean(pm.get_prepare_embeddings_duration()),
                **memory_mb(),
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 10 == 0 or i == len(todo):
                rate = (time.perf_counter() - t_start) / i
                eta = rate * (len(todo) - i) / 60
                mem = memory_mb()
                log.info(
                    "%s: %d/%d, %.2f s/sample, eta %.0f min, rss %d MB, free %d MB",
                    out.name,
                    i,
                    len(todo),
                    rate,
                    eta,
                    mem["rss_mb"],
                    mem["free_mb"],
                )
