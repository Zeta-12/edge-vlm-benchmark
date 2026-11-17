"""Latency of one exported VLM on one device, with a fixed protocol.

Time to first token at three fixed image sizes, so every model of a family
sees the same number of visual tokens, then decode speed on a long answer.
Each config runs in its own process like the probe:
`python -m edge_vlm.latency <model_dir> <device> <out.json>`.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
import traceback
from pathlib import Path

# multiples of both 28 (Qwen2.5-VL) and 32 (Qwen3-VL) px, so no model pads them;
# the largest one is exactly data.MAX_PIXELS
SIZES = [(448, 448), (896, 672), (1120, 896)]
N_IMAGES = 5
PROMPT = "What is the main text in this image?\nAnswer the question using a single word or phrase."
DECODE_PROMPT = "Describe this image in detail."
DECODE_SIZE = (896, 672)
DECODE_TOKENS = 128
DECODE_REPEATS = 3


def images(n: int = N_IMAGES):
    """The first n TextVQA images of the eval order (real photos, content doesn't matter much)."""
    from edge_vlm import data

    return [s.pil() for s in data.subset(data.load("textvqa"), n, seed=0)]


def on_ac() -> bool | None:
    import psutil

    battery = psutil.sensors_battery()
    return None if battery is None else bool(battery.power_plugged)


def _metrics(res) -> dict:
    from edge_vlm.evaluate import _mean

    pm = res.perf_metrics
    return {
        "input_tokens": int(_mean(pm.get_num_input_tokens())),
        "generated_tokens": int(_mean(pm.get_num_generated_tokens())),
        "ttft_ms": _mean(pm.get_ttft()),
        "embeddings_ms": _mean(pm.get_prepare_embeddings_duration()),
        "tpot_ms": _mean(pm.get_tpot()),
        "tok_per_s": _mean(pm.get_throughput()),
    }


def summary(runs: list[dict], keys: tuple[str, ...]) -> dict:
    out = {"input_tokens": runs[0]["input_tokens"], "generated_tokens": runs[0]["generated_tokens"]}
    for k in keys:
        out[k] = statistics.median(r[k] for r in runs)
        out[k + "_all"] = [round(r[k], 1) for r in runs]
    return out


def run(model_dir: Path, device: str) -> dict:
    import openvino as ov
    import openvino_genai as ov_genai
    import psutil
    from PIL import Image

    from edge_vlm.evaluate import DEVICE_CONFIG, keep_awake, to_tensor

    config = DEVICE_CONFIG.get(device, {})
    rec = {
        "model": model_dir.parent.name,
        "format": model_dir.name,
        "device": device,
        "config": config,
        "openvino": ov.__version__,
        "on_ac": on_ac(),
        "free_mb_before": round(psutil.virtual_memory().available / 2**20),
        "ok": False,
    }
    keep_awake(True)
    try:
        # the pipeline reuses the KV cache of an identical prefix (same image), which makes a
        # repeated image 3-4x faster; so the warmup image is never timed and no image repeats
        *imgs, warm = images(N_IMAGES + 1)
        t0 = time.perf_counter()
        pipe = ov_genai.VLMPipeline(str(model_dir), device, **config)
        rec["load_s"] = time.perf_counter() - t0

        first = ov_genai.GenerationConfig()
        first.max_new_tokens = 1
        first.do_sample = False
        rec["prefill"] = []
        for size in SIZES:
            tensors = [to_tensor(img.resize(size, Image.Resampling.BICUBIC)) for img in imgs]
            warmup = to_tensor(warm.resize(size, Image.Resampling.BICUBIC))
            pipe.generate(PROMPT, images=[warmup], generation_config=first)  # new shape
            runs = [_metrics(pipe.generate(PROMPT, images=[t], generation_config=first)) for t in tensors]
            rec["prefill"].append({"size": list(size), **summary(runs, ("ttft_ms", "embeddings_ms"))})

        long = ov_genai.GenerationConfig()
        long.max_new_tokens = DECODE_TOKENS
        long.ignore_eos = True
        long.do_sample = False
        runs = []
        for img in imgs[:DECODE_REPEATS]:
            t = to_tensor(img.resize(DECODE_SIZE, Image.Resampling.BICUBIC))
            runs.append(_metrics(pipe.generate(DECODE_PROMPT, images=[t], generation_config=long)))
        rec["decode"] = {"size": list(DECODE_SIZE), **summary(runs, ("ttft_ms", "tpot_ms", "tok_per_s"))}
        rec["ok"] = True
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {str(e)[:2000]}"
        rec["traceback"] = traceback.format_exc()[-4000:]
    finally:
        keep_awake(False)
    mem = psutil.Process().memory_info()
    rec["peak_rss_mb"] = getattr(mem, "peak_wset", mem.rss) / 2**20
    return rec


def load(folder: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.json"))]


def markdown(records: list[dict]) -> str:
    head = ["model", "format", "device", "load s"]
    head += [f"TTFT {w}x{h} ms" for w, h in SIZES]
    head += ["decode tok/s", "peak RAM GB"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in records:
        row = [r["model"], r["format"], r["device"]]
        if not r["ok"]:
            row += ["failed: " + r["error"].splitlines()[0][:60]] + [""] * (len(head) - 4)
        else:
            row.append(f"{r['load_s']:.0f}")
            row += [f"{p['ttft_ms']:.0f} ({p['input_tokens']} tok)" for p in r["prefill"]]
            row.append(f"{r['decode']['tok_per_s']:.1f}")
            row.append(f"{r['peak_rss_mb'] / 1024:.1f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    model_dir, device, out = (argv or sys.argv[1:])[:3]
    rec = run(Path(model_dir), device)
    Path(out).write_text(json.dumps(rec, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
