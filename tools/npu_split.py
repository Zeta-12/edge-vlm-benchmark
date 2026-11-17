"""TextVQA on the NPU with the vision sub-models moved to another device.

Tells whether the NPU accuracy drop on TextVQA comes from the vision encoder
or from the LLM. Relies on MODEL_PROPERTIES (per sub-model properties), which
VLMPipeline 2026.4 reads but doesn't document, so a one-sample check runs
first: if the vision really moved, the embedding time drops.

    PYTHONPATH=src python tools/npu_split.py models/qwen3-vl-2b/int4-sym-cw 100 GPU
"""

import json
import sys
import time
from pathlib import Path

from edge_vlm import data, evaluate, metrics

VISION = ("vision_embeddings", "vision_embeddings_merger", "vision_embeddings_pos")


def check(model_dir: Path, config: dict, sample) -> dict:
    import openvino_genai as ov_genai

    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = 16
    t0 = time.perf_counter()
    pipe = ov_genai.VLMPipeline(str(model_dir), "NPU", **{**evaluate.DEVICE_CONFIG["NPU"], **config})
    load_s = time.perf_counter() - t0
    img = [evaluate.to_tensor(sample.pil())]
    pipe.generate(sample.prompt, images=img, generation_config=cfg)
    res = pipe.generate(sample.prompt, images=img, generation_config=cfg)
    pm = res.perf_metrics
    return {
        "load_s": load_s,
        "embeddings_ms": pm.get_prepare_embeddings_duration().mean,
        "ttft_ms": pm.get_ttft().mean,
        "pred": res.texts[0],
    }


def main(model_dir: str, n: str = "100", vision_device: str = "GPU") -> None:
    model_dir = Path(model_dir)
    samples = data.subset(data.load("textvqa"), int(n), 0)
    base = check(model_dir, {}, samples[0])
    print("NPU only:", base, flush=True)
    config = {"MODEL_PROPERTIES": {role: {"DEVICE": vision_device} for role in VISION}}
    try:
        split = check(model_dir, config, samples[0])
    except Exception as e:
        print(f"MODEL_PROPERTIES rejected: {type(e).__name__}: {str(e)[:1000]}", flush=True)
        return
    print(f"vision on {vision_device}:", split, flush=True)
    # 2026.4 accepts the property but keeps the vision on the NPU (6.1 s vs 5.8 s)
    if split["embeddings_ms"] > 0.7 * base["embeddings_ms"]:
        print("embedding time unchanged: the vision did not move, stopping", flush=True)
        return
    out = Path("runs/eval") / model_dir.parent.name / model_dir.name / f"NPU-vision{vision_device}"
    out = out / f"textvqa_n{n}_s0.jsonl"
    evaluate.run(model_dir, "NPU", "textvqa", samples, out, config=config)
    res = metrics.score("textvqa", evaluate.read_records(out))
    out.with_suffix(".json").write_text(json.dumps({**res, "config": config}, indent=2), encoding="utf-8")
    print(out, res, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
