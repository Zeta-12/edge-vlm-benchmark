"""Load one exported VLM on one device and run one prompt.

Run in its own process (`python -m edge_vlm.probe ...`) so a driver crash or a
compile failure on one device doesn't take the others down, and so peak RAM is
per (model, device).
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

SIGN_TEXT = "ROAD WORKS 300 m"


def make_sign(path: Path, text: str = SIGN_TEXT) -> Path:
    """A 640x480 picture of a sign, so the probe has a known answer."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (640, 480), (120, 160, 200))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 360, 640, 480), fill=(90, 90, 90))
    d.rectangle((300, 250, 340, 400), fill=(60, 60, 60))
    d.rectangle((90, 100, 550, 260), fill=(250, 200, 0), outline=(0, 0, 0), width=6)
    font = ImageFont.load_default(size=40)
    w = d.textlength(text, font=font)
    d.text(((640 - w) / 2, 155), text, fill=(0, 0, 0), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def load_image(path: Path):
    import numpy as np
    import openvino as ov
    from PIL import Image

    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    return ov.Tensor(img[None])


def _metric(fn):
    try:
        v = fn()
    except Exception:
        return None
    return float(v.mean) if hasattr(v, "mean") else float(v)


def run(model_dir: Path, device: str, image: Path, prompt: str, max_new_tokens: int = 64) -> dict:
    import openvino as ov
    import openvino_genai as ov_genai
    import psutil

    rec = {
        "model": model_dir.parent.name,
        "format": model_dir.name,
        "device": device,
        "openvino": ov.__version__,
        "prompt": prompt,
        "ok": False,
    }
    try:
        t0 = time.perf_counter()
        pipe = ov_genai.VLMPipeline(str(model_dir), device)
        rec["load_s"] = time.perf_counter() - t0

        cfg = ov_genai.GenerationConfig()
        cfg.max_new_tokens = max_new_tokens
        cfg.do_sample = False
        img = load_image(image)

        pipe.generate(prompt, images=[img], generation_config=cfg)  # warmup
        res = pipe.generate(prompt, images=[img], generation_config=cfg)
        pm = res.perf_metrics
        rec.update(
            ok=True,
            text=res.texts[0],
            input_tokens=_metric(pm.get_num_input_tokens),
            generated_tokens=_metric(pm.get_num_generated_tokens),
            ttft_ms=_metric(pm.get_ttft),
            tpot_ms=_metric(pm.get_tpot),
            tok_per_s=_metric(pm.get_throughput),
            generate_ms=_metric(pm.get_generate_duration),
            embeddings_ms=_metric(getattr(pm, "get_prepare_embeddings_duration", lambda: None)),
        )
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {str(e)[:2000]}"
        rec["traceback"] = traceback.format_exc()[-4000:]
    mem = psutil.Process().memory_info()
    rec["peak_rss_mb"] = getattr(mem, "peak_wset", mem.rss) / 2**20
    return rec


def main(argv: list[str] | None = None) -> None:
    model_dir, device, image, prompt, out = (argv or sys.argv[1:])[:5]
    rec = run(Path(model_dir), device, Path(image), prompt)
    Path(out).write_text(json.dumps(rec, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
