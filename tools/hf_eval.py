"""Original HF model (PyTorch, CPU, fp32, greedy) on a task subset, same jsonl format as `evb eval`.

Used to check that the OpenVINO export answers the same thing. Run it in the
export env of the family with the repo's src on the path, e.g.

    PYTHONPATH=src uv run --isolated --no-project --with transformers==5.0.0 --with torch==2.15.0 \
        --with torchvision --with pillow --with pyarrow --with huggingface_hub \
        python tools/hf_eval.py <snapshot> textvqa 50 runs/eval/<model>/hf-fp32/CPU/textvqa_n50_s0.jsonl
"""

import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from edge_vlm import data
from edge_vlm.evaluate import MAX_NEW_TOKENS, done_ids


def main(snapshot: str, task: str, n: str, out: str, seed: str = "0") -> None:
    out = Path(out)
    samples = data.subset(data.load(task), int(n), int(seed), data.STRATIFY.get(task))
    todo = [s for s in samples if s.id not in done_ids(out)]
    processor = AutoProcessor.from_pretrained(snapshot)
    model = AutoModelForImageTextToText.from_pretrained(snapshot, dtype=torch.float32).eval()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as f:
        for i, s in enumerate(todo, 1):
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": s.prompt}]}]
            text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            inputs = processor(text=[text], images=[s.pil()], return_tensors="pt")
            t0 = time.perf_counter()
            with torch.inference_mode():
                ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS[task], do_sample=False)
            new = ids[:, inputs["input_ids"].shape[1] :]
            rec = {
                "id": s.id,
                "pred": processor.batch_decode(new, skip_special_tokens=True)[0],
                "answers": s.answers,
                "meta": s.meta,
                "input_tokens": int(inputs["input_ids"].shape[1]),
                "generated_tokens": int(new.shape[1]),
                "generate_ms": (time.perf_counter() - t0) * 1000,
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"{i}/{len(todo)} {rec['generate_ms'] / 1000:.1f} s {rec['pred']!r}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
