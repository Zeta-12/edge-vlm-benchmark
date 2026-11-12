"""Answer of the original HF model (PyTorch, CPU, greedy) on one image + prompt.

Reference for the OpenVINO pipeline: if this one is wrong too, it's the model,
not the export. Run it in the export env of the family, e.g.

    uv run --isolated --no-project --with transformers==5.2.0 --with torch==2.15.0 \
        --with torchvision --with pillow python tools/hf_reference.py <snapshot> <image> "<prompt>"
"""

import json
import sys
import time

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


def main(snapshot: str, image: str, prompt: str, max_new_tokens: int = 64) -> None:
    processor = AutoProcessor.from_pretrained(snapshot)
    model = AutoModelForImageTextToText.from_pretrained(snapshot, dtype=torch.float32)
    model.eval()
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[text], images=[Image.open(image).convert("RGB")], return_tensors="pt")
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    answer = processor.batch_decode(out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0]
    rec = {
        "snapshot": snapshot,
        "input_tokens": int(inputs["input_ids"].shape[1]),
        "seconds": time.perf_counter() - t0,
        "prompt_text": text,
        "answer": answer,
    }
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:4])
