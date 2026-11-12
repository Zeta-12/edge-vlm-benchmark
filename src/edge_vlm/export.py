"""HF checkpoint -> OpenVINO IR with optimum-cli.

optimum-intel caps the transformers version per model family (qwen3_vl needs
<= 5.0, gemma4 needs >= 5.5), so each export runs in a throwaway uv env with
the pin of its family. The benchmark env itself only needs openvino-genai.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

log = logging.getLogger(__name__)

OPTIMUM = "optimum-intel[openvino]==2.2.0"
TORCH = "torch==2.15.0"
OPENVINO = ["openvino==2026.4.1", "openvino-tokenizers==2026.4.1.0"]

# transformers pin per model_type, from the MIN/MAX_TRANSFORMERS_VERSION of optimum-intel 2.2.0
TRANSFORMERS = {
    "qwen2_5_vl": "transformers==5.0.0",
    "qwen3_vl": "transformers==5.0.0",
    "qwen3_5": "transformers==5.2.0",
    "gemma3": "transformers==5.5.4",
    "gemma3n": "transformers==5.5.4",
    "gemma4": "transformers==5.5.4",
}

# --ratio 1.0 so every LM weight gets the 4-bit format (the default keeps some layers in int8)
FORMATS = {
    "fp16": ["--weight-format", "fp16"],
    "int8": ["--weight-format", "int8"],
    "int4-asym-g128": ["--weight-format", "int4", "--group-size", "128", "--ratio", "1.0"],
    "int4-sym-g128": ["--weight-format", "int4", "--sym", "--group-size", "128", "--ratio", "1.0"],
    "int4-sym-cw": ["--weight-format", "int4", "--sym", "--group-size", "-1", "--ratio", "1.0"],
}


def model_type(snapshot: Path) -> str:
    return json.loads((snapshot / "config.json").read_text(encoding="utf-8"))["model_type"]


def uv_exe() -> str:
    exe = os.environ.get("EVB_UV") or shutil.which("uv")
    if not exe:
        raise SystemExit("uv not found, put it on PATH or set EVB_UV")
    return exe


def export_cmd(snapshot: Path, out_dir: Path, fmt: str, python: str | None = None) -> list[str]:
    mtype = model_type(snapshot)
    if mtype not in TRANSFORMERS:
        raise SystemExit(f"no export recipe for model_type {mtype!r}")
    if fmt not in FORMATS:
        raise SystemExit(f"unknown format {fmt!r}, choose from {', '.join(FORMATS)}")
    cmd = [uv_exe(), "run", "--isolated", "--no-project"]
    if python:
        cmd += ["--python", python]
    for req in [OPTIMUM, TRANSFORMERS[mtype], TORCH, "torchvision", *OPENVINO]:
        cmd += ["--with", req]
    cmd += ["optimum-cli", "export", "openvino", "-m", str(snapshot), "--task", "image-text-to-text"]
    return cmd + FORMATS[fmt] + [str(out_dir)]


def export(snapshot: Path, out_dir: Path, fmt: str, python: str | None = None) -> Path:
    if (out_dir / "openvino_language_model.xml").is_file():
        log.info("%s already exported", out_dir)
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = export_cmd(snapshot, out_dir, fmt, python)
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    t0 = time.perf_counter()
    with open(out_dir.parent / f"export_{fmt}.log", "w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
    if proc.returncode != 0:
        raise SystemExit(f"export failed ({proc.returncode}), see {out_dir.parent / f'export_{fmt}.log'}")
    meta = {"snapshot": snapshot.name, "format": fmt, "flags": FORMATS[fmt], "seconds": time.perf_counter() - t0}
    (out_dir / "evb_export.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("exported %s in %.0f s", out_dir, meta["seconds"])
    return out_dir
