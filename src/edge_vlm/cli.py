"""`evb` command line."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

DEVICES = ("CPU", "GPU", "NPU")


def cmd_download(args) -> None:
    from edge_vlm.hub import fetch

    for model_id in args.model_id:
        print(fetch(model_id))


def cmd_export(args) -> None:
    from edge_vlm.export import export

    for fmt in args.format:
        export(args.snapshot, args.out / args.name / fmt, fmt, args.python)


def cmd_probe(args) -> None:
    from edge_vlm.probe import make_sign

    args.out.mkdir(parents=True, exist_ok=True)
    if args.image is None:
        args.image = make_sign(args.out / "sign.png")
    for model_dir in args.model_dir:
        for device in args.devices:
            out = args.out / f"{model_dir.parent.name}_{model_dir.name}_{device}.json"
            if out.is_file() and not args.force:
                logging.info("skip %s (exists)", out.name)
                continue
            cmd = [sys.executable, "-m", "edge_vlm.probe", str(model_dir), device, str(args.image), args.prompt]
            cmd.append(str(out))
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
            if not out.is_file():  # hard crash, the child had no chance to write
                tail = (proc.stderr or proc.stdout)[-2000:]
                rec = {"model": model_dir.parent.name, "format": model_dir.name, "device": device, "ok": False}
                rec["error"] = f"process exited with {proc.returncode}: {tail}"
                out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
            rec = json.loads(out.read_text(encoding="utf-8"))
            name = f"{rec['model']} {rec['format']} {device}"
            if rec["ok"]:
                logging.info(
                    "%s: load %.1f s, ttft %.0f ms, %.1f tok/s, %.0f MB, %r",
                    name,
                    rec["load_s"],
                    rec["ttft_ms"],
                    rec["tok_per_s"],
                    rec["peak_rss_mb"],
                    rec["text"][:80],
                )
            else:
                logging.info("%s: FAILED %s", name, rec["error"][:300])


def cmd_matrix(args) -> None:
    from edge_vlm import matrix

    table = matrix.markdown(matrix.load(args.probe_dir))
    print(table)
    if args.out:
        args.out.write_text(table + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="evb", description="Small VLMs on CPU / iGPU / NPU with OpenVINO GenAI.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("download", help="download HF checkpoints, checked file by file")
    s.add_argument("model_id", nargs="+")
    s.set_defaults(func=cmd_download)

    s = sub.add_parser("export", help="convert a HF snapshot to OpenVINO IR")
    s.add_argument("snapshot", type=Path)
    s.add_argument("--name", required=True, help="short model name, used as folder name")
    s.add_argument("--format", nargs="+", default=["fp16"])
    s.add_argument("--out", type=Path, default=Path("models"))
    s.add_argument("--python", help="python interpreter for the export env")
    s.set_defaults(func=cmd_export)

    s = sub.add_parser("probe", help="load each model on each device and run one prompt")
    s.add_argument("model_dir", type=Path, nargs="+")
    s.add_argument("--devices", nargs="+", default=list(DEVICES))
    s.add_argument("--image", type=Path, help="default: a generated sign with known text")
    s.add_argument("--prompt", default="What is written on the sign? Answer with the text only.")
    s.add_argument("--out", type=Path, default=Path("runs/probe"))
    s.add_argument("--timeout", type=int, default=1800)
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_probe)

    s = sub.add_parser("matrix", help="support matrix from the probe records")
    s.add_argument("probe_dir", type=Path, nargs="?", default=Path("runs/probe"))
    s.add_argument("--out", type=Path)
    s.set_defaults(func=cmd_matrix)
    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
