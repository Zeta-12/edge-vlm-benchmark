"""Benchmark datasets: download (pinned revisions), load, fixed subsets.

Parquet mirrors from the HF hub with the images inside, so there is nothing
to extract. TextVQA is CC BY 4.0, POPE annotations are MIT on COCO val2014
images.
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Source:
    repo_id: str
    revision: str
    patterns: tuple[str, ...]


SOURCES = {
    "textvqa": Source("lmms-lab/textvqa", "9c0699cd19768ac5ab97568f6b3cbac4c0062884", ("data/validation-*",)),
    "pope": Source("lmms-lab/POPE", "4db1276663dfa5eb8ad16a52d24c31a09e470896", ("data/test-*",)),
    "mmstar": Source("Lin-Chen/MMStar", "bc98d668301da7b14f648724866e57302778ab27", ("mmstar.parquet",)),
}

# same wording as lmms-eval, which the Qwen model cards follow
SHORT = "\nAnswer the question using a single word or phrase."
PROMPTS = {
    "textvqa": "{question}" + SHORT,
    "pope": "{question}" + SHORT,
    "mmstar": "{question}\nAnswer with the option's letter from the given choices directly.",
}


@dataclass
class Sample:
    id: str
    image: bytes
    question: str
    answers: list[str]
    meta: dict = field(default_factory=dict)

    @property
    def prompt(self) -> str:
        return PROMPTS[self.meta["task"]].format(question=self.question)

    def pil(self):
        from PIL import Image

        return Image.open(io.BytesIO(self.image)).convert("RGB")


def local(task: str) -> Path | None:
    """Snapshot of the pinned revision if it's already in the HF cache."""
    from huggingface_hub import snapshot_download

    src = SOURCES[task]
    try:
        root = Path(snapshot_download(src.repo_id, revision=src.revision, repo_type="dataset", local_files_only=True))
    except Exception:
        return None
    return root if any(f for pat in src.patterns for f in root.glob(pat)) else None


def download(task: str) -> Path:
    from edge_vlm.hub import fetch

    src = SOURCES[task]
    return fetch(src.repo_id, revision=src.revision, repo_type="dataset", allow_patterns=list(src.patterns))


def _rows(task: str, root: Path | None):
    import pyarrow.parquet as pq

    root = root or local(task) or download(task)
    files = sorted(f for pat in SOURCES[task].patterns for f in root.glob(pat))
    if not files:
        raise SystemExit(f"no parquet file for {task} in {root}")
    for f in files:
        yield from pq.read_table(f).to_pylist()


def _image_bytes(v) -> bytes:
    return v["bytes"] if isinstance(v, dict) else v


def load(task: str, root: Path | None = None) -> list[Sample]:
    out = []
    for r in _rows(task, root):
        if task == "textvqa":
            out.append(Sample(str(r["question_id"]), _image_bytes(r["image"]), r["question"], list(r["answers"]), {}))
        elif task == "pope":
            meta = {"category": r["category"], "image_source": r["image_source"]}
            out.append(Sample(str(r["question_id"]), _image_bytes(r["image"]), r["question"], [r["answer"]], meta))
        elif task == "mmstar":
            meta = {"category": r["category"], "l2_category": r["l2_category"]}
            out.append(Sample(str(r["index"]), _image_bytes(r["image"]), r["question"], [r["answer"]], meta))
        else:
            raise SystemExit(f"unknown task {task!r}")
    for s in out:
        s.meta["task"] = task
    return out


def order(samples: list[Sample], seed: int = 0, stratify: str | None = None) -> list[int]:
    """Fixed permutation of the sample indices. With stratify, the shuffled groups are
    interleaved round robin so that every prefix stays balanced."""
    rng = random.Random(seed)
    if stratify is None:
        idx = list(range(len(samples)))
        rng.shuffle(idx)
        return idx
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(samples):
        groups.setdefault(s.meta[stratify], []).append(i)
    perms = []
    for key in sorted(groups):
        g = groups[key][:]
        rng.shuffle(g)
        perms.append(g)
    return [g[k] for k in range(max(map(len, perms))) for g in perms if k < len(g)]


def subset(samples: list[Sample], n: int | None, seed: int = 0, stratify: str | None = None) -> list[Sample]:
    """First n samples of the fixed permutation, in that order.

    Subsets are nested (n=100 is a prefix of n=500), and since runs go in this
    order, a run stopped early is still a random (and balanced) sample.
    """
    idx = order(samples, seed, stratify)
    return [samples[i] for i in (idx if n is None else idx[:n])]


STRATIFY = {"pope": "category", "mmstar": "category"}
