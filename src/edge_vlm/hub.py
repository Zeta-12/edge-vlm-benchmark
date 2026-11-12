"""Hugging Face downloads that survive a flaky connection.

`hf download` can exit 0 with files missing when connections get reset, so we
check every file against the repo listing and retry until nothing is missing.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

log = logging.getLogger(__name__)


def _retry(fn, tries: int, wait: float):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # connection resets show up as several exception types
            if i == tries - 1:
                raise
            log.warning("attempt %d failed (%s: %s), retrying", i + 1, type(e).__name__, str(e)[:120])
            time.sleep(wait)


def missing_files(local: Path, expected: dict[str, int]) -> list[str]:
    out = []
    for name, size in expected.items():
        f = local / name
        if not f.is_file() or (size is not None and f.stat().st_size != size):
            out.append(name)
    return out


def fetch(model_id: str, revision: str | None = None, tries: int = 30, wait: float = 3.0) -> Path:
    """Download a full model repo into the HF cache and return the snapshot dir."""
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    info = _retry(lambda: api.model_info(model_id, revision=revision, files_metadata=True), tries, wait)
    expected = {s.rfilename: s.size for s in info.siblings}

    for i in range(tries):
        try:
            local = Path(snapshot_download(model_id, revision=info.sha, max_workers=4))
        except Exception as e:
            log.warning("download attempt %d failed (%s), retrying", i + 1, type(e).__name__)
            time.sleep(wait)
            continue
        todo = missing_files(local, expected)
        if not todo:
            log.info("%s@%s complete (%d files)", model_id, info.sha[:7], len(expected))
            return local
        log.warning("attempt %d: %d files missing, e.g. %s", i + 1, len(todo), todo[0])
        time.sleep(wait)
    raise RuntimeError(f"could not download {model_id} after {tries} attempts")
