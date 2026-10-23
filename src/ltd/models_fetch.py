"""Download model weights into ``assets/models/`` inside the repository.

``env.py`` already points ``HF_HUB_CACHE`` at ``assets/models``, so a plain
``from_pretrained`` would land in the repo too. We still snapshot explicitly so
that (a) downloads happen once, up front, with a progress bar, before any GPU is
held, and (b) generation can run with ``HF_HUB_OFFLINE=1``.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .config import Config, ModelSpec
from .env import MODELS_DIR
from .logging_utils import LoggedProgress, get_logger

log = get_logger("models")

#: Written next to each snapshot so the exact weights used are recoverable later.
META_FILENAME = ".ltd_snapshot.json"

# Weight shards + everything the tokenizer and chat template need.
ALLOW_PATTERNS = [
    "*.safetensors",
    "*.safetensors.index.json",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "preprocessor_config.json",
]


def local_model_path(model: ModelSpec) -> Path:
    """Where the snapshot lives (``assets/models/snapshots/<owner__name>``)."""
    return MODELS_DIR / "snapshots" / model.local_dirname


def is_downloaded(model: ModelSpec) -> bool:
    path = local_model_path(model)
    if not (path / "config.json").exists():
        return False
    return any(path.glob("*.safetensors"))


def resolve_model_path(model: ModelSpec) -> str:
    """Local snapshot path if present, else the Hub id (which will download)."""
    path = local_model_path(model)
    if is_downloaded(model):
        return str(path)
    return model.hf_id


def download_model(model: ModelSpec, force: bool = False) -> Path:
    from huggingface_hub import snapshot_download

    target = local_model_path(model)
    if is_downloaded(model) and not force:
        log.info("%s already present at %s", model.hf_id, target)
        return target

    target.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s -> %s", model.hf_id, target)
    # hf_transfer gives a big speedup on fat pipes when installed.
    try:
        import hf_transfer  # noqa: F401

        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    except ImportError:
        pass

    path = snapshot_download(
        repo_id=model.hf_id,
        local_dir=str(target),
        allow_patterns=ALLOW_PATTERNS,
        max_workers=8,
        tqdm_class=None,
    )
    size_gb = sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) / 1e9
    log.info("downloaded %s (%.2f GB) -> %s", model.hf_id, size_gb, path)
    _write_snapshot_meta(model, Path(path), size_gb)
    return Path(path)


def _write_snapshot_meta(model: ModelSpec, path: Path, size_gb: float) -> None:
    """Pin the exact Hub revision alongside the weights.

    "Qwen/Qwen3-8B" is a moving target; a year from now the results are only
    interpretable if the commit that produced them is recorded.
    """
    revision = None
    try:
        from huggingface_hub import HfApi

        revision = HfApi().model_info(model.hf_id).sha
    except Exception as exc:  # offline, rate-limited, renamed repo, ...
        log.warning("could not resolve the Hub revision for %s (%s)", model.hf_id, exc)

    payload = {
        "hf_id": model.hf_id,
        "revision": revision,
        "model_key": model.key,
        "dtype": model.dtype,
        "size_gb": round(size_gb, 3),
        "downloaded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (path / META_FILENAME).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("%s revision=%s", model.hf_id, revision or "unknown")


def snapshot_meta(model: ModelSpec) -> dict[str, str | None]:
    path = local_model_path(model) / META_FILENAME
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


#: Resolution is attempted at most once per model per process. ``ModelSpec`` holds
#: a dict (``decoding``) so it is unhashable and cannot key an lru_cache.
_REVISION_CACHE: dict[str, str | None] = {}


def model_revision(model: ModelSpec) -> str | None:
    """The Hub commit the local weights came from, or ``None`` if unresolvable.

    Backfills the snapshot metadata for weights downloaded before revision
    recording existed. Never blocks generation: an offline GPU box records
    ``None`` and logs a warning once.
    """
    if model.key in _REVISION_CACHE:
        return _REVISION_CACHE[model.key]

    meta = snapshot_meta(model)
    if meta.get("revision"):
        _REVISION_CACHE[model.key] = str(meta["revision"])
        return _REVISION_CACHE[model.key]

    # Resolve even when the weights came from the Hub cache rather than an
    # explicit `ltd fetch-models` snapshot: what matters for provenance is which
    # revision the run used, and an implicit download always takes `main`.
    try:
        from huggingface_hub import HfApi

        revision = HfApi().model_info(model.hf_id).sha
    except Exception as exc:
        log.warning(
            "no recorded revision for %s and the Hub is unreachable (%s); "
            "predictions will record model_revision=null",
            model.hf_id, exc,
        )
        _REVISION_CACHE[model.key] = None
        return None

    # Persist alongside the snapshot when there is one; otherwise the revision
    # still reaches the predictions via the cache below.
    snapshot_dir = local_model_path(model)
    if snapshot_dir.exists():
        meta = dict(meta) or {"hf_id": model.hf_id, "model_key": model.key}
        meta["revision"] = revision
        meta["revision_backfilled_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        (snapshot_dir / META_FILENAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    _REVISION_CACHE[model.key] = str(revision)
    log.info("resolved revision for %s: %s", model.hf_id, revision)
    return str(revision)


def download_all(config: Config, models: Iterable[str] | None = None, force: bool = False) -> dict[str, str]:
    keys = config.resolve_models(models)
    out: dict[str, str] = {}
    with LoggedProgress(len(keys), "fetch models", unit="model") as bar:
        for key in keys:
            spec = config.model(key)
            out[key] = str(download_model(spec, force=force))
            bar.update(1, {"model": key})
    return out


def load_tokenizer(model: ModelSpec):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(resolve_model_path(model), trust_remote_code=False)
