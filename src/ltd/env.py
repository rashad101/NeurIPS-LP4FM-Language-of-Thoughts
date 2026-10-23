"""Repo-local environment pinning.

This module MUST be imported before ``torch``, ``transformers``, ``datasets``,
``vllm`` or ``matplotlib``. Those libraries read their cache locations from the
environment at import time, and by default they write to ``$HOME`` (``~/.cache``,
``~/.triton``, ``~/.config/matplotlib``, ...). Every path below is redirected
into the repository so that a run leaves nothing outside the checkout.

Anything already set in the environment is respected, so ``run.sh`` (which
exports the same variables before Python starts) and this module agree.
"""

from __future__ import annotations

import os
from pathlib import Path

# src/ltd/env.py -> src/ltd -> src -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[2]

# Top-level, repo-local directories.
ASSETS_DIR = REPO_ROOT / "assets"
CACHE_DIR = ASSETS_DIR / "cache"
HF_HOME_DIR = ASSETS_DIR / "hf_home"
MODELS_DIR = ASSETS_DIR / "models"
DATA_DIR = REPO_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RUNS_DIR = REPO_ROOT / "runs"
LOGS_DIR = REPO_ROOT / "logs"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
CONFIGS_DIR = REPO_ROOT / "configs"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

_ALL_DIRS = (
    ASSETS_DIR,
    CACHE_DIR,
    HF_HOME_DIR,
    MODELS_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    RUNS_DIR,
    LOGS_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    ARTIFACTS_DIR,
)

#: Environment variables redirected into the repo. Values are relative to
#: ``REPO_ROOT`` and are created on import.
_ENV_MAP: dict[str, Path] = {
    # Hugging Face
    "HF_HOME": HF_HOME_DIR,
    "HF_HUB_CACHE": MODELS_DIR,
    "HUGGINGFACE_HUB_CACHE": MODELS_DIR,
    "HF_DATASETS_CACHE": CACHE_DIR / "datasets",
    "HF_MODULES_CACHE": CACHE_DIR / "hf_modules",
    "HF_ASSETS_CACHE": CACHE_DIR / "hf_assets",
    "TRANSFORMERS_CACHE": MODELS_DIR,
    # Generic caches that otherwise land in $HOME
    "XDG_CACHE_HOME": CACHE_DIR / "xdg",
    "XDG_CONFIG_HOME": CACHE_DIR / "xdg_config",
    "TORCH_HOME": CACHE_DIR / "torch",
    "TORCHINDUCTOR_CACHE_DIR": CACHE_DIR / "torchinductor",
    "TRITON_CACHE_DIR": CACHE_DIR / "triton",
    "MPLCONFIGDIR": CACHE_DIR / "matplotlib",
    "NLTK_DATA": CACHE_DIR / "nltk",
    "TIKTOKEN_CACHE_DIR": CACHE_DIR / "tiktoken",
    "NUMBA_CACHE_DIR": CACHE_DIR / "numba",
    "FONTCONFIG_PATH": CACHE_DIR / "fontconfig",
    # vLLM and friends
    "VLLM_CACHE_ROOT": CACHE_DIR / "vllm",
    "VLLM_ASSETS_CACHE": CACHE_DIR / "vllm_assets",
    "VLLM_CONFIG_ROOT": CACHE_DIR / "vllm_config",
    "OUTLINES_CACHE_DIR": CACHE_DIR / "outlines",
    "FLASHINFER_WORKSPACE_BASE": CACHE_DIR / "flashinfer",
    # uv (only relevant if uv is invoked from inside Python)
    "UV_CACHE_DIR": REPO_ROOT / ".uv-cache",
}

#: Plain string environment defaults (not paths).
_ENV_DEFAULTS: dict[str, str] = {
    # Silence the tokenizers fork warning that fires under DataLoader workers.
    "TOKENIZERS_PARALLELISM": "false",
    # Keep matplotlib headless on the GPU box.
    "MPLBACKEND": "Agg",
    # Datasets progress bars are handled by our own tqdm wrappers.
    "HF_HUB_DISABLE_TELEMETRY": "1",
}

_configured = False


def configure(create_dirs: bool = True) -> None:
    """Point every cache-bearing environment variable inside the repository.

    Idempotent, and never overrides a variable the caller already set.
    """
    global _configured
    if _configured:
        return

    for key, value in _ENV_MAP.items():
        os.environ.setdefault(key, str(value))
    for key, value in _ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)

    if create_dirs:
        for directory in _ALL_DIRS:
            directory.mkdir(parents=True, exist_ok=True)
        # Also create whatever the (possibly caller-overridden) env points at.
        for key in _ENV_MAP:
            try:
                Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
            except OSError:  # e.g. a read-only override; not fatal
                pass

    _configured = True


def escaped_home_report() -> list[str]:
    """Return env vars that still resolve outside the repo (diagnostics)."""
    offenders = []
    for key in _ENV_MAP:
        value = os.environ.get(key)
        if not value:
            offenders.append(f"{key}=<unset>")
            continue
        try:
            Path(value).resolve().relative_to(REPO_ROOT)
        except ValueError:
            offenders.append(f"{key}={value}")
    return offenders


configure()
