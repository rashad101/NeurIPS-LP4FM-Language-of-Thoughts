#!/usr/bin/env bash
# Sourced by run.sh and safe to source directly:  source scripts/env.sh
#
# Pins every cache-bearing environment variable inside the repository, so a run
# writes nothing to $HOME. src/ltd/env.py sets the same variables from Python;
# doing it here too means even a subprocess that starts before Python (uv, git,
# nvidia tooling) stays inside the checkout.

# Resolve the repo root from this file's location (handles spaces in the path).
_LTD_THIS="${BASH_SOURCE[0]:-$0}"
LTD_ROOT="$(cd -- "$(dirname -- "$_LTD_THIS")/.." && pwd)"
export LTD_ROOT

export LTD_ASSETS="$LTD_ROOT/assets"
export LTD_CACHE="$LTD_ASSETS/cache"

# --- uv / Python -----------------------------------------------------------
export UV_CACHE_DIR="$LTD_ROOT/.uv-cache"
export UV_PROJECT_ENVIRONMENT="$LTD_ROOT/.venv"
export VIRTUAL_ENV="$LTD_ROOT/.venv"
export PYTHONNOUSERSITE=1

# --- Hugging Face ----------------------------------------------------------
export HF_HOME="$LTD_ASSETS/hf_home"
export HF_HUB_CACHE="$LTD_ASSETS/models"
export HUGGINGFACE_HUB_CACHE="$LTD_ASSETS/models"
export TRANSFORMERS_CACHE="$LTD_ASSETS/models"
export HF_DATASETS_CACHE="$LTD_CACHE/datasets"
export HF_MODULES_CACHE="$LTD_CACHE/hf_modules"
export HF_ASSETS_CACHE="$LTD_CACHE/hf_assets"
export HF_HUB_DISABLE_TELEMETRY=1

# --- generic caches that otherwise land in $HOME ---------------------------
export XDG_CACHE_HOME="$LTD_CACHE/xdg"
export XDG_CONFIG_HOME="$LTD_CACHE/xdg_config"
export TORCH_HOME="$LTD_CACHE/torch"
export TORCHINDUCTOR_CACHE_DIR="$LTD_CACHE/torchinductor"
export TRITON_CACHE_DIR="$LTD_CACHE/triton"
export MPLCONFIGDIR="$LTD_CACHE/matplotlib"
export MPLBACKEND=Agg
export NLTK_DATA="$LTD_CACHE/nltk"
export TIKTOKEN_CACHE_DIR="$LTD_CACHE/tiktoken"
export NUMBA_CACHE_DIR="$LTD_CACHE/numba"
export FONTCONFIG_PATH="$LTD_CACHE/fontconfig"

# --- vLLM ------------------------------------------------------------------
export VLLM_CACHE_ROOT="$LTD_CACHE/vllm"
export VLLM_ASSETS_CACHE="$LTD_CACHE/vllm_assets"
export VLLM_CONFIG_ROOT="$LTD_CACHE/vllm_config"
export OUTLINES_CACHE_DIR="$LTD_CACHE/outlines"
export FLASHINFER_WORKSPACE_BASE="$LTD_CACHE/flashinfer"

export TOKENIZERS_PARALLELISM=false

mkdir -p \
  "$LTD_ASSETS" "$LTD_CACHE" "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" \
  "$HF_MODULES_CACHE" "$HF_ASSETS_CACHE" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" \
  "$TORCH_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR" \
  "$NLTK_DATA" "$TIKTOKEN_CACHE_DIR" "$NUMBA_CACHE_DIR" "$FONTCONFIG_PATH" \
  "$VLLM_CACHE_ROOT" "$VLLM_ASSETS_CACHE" "$VLLM_CONFIG_ROOT" "$OUTLINES_CACHE_DIR" \
  "$FLASHINFER_WORKSPACE_BASE" "$UV_CACHE_DIR" \
  "$LTD_ROOT/logs" "$LTD_ROOT/runs" "$LTD_ROOT/results" "$LTD_ROOT/data"
