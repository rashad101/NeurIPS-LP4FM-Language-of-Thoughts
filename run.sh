#!/usr/bin/env bash
#
# One entrypoint for the whole experiment.
#
#   ./run.sh setup                 create the uv venv and install deps (repo-local)
#   ./run.sh doctor                environment / GPU / data / model report
#   ./run.sh data                  download and build MGSM + Global-MMLU item sets
#   ./run.sh models                download model weights into assets/models
#   ./run.sh generate [args...]    inference (resumable)
#   ./run.sh score [args...]       extract answers and metrics -> results.parquet
#   ./run.sh analyze               H1-H5 tests and tables
#   ./run.sh figures               the four figures
#   ./run.sh report [args...]      score + fertility + analyze + figures + policy card
#   ./run.sh status [args...]      per-cell progress of a run
#   ./run.sh verify [args...]      check prediction shards for corruption
#   ./run.sh export [args...]      archive predictions to exports/ (parquet + manifest)
#   ./run.sh smoke                 tiny end-to-end check on a 0.6B model
#   ./run.sh all [args...]         setup + data + models + generate + report
#   ./run.sh clean-caches          delete assets/cache (keeps weights and data)
#
# Everything (venv, weights, datasets, caches, logs, runs, results) stays inside
# this directory. Nothing is written to $HOME.
#
# Examples
#   ./run.sh all
#   ./run.sh generate --models qwen3-8b --run-id qwen3-8b-pass1
#   ./run.sh generate --conditions target en_cot --langs hi bn te
#   ./run.sh generate --run-id qwen3-8b-pass1          # resumes where it stopped
#   ./run.sh report --run-id all
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# shellcheck source=scripts/env.sh
source "$SCRIPT_DIR/scripts/env.sh"

PYTHON_VERSION="${LTD_PYTHON_VERSION:-3.11}"
VENV="$LTD_ROOT/.venv"
STAGE_LOG_DIR="$LTD_ROOT/logs"
mkdir -p "$STAGE_LOG_DIR"
DRIVER_LOG="$STAGE_LOG_DIR/run_sh.log"

log()  { printf '\033[1;34m[run.sh]\033[0m %s\n' "$*" | tee -a "$DRIVER_LOG" >&2; }
warn() { printf '\033[1;33m[run.sh]\033[0m %s\n' "$*" | tee -a "$DRIVER_LOG" >&2; }
die()  { printf '\033[1;31m[run.sh]\033[0m %s\n' "$*" | tee -a "$DRIVER_LOG" >&2; exit 1; }

on_error() {
  local code=$?
  warn "stage '${CURRENT_STAGE:-?}' failed with exit code $code"
  warn "logs: $DRIVER_LOG and $STAGE_LOG_DIR/"
  warn "generation is checkpointed — rerun the same command (same --run-id) to resume"
  exit "$code"
}
trap on_error ERR

banner() {
  CURRENT_STAGE="$1"
  log "=============================================================="
  log "stage: $1   ($(date '+%Y-%m-%d %H:%M:%S'))"
  log "=============================================================="
}

# --------------------------------------------------------------------------
# setup
# --------------------------------------------------------------------------
ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    log "uv $(uv --version)"
    return
  fi
  warn "uv not found — installing it into $LTD_ROOT/.local"
  export UV_INSTALL_DIR="$LTD_ROOT/.local/bin"
  export CARGO_HOME="$LTD_ROOT/.local/cargo"
  mkdir -p "$UV_INSTALL_DIR"
  curl -LsSf https://astral.sh/uv/install.sh | env INSTALLER_NO_MODIFY_PATH=1 sh
  export PATH="$UV_INSTALL_DIR:$PATH"
  command -v uv >/dev/null 2>&1 || die "uv installation failed; install it manually"
  log "uv installed: $(uv --version)"
}

stage_setup() {
  banner "setup (uv environment, repo-local)"
  ensure_uv

  if [[ ! -x "$VENV/bin/python" ]]; then
    log "creating venv at $VENV (python $PYTHON_VERSION)"
    uv venv --python "$PYTHON_VERSION" "$VENV"
  else
    log "venv already exists at $VENV"
  fi

  local extras="dev"
  if [[ "$(uname -s)" == "Linux" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
      extras="gpu"
      log "NVIDIA GPU detected -> installing the [gpu] extra (vLLM)"
      nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tee -a "$DRIVER_LOG" >&2
    else
      warn "Linux without nvidia-smi -> installing the [dev] extra (no vLLM)"
    fi
  else
    log "$(uname -s) detected -> installing the [dev] extra (transformers backend only)"
  fi

  log "installing the project with extras: [$extras]"
  VIRTUAL_ENV="$VENV" uv pip install --python "$VENV/bin/python" -e ".[$extras]" \
    2>&1 | tee -a "$DRIVER_LOG"

  log "installed:"
  VIRTUAL_ENV="$VENV" uv pip list --python "$VENV/bin/python" 2>/dev/null \
    | grep -Ei '^(torch|vllm|transformers|datasets|pandas|numpy|statsmodels|matplotlib|py3langid) ' \
    | tee -a "$DRIVER_LOG" >&2 || true
  log "setup done. venv: $VENV"
}

ltd() {
  [[ -x "$VENV/bin/python" ]] || die "venv missing — run './run.sh setup' first"
  "$VENV/bin/python" -m ltd "$@"
}

# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------
stage_doctor()   { banner "doctor";                ltd doctor "$@"; }
stage_data()     { banner "data (item sets)";       ltd fetch-data --verify "$@"; }
stage_models()   { banner "models (weights)";       ltd fetch-models "$@"; }
stage_prompts()  { banner "prompts (inventory)";    ltd prompts "$@"; }
stage_generate() { banner "generate (inference)";   ltd generate "$@"; }
stage_score()    { banner "score";                  ltd score "$@"; }
stage_fert()     { banner "fertility";              ltd fertility "$@"; }
stage_analyze()  { banner "analyze";                ltd analyze "$@"; }
stage_figures()  { banner "figures";                ltd figures "$@"; }
stage_card()     { banner "policy card";            ltd policy-card "$@"; }
stage_status()   { banner "status";                 ltd status "$@"; }
stage_report()   { banner "report";                 ltd report "$@"; }
stage_export()   { banner "export (archive predictions)"; ltd export "$@"; }
stage_verify()   { banner "verify (shard integrity)";     ltd verify "$@"; }

stage_smoke() {
  banner "smoke (tiny end-to-end check)"
  local run_id="smoke-$(date '+%Y%m%d-%H%M%S')"
  log "run id: $run_id"
  ltd fetch-data --tasks all --langs en hi bn de
  ltd fetch-models --models smoke-qwen-0_6b
  ltd generate --run-id "$run_id" --models smoke-qwen-0_6b \
    --langs en hi bn de --tasks mgsm gmmlu --conditions all \
    --limit 2 --max-new-tokens 160 --chunk-size 2
  ltd score --run-id "$run_id"
  ltd fertility --models smoke-qwen-0_6b --langs en hi bn de
  ltd analyze
  ltd figures
  ltd policy-card
  log "smoke run '$run_id' completed. Numbers from a 0.6B model are NOT results."
}

stage_all() {
  stage_setup
  stage_doctor
  stage_data
  stage_models
  stage_prompts
  stage_generate "$@"
  stage_report "$@"
  log "ALL STAGES COMPLETE"
  log "  results:   $LTD_ROOT/results/results.parquet"
  log "  analysis:  $LTD_ROOT/results/analysis.json"
  log "  figures:   $LTD_ROOT/results/figures/"
  log "  artifacts: $LTD_ROOT/artifacts/"
  log "  logs:      $LTD_ROOT/logs/"
}

stage_clean_caches() {
  banner "clean-caches"
  warn "removing $LTD_CACHE (model weights in assets/models and data/ are kept)"
  rm -rf "$LTD_CACHE"
  mkdir -p "$LTD_CACHE"
  log "done"
}

usage() { sed -n '2,32p' "$0"; }

# --------------------------------------------------------------------------
main() {
  local stage="${1:-all}"
  shift || true
  log "invoked: $stage $*"
  case "$stage" in
    setup)         stage_setup ;;
    doctor)        stage_doctor "$@" ;;
    data)          stage_data "$@" ;;
    models)        stage_models "$@" ;;
    prompts)       stage_prompts "$@" ;;
    generate|gen)  stage_generate "$@" ;;
    score)         stage_score "$@" ;;
    fertility)     stage_fert "$@" ;;
    analyze)       stage_analyze "$@" ;;
    figures)       stage_figures "$@" ;;
    policy-card)   stage_card "$@" ;;
    status)        stage_status "$@" ;;
    export)        stage_export "$@" ;;
    verify)        stage_verify "$@" ;;
    report)        stage_report "$@" ;;
    smoke)         stage_smoke ;;
    all)           stage_all "$@" ;;
    clean-caches)  stage_clean_caches ;;
    -h|--help|help) usage ;;
    *)             die "unknown stage '$stage' (try './run.sh --help')" ;;
  esac
}

main "$@"
