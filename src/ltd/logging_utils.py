"""Logging and progress-bar helpers.

Every command logs to three places:

1. the console (stderr, INFO level, terse),
2. ``logs/<run_id>/<command>.log`` (DEBUG level, full detail),
3. ``logs/ltd.log`` (INFO level, append-only across all runs — the audit trail).

Progress bars are ``tqdm`` on stderr. Because a bar's output does not reach the
log file, :class:`LoggedProgress` mirrors periodic progress lines into the log so
a crashed run can still be diagnosed from ``logs/`` alone.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from tqdm.auto import tqdm

from .env import LOGS_DIR, REPO_ROOT

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured_for: str | None = None


class _TqdmStream:
    """Route logging through ``tqdm.write`` so bars are not shredded."""

    @staticmethod
    def write(message: str) -> None:
        tqdm.write(message.rstrip("\n"), file=sys.stderr)

    @staticmethod
    def flush() -> None:
        try:
            sys.stderr.flush()
        except ValueError:  # closed stream at interpreter shutdown
            pass


def setup_logging(command: str, run_id: str | None = None, verbose: bool = False) -> Path:
    """Install console + per-run + global file handlers. Returns the log path."""
    global _configured_for

    run_dir = LOGS_DIR / (run_id or "adhoc")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / f"{command}.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler(_TqdmStream)  # type: ignore[arg-type]
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)-7s | %(message)s"))
    root.addHandler(console)

    per_run = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    per_run.setLevel(logging.DEBUG)
    per_run.setFormatter(formatter)
    root.addHandler(per_run)

    global_log = logging.FileHandler(LOGS_DIR / "ltd.log", mode="a", encoding="utf-8")
    global_log.setLevel(logging.INFO)
    global_log.setFormatter(logging.Formatter(
        f"%(asctime)s | %(levelname)-7s | run={run_id or '-'} cmd={command} | %(message)s",
        datefmt=_DATE_FORMAT,
    ))
    root.addHandler(global_log)

    # Third-party noise. httpcore/httpx in particular log entire HTTP headers at
    # DEBUG, which drowns the run's own log file.
    for noisy in (
        "urllib3", "filelock", "fsspec", "matplotlib", "matplotlib.font_manager",
        "PIL", "datasets", "httpx", "httpcore", "huggingface_hub", "hf_xet",
        "botocore", "s3transfer", "asyncio", "aiohttp", "numexpr", "pyarrow",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured_for = command
    logging.getLogger("ltd").debug("logging initialised -> %s", log_path)
    return log_path


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"ltd.{name}")


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        sha = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10, check=False,
        ).stdout.strip()
        return f"{sha}{'-dirty' if dirty else ''}" if sha else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def gpu_report() -> list[str]:
    """Human-readable GPU inventory, empty when no NVIDIA driver is present."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def gpu_memory_mb() -> list[dict[str, int]]:
    """Per-GPU memory in MB via ``nvidia-smi``: index, used, free, total.

    Deliberately uses nvidia-smi rather than ``torch.cuda.mem_get_info``: vLLM may
    run its workers in separate processes, whose allocations torch cannot see from
    the parent. Empty list when there is no NVIDIA driver.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    gpus = []
    for line in out.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            gpus.append({
                "index": int(parts[0]), "used_mb": int(parts[1]),
                "free_mb": int(parts[2]), "total_mb": int(parts[3]),
            })
        except ValueError:
            continue
    return gpus


def total_gpu_used_mb() -> int | None:
    gpus = gpu_memory_mb()
    if not gpus:
        return None
    return sum(g["used_mb"] for g in gpus)


def log_gpu_memory(context: str, logger_name: str = "gpu") -> int | None:
    """Log current GPU usage and return total used MB (``None`` without a GPU)."""
    log = get_logger(logger_name)
    gpus = gpu_memory_mb()
    if not gpus:
        return None
    for gpu in gpus:
        log.info(
            "[%s] gpu%d: %.1f/%.1f GiB used, %.1f GiB free",
            context, gpu["index"], gpu["used_mb"] / 1024,
            gpu["total_mb"] / 1024, gpu["free_mb"] / 1024,
        )
    return sum(g["used_mb"] for g in gpus)


def log_environment(command: str, extra: dict[str, Any] | None = None) -> None:
    log = get_logger("env")
    log.info("command=%s git=%s", command, git_sha())
    log.info("python=%s platform=%s", sys.version.split()[0], platform.platform())
    log.info("repo_root=%s", REPO_ROOT)
    gpus = gpu_report()
    if gpus:
        for line in gpus:
            log.info("gpu: %s", line)
    else:
        log.info("gpu: none detected (nvidia-smi unavailable)")
    for key in ("HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "VLLM_CACHE_ROOT"):
        log.debug("env %s=%s", key, os.environ.get(key))
    if extra:
        for key, value in extra.items():
            log.info("%s=%s", key, value)


def progress(
    iterable: Iterable[Any] | None = None,
    *,
    total: int | None = None,
    desc: str = "",
    unit: str = "it",
    leave: bool = True,
) -> tqdm:
    """A configured tqdm bar (stderr, stable format, no notebook widgets)."""
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        leave=leave,
        dynamic_ncols=True,
        file=sys.stderr,
        smoothing=0.05,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )


class LoggedProgress:
    """tqdm bar that also writes progress checkpoints into the log file.

    Usage::

        with LoggedProgress(total=n, desc="generate", log_every=0.1) as bar:
            for ...:
                bar.update(k)
    """

    def __init__(
        self,
        total: int,
        desc: str,
        *,
        unit: str = "it",
        log_every: float = 0.1,
        logger_name: str = "progress",
    ) -> None:
        self.total = max(int(total), 0)
        self.desc = desc
        self.log = get_logger(logger_name)
        self.log_every = log_every
        self._next_threshold = log_every
        self._start = time.time()
        self.bar = progress(total=self.total, desc=desc, unit=unit)

    def __enter__(self) -> "LoggedProgress":
        self.log.info("[%s] start (total=%d)", self.desc, self.total)
        return self

    def update(self, n: int = 1, postfix: dict[str, Any] | None = None) -> None:
        self.bar.update(n)
        if postfix:
            self.bar.set_postfix(postfix, refresh=False)
        if self.total <= 0:
            return
        frac = self.bar.n / self.total
        if frac >= self._next_threshold or self.bar.n >= self.total:
            elapsed = time.time() - self._start
            rate = self.bar.n / elapsed if elapsed > 0 else 0.0
            eta = (self.total - self.bar.n) / rate if rate > 0 else float("nan")
            self.log.info(
                "[%s] %5.1f%% (%d/%d) elapsed=%.0fs rate=%.2f/s eta=%.0fs%s",
                self.desc, 100 * frac, self.bar.n, self.total, elapsed, rate, eta,
                f" {postfix}" if postfix else "",
            )
            while self._next_threshold <= frac:
                self._next_threshold += self.log_every

    def set_postfix(self, **kwargs: Any) -> None:
        self.bar.set_postfix(kwargs, refresh=False)

    def close(self) -> None:
        self.bar.close()
        elapsed = time.time() - self._start
        self.log.info("[%s] done (%d/%d) in %.1fs", self.desc, self.bar.n, self.total, elapsed)

    def __exit__(self, *exc: Any) -> None:
        self.close()


@contextmanager
def timed(desc: str, logger_name: str = "timing") -> Iterator[None]:
    log = get_logger(logger_name)
    log.info("%s ...", desc)
    start = time.time()
    try:
        yield
    finally:
        log.info("%s finished in %.1fs", desc, time.time() - start)


def log_table(rows: Sequence[Sequence[Any]], headers: Sequence[str], logger_name: str = "table") -> None:
    """Log a small fixed-width table (readable in the log file and console)."""
    log = get_logger(logger_name)
    cols = [str(h) for h in headers]
    widths = [len(c) for c in cols]
    str_rows = [[("" if v is None else str(v)) for v in row] for row in rows]
    for row in str_rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))
    sep = "-+-".join("-" * w for w in widths)
    log.info(" | ".join(c.ljust(w) for c, w in zip(cols, widths)))
    log.info(sep)
    for row in str_rows:
        log.info(" | ".join(cell.ljust(w) for cell, w in zip(row, widths)))
