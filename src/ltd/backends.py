"""Generation backends.

``vllm`` is the backend for the 48 GB GPU box. ``hf`` is a transformers fallback
so the whole pipeline can be exercised on a laptop (CPU/MPS) before shipping.

Both expose the same contract::

    backend.generate(prompts, max_new_tokens, seed) -> list[Generation]

and generation is driven in *chunks* by :mod:`ltd.generate`, which flushes
predictions to disk after every chunk. That is what makes a run resumable: the
backend itself is stateless.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import Any, Sequence

from .config import GenerationSpec, ModelSpec
from .logging_utils import get_logger
from .models_fetch import resolve_model_path

log = get_logger("backend")


@dataclass
class Generation:
    text: str
    n_prompt_tokens: int
    n_completion_tokens: int
    finish_reason: str

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


def vllm_available() -> bool:
    try:
        import vllm  # noqa: F401
    except Exception:  # ImportError, or a broken CUDA install
        return False
    return True


def _vllm_version() -> str:
    try:
        import vllm

        return str(vllm.__version__)
    except Exception:
        return "unknown"


def _engine_supports(arg: str) -> bool:
    """Whether the installed vLLM's ``EngineArgs`` accepts ``arg``.

    ``LLM(**kwargs)`` forwards everything to ``EngineArgs``, so an argument the
    running version has retired raises ``TypeError`` at construction time.
    """
    import dataclasses

    try:
        from vllm.engine.arg_utils import EngineArgs
    except Exception:
        return False
    return any(f.name == arg for f in dataclasses.fields(EngineArgs))


def _flashinfer_can_jit() -> bool:
    """Whether FlashInfer can JIT-compile kernels for the local GPU.

    FlashInfer builds with the *system* nvcc, not the toolkit torch was built
    against, and it refuses to target sm120 (Blackwell) unless CUDA >= 12.9.
    On such a box ``TARGET_CUDA_ARCHS`` comes back empty and every JIT build
    dies with the misleading "requires GPUs with sm75 or higher".
    """
    try:
        from flashinfer.compilation_context import CompilationContext
    except Exception:  # not installed — vLLM will pick its own fallback
        return True
    try:
        return bool(CompilationContext().TARGET_CUDA_ARCHS)
    except Exception:
        return False


def _guard_flashinfer_sampler() -> None:
    """Fall back to the Torch sampler when FlashInfer's cannot be built.

    vLLM defaults to the FlashInfer top-k/top-p sampler and JIT-compiles it on
    first use, i.e. during engine startup, so an unbuildable FlashInfer takes
    the whole engine down. Set the opt-out before the engine (and the
    ``EngineCore`` subprocess that inherits this environment) starts.
    """
    import os

    if os.environ.get("VLLM_USE_FLASHINFER_SAMPLER") is not None:
        return  # respect an explicit operator choice
    if _flashinfer_can_jit():
        return
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    log.warning(
        "FlashInfer cannot compile for this GPU (system nvcc is too old for its "
        "compute capability) — using the Torch top-k/top-p sampler instead. "
        "Sampling stays seeded and reproducible; only the kernel differs."
    )


def choose_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    if vllm_available():
        return "vllm"
    log.warning(
        "vLLM unavailable (platform=%s) — falling back to the transformers backend. "
        "This is fine for dry runs but far too slow for the full experiment.",
        platform.system(),
    )
    return "hf"


class BaseBackend:
    name = "base"

    def __init__(self, model: ModelSpec, generation: GenerationSpec) -> None:
        self.model = model
        self.generation = generation
        self.tokenizer: Any = None

    def generate(self, prompts: Sequence[str], max_new_tokens: int, seed: int) -> list[Generation]:
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        if self.tokenizer is None:
            return len(text.split())
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def shutdown(self) -> None:
        pass

    # -- shared config --------------------------------------------------
    def _decoding(self) -> dict[str, float]:
        return dict(self.model.decoding)


class VLLMBackend(BaseBackend):
    name = "vllm"

    def __init__(self, model: ModelSpec, generation: GenerationSpec) -> None:
        super().__init__(model, generation)
        _guard_flashinfer_sampler()
        from transformers import AutoTokenizer
        from vllm import LLM

        path = resolve_model_path(model)
        log.info("vLLM: loading %s (dtype=%s, max_model_len=%d)", path, model.dtype, model.max_model_len)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        kwargs: dict[str, Any] = dict(
            model=path,
            tokenizer=path,
            dtype=model.dtype,
            max_model_len=model.max_model_len,
            gpu_memory_utilization=generation.gpu_memory_utilization,
            max_num_seqs=generation.max_num_seqs,
            enforce_eager=generation.enforce_eager,
            tensor_parallel_size=generation.tensor_parallel_size,
            seed=0,
            trust_remote_code=False,
            disable_log_stats=True,
        )
        # ``swap_space`` (CPU preemption buffer) was dropped in the vLLM V1 engine,
        # which recomputes preempted sequences instead of swapping them out.
        # Pass it only where the installed vLLM still accepts it.
        if _engine_supports("swap_space"):
            kwargs["swap_space"] = generation.swap_space_gb
        else:
            log.info("vLLM %s ignores swap_space_gb (no CPU swap in the V1 engine)", _vllm_version())
        self.llm = LLM(**kwargs)
        log.info("vLLM ready for %s", model.key)

    def generate(self, prompts: Sequence[str], max_new_tokens: int, seed: int) -> list[Generation]:
        from vllm import SamplingParams

        decoding = self._decoding()
        params = SamplingParams(
            n=1,
            temperature=decoding.get("temperature", 0.6),
            top_p=decoding.get("top_p", 0.95),
            top_k=int(decoding.get("top_k", -1)),
            min_p=decoding.get("min_p", 0.0),
            max_tokens=max_new_tokens,
            seed=seed,
        )
        outputs = self.llm.generate(list(prompts), params, use_tqdm=False)
        results: list[Generation] = []
        for output in outputs:
            completion = output.outputs[0]
            results.append(
                Generation(
                    text=completion.text,
                    n_prompt_tokens=len(output.prompt_token_ids or []),
                    n_completion_tokens=len(completion.token_ids or []),
                    finish_reason=str(completion.finish_reason or "unknown"),
                )
            )
        return results

    def shutdown(self) -> None:
        """Release the GPU fully, so the next model can be loaded.

        ``del llm`` alone is *not* enough. vLLM pre-allocates the KV cache at
        ``gpu_memory_utilization`` (0.90 → ~43 GB of a 48 GB card), and that memory
        stays resident while any reference to the engine survives. Several do:
        the engine core (a separate process under V1), the model executor, and
        vLLM's global distributed state. Each is torn down here, in order, before
        the collector runs — otherwise loading the second model OOMs immediately.

        Every step is best-effort: vLLM's internals differ across V0/V1 and across
        releases, so a missing attribute must not abort the teardown.
        """
        import contextlib
        import gc

        llm = getattr(self, "llm", None)

        if llm is not None:
            engine = getattr(llm, "llm_engine", None)
            if engine is not None:
                # V1: the engine core owns worker processes; ask it to stop.
                for method in ("shutdown", "close"):
                    fn = getattr(engine, method, None)
                    if callable(fn):
                        with contextlib.suppress(Exception):
                            fn()
                core = getattr(engine, "engine_core", None)
                if core is not None:
                    with contextlib.suppress(Exception):
                        core.shutdown()
                # V0: the model executor holds the weights and the KV cache.
                with contextlib.suppress(Exception):
                    del engine.model_executor
                with contextlib.suppress(Exception):
                    del llm.llm_engine

        with contextlib.suppress(Exception):
            del self.llm
        self.llm = None
        self.tokenizer = None

        # vLLM's module-level parallel state keeps references to the workers.
        with contextlib.suppress(Exception):
            from vllm.distributed.parallel_state import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )

            destroy_model_parallel()
            destroy_distributed_environment()
        with contextlib.suppress(Exception):
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()

        # Only now is collection meaningful.
        gc.collect()
        with contextlib.suppress(Exception):
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
        gc.collect()
        log.info("vLLM backend for %s torn down", self.model.key)


class HFBackend(BaseBackend):
    """transformers backend: correctness-equivalent, throughput-irrelevant."""

    name = "hf"

    def __init__(self, model: ModelSpec, generation: GenerationSpec) -> None:
        super().__init__(model, generation)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = resolve_model_path(model)
        if torch.cuda.is_available():
            device, dtype = "cuda", getattr(torch, model.dtype)
        elif torch.backends.mps.is_available():
            device, dtype = "mps", torch.float16
        else:
            device, dtype = "cpu", torch.float32
        log.info("transformers: loading %s on %s (%s)", path, device, dtype)

        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.net = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device)
        self.net.eval()

    def generate(self, prompts: Sequence[str], max_new_tokens: int, seed: int) -> list[Generation]:
        torch = self.torch
        decoding = self._decoding()
        results: list[Generation] = []
        batch_size = max(1, self.generation.hf_batch_size)

        for start in range(0, len(prompts), batch_size):
            batch = list(prompts[start : start + batch_size])
            torch.manual_seed(seed + start)
            encoded = self.tokenizer(
                batch, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(self.device)
            top_k = int(decoding.get("top_k", -1))
            with torch.no_grad():
                out = self.net.generate(
                    **encoded,
                    do_sample=True,
                    temperature=decoding.get("temperature", 0.6),
                    top_p=decoding.get("top_p", 0.95),
                    top_k=top_k if top_k > 0 else 0,
                    min_p=decoding.get("min_p", 0.0) or None,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            prompt_len = encoded["input_ids"].shape[1]
            for row in range(len(batch)):
                completion_ids = out[row][prompt_len:]
                # Strip left padding influence: count non-pad prompt tokens.
                n_prompt = int(encoded["attention_mask"][row].sum().item())
                real = [
                    int(t) for t in completion_ids.tolist()
                    if t != self.tokenizer.pad_token_id
                ]
                text = self.tokenizer.decode(real, skip_special_tokens=True)
                finish = "length" if len(real) >= max_new_tokens else "stop"
                results.append(
                    Generation(
                        text=text,
                        n_prompt_tokens=n_prompt,
                        n_completion_tokens=len(real),
                        finish_reason=finish,
                    )
                )
        return results

    def shutdown(self) -> None:
        """Drop the weights and free the accelerator, mirroring VLLMBackend."""
        import contextlib
        import gc

        with contextlib.suppress(Exception):
            del self.net
        self.net = None
        self.tokenizer = None
        gc.collect()
        with contextlib.suppress(Exception):
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
        log.info("transformers backend for %s torn down", self.model.key)



def build_backend(model: ModelSpec, generation: GenerationSpec, requested: str = "auto") -> BaseBackend:
    kind = choose_backend(requested)
    if kind == "vllm":
        return VLLMBackend(model, generation)
    if kind == "hf":
        return HFBackend(model, generation)
    raise ValueError(f"unknown backend {kind!r} (expected auto|vllm|hf)")
