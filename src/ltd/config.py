"""Typed configuration loaded from ``configs/default.yaml``."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from .env import CONFIGS_DIR

CONDITION_LABELS = {
    "target": "Target-CoT",
    "en_cot": "En-CoT",
    "free": "Free-CoT",
    "mixed": "Mixed-CoT",
}
CONDITION_ORDER = ["target", "en_cot", "free", "mixed"]


@dataclass(frozen=True)
class LanguageSpec:
    code: str
    name: str
    script: str
    family: str
    mgsm_source: str
    gmmlu_config: str
    typ_distance: float
    resource_proxy_gb: float
    resource_tier: str
    is_anchor: bool = False


@dataclass(frozen=True)
class TaskSpec:
    name: str
    kind: str  # "numeric" | "mcq"
    n_items: int
    hf_dataset: str
    hf_split: str
    max_new_tokens: int
    cultural_sensitivity_label: str | None = None
    hindi_bridge: dict[str, str] | None = None


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    family: str  # "r1_distill" | "qwen3"
    thinking: str  # "always" | "switchable"
    dtype: str
    max_model_len: int
    decoding: dict[str, float]

    @property
    def local_dirname(self) -> str:
        return self.hf_id.replace("/", "__")


@dataclass
class GenerationSpec:
    backend: str = "auto"
    n_samples: int = 1
    chunk_size: int = 64
    gpu_memory_utilization: float = 0.90
    max_num_seqs: int = 128
    swap_space_gb: int = 4
    enforce_eager: bool = False
    tensor_parallel_size: int = 1
    hf_batch_size: int = 4
    model_isolation: str = "auto"


@dataclass
class ScoringSpec:
    compliance_threshold: float = 0.50
    anchor_threshold: float = 0.05
    min_anchor_tokens: int = 1
    faithfulness_subsample: int = 50
    min_anchor_token_len: int = 4
    bootstrap_iters: int = 10000
    bootstrap_alpha: float = 0.05


@dataclass
class Config:
    seed: int
    languages: dict[str, LanguageSpec]
    tasks: dict[str, TaskSpec]
    models: dict[str, ModelSpec]
    conditions: list[str]
    instruction_language: str
    default_languages: list[str]
    default_tasks: list[str]
    default_models: list[str]
    generation: GenerationSpec
    scoring: ScoringSpec
    raw: dict[str, Any] = field(default_factory=dict, repr=False)
    source_path: Path | None = None

    # -- lookups ----------------------------------------------------------
    def language(self, code: str) -> LanguageSpec:
        try:
            return self.languages[code]
        except KeyError:
            raise KeyError(
                f"unknown language {code!r}; known: {sorted(self.languages)}"
            ) from None

    def task(self, name: str) -> TaskSpec:
        try:
            return self.tasks[name]
        except KeyError:
            raise KeyError(f"unknown task {name!r}; known: {sorted(self.tasks)}") from None

    def model(self, key: str) -> ModelSpec:
        if key in self.models:
            return self.models[key]
        for spec in self.models.values():  # allow passing the full HF id
            if spec.hf_id == key:
                return spec
        raise KeyError(f"unknown model {key!r}; known: {sorted(self.models)}")

    def resolve_languages(self, codes: Iterable[str] | None) -> list[str]:
        return self._resolve(codes, self.default_languages, self.languages, "language")

    def resolve_tasks(self, names: Iterable[str] | None) -> list[str]:
        return self._resolve(names, self.default_tasks, self.tasks, "task")

    def resolve_models(self, keys: Iterable[str] | None) -> list[str]:
        if not keys:
            return list(self.default_models)
        out = []
        for key in keys:
            if key == "all":
                out.extend(self.models)
            else:
                out.append(self.model(key).key)
        return _dedup(out)

    def resolve_conditions(self, names: Iterable[str] | None) -> list[str]:
        if not names:
            return list(self.conditions)
        out = []
        for name in names:
            if name == "all":
                out.extend(self.conditions)
                continue
            if name not in CONDITION_LABELS:
                raise KeyError(
                    f"unknown condition {name!r}; known: {sorted(CONDITION_LABELS)}"
                )
            out.append(name)
        return _dedup(out)

    @staticmethod
    def _resolve(
        requested: Iterable[str] | None,
        default: list[str],
        universe: dict[str, Any],
        what: str,
    ) -> list[str]:
        if not requested:
            return list(default)
        out: list[str] = []
        for item in requested:
            if item == "all":
                out.extend(universe)
                continue
            if item not in universe:
                raise KeyError(f"unknown {what} {item!r}; known: {sorted(universe)}")
            out.append(item)
        return _dedup(out)

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.raw, default=str))


def _dedup(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def load_config(path: str | Path | None = None) -> Config:
    cfg_path = Path(path) if path else CONFIGS_DIR / "default.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    languages = {
        code: LanguageSpec(code=code, **spec) for code, spec in raw["languages"].items()
    }
    tasks = {
        name: TaskSpec(name=name, **spec) for name, spec in raw["tasks"].items()
    }
    models = {
        key: ModelSpec(key=key, **spec) for key, spec in raw["models"].items()
    }
    gen_raw = dict(raw.get("generation", {}))
    scoring_raw = dict(raw.get("scoring", {}))

    return Config(
        seed=int(raw["seed"]),
        languages=languages,
        tasks=tasks,
        models=models,
        conditions=list(raw["conditions"]),
        instruction_language=raw.get("instruction_language", "en"),
        default_languages=list(raw["default_languages"]),
        default_tasks=list(raw["default_tasks"]),
        default_models=list(raw["default_models"]),
        generation=GenerationSpec(**gen_raw),
        scoring=ScoringSpec(**scoring_raw),
        raw=raw,
        source_path=cfg_path,
    )
