"""LLM paraphrase attack: lookup attacker over a precomputed adversarial set.

Pre-generation pipeline (see ``scripts/{mbpp,humaneval,canitedit}_llm_*.py``)
writes one JSONL file per dataset where each row carries the original prompt
and the LLM-paraphrased adversarial prompt for one task. Those JSONL files
are intended to be uploaded to the Hugging Face Hub.

At evaluation time this attack does NOT call any LLM. Instead it:

1. Loads every adversarial row from any combination of: a single JSONL file,
   a directory of JSONL files, or a HF dataset repo with one config per
   dataset (``mbpp`` / ``humaneval`` / ``canitedit``).
2. Builds an ``input_text -> adversarial_prompt`` lookup. The keying
   strategy is decided per-row from ``row["dataset"]`` (or inferred from
   the ``task_id`` prefix when missing) so the attacker does NOT need a
   ``dataset`` config key — it matches the interface of the other attacks.
   * For ``mbpp``/``humaneval`` rows the key is the raw ``original_prompt``,
     which is exactly what the framework passes in.
   * For ``canitedit`` rows the framework first runs
     ``mask_protected_content`` on the prompt, so we mask each row's
     original at load time to mirror that behavior.
3. Returns the precomputed adversarial prompt on lookup, or falls back to
   the input unchanged when the task is missing from the set.

Source resolution order (first hit wins):
    1. ``config["local_path"]``  — file or directory.
    2. ``config["hub_repo"]``    — HF dataset repo id.
    3. ``$LLM_PARAPHRASE_PATH``  — env var, file or directory.
    4. ``$LLM_PARAPHRASE_REPO``  — env var, HF dataset repo id.
    5. ``DEFAULT_HUB_REPO`` (this module) — published adversarial set.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .base_attack import BaseAttack

_VALID_INPUT_TYPES = {"prompt", "code", "instruction"}
_KNOWN_DATASETS: Tuple[str, ...] = ("mbpp", "humaneval", "canitedit")

# Default published adversarial set on the Hugging Face Hub.
DEFAULT_HUB_REPO = "TheFatBlue/llm-attacked-prompts-clm"


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Yield decoded objects from a JSONL file, skipping blank lines."""
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _infer_dataset_from_task_id(task_id: str) -> str:
    """Best-effort dataset inference from a task_id prefix."""
    if task_id.startswith("Mbpp/"):
        return "mbpp"
    if task_id.startswith("HumanEval/"):
        return "humaneval"
    # CanItEdit task ids look like "<full_name>/instruction_descriptive" or
    # "<full_name>/instruction_lazy"; treat anything unrecognized as canitedit.
    return "canitedit"


class LLMParaphraseAttack(BaseAttack):
    """Lookup-only attack returning precomputed LLM-paraphrased prompts.

    Configuration keys:

    Required:
        ``input_type``: ``"prompt"`` | ``"code"`` | ``"instruction"``.
            Set automatically by ``AttackEvaluator`` from ``--dataset``.

    Optional source overrides (any one — first hit wins, defaults below):
        ``local_path``: path to a JSONL file or a directory of ``*.jsonl``
            files matching the writer schema.
        ``hub_repo``:   HF dataset repo id with one config per dataset.
            ``hub_split`` (default ``"test"``) and ``hub_revision`` further
            qualify it.

    Optional:
        ``strict``: when True (default) raise on a missing task at lookup time;
                    when False, silently return the input unchanged.
    """

    # ------------------------------------------------------------------
    # Construction / validation
    # ------------------------------------------------------------------

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.input_type: str = self.config["input_type"]
        self.strict: bool = bool(self.config.get("strict", True))

        rows = self._load_rows()
        self._lookup, self._task_lookup = self._build_lookup(rows)
        print(
            f"[LLMParaphraseAttack] Loaded {len(self._lookup)} adversarial prompts "
            f"({len(self._task_lookup)} task ids), input_type={self.input_type}"
        )

    def validate_config(self) -> None:
        if "input_type" not in self.config:
            raise ValueError("LLMParaphraseAttack: config missing key 'input_type'")
        if self.config["input_type"] not in _VALID_INPUT_TYPES:
            raise ValueError(
                f"LLMParaphraseAttack: input_type must be one of {sorted(_VALID_INPUT_TYPES)}, "
                f"got {self.config['input_type']!r}"
            )

    # ------------------------------------------------------------------
    # Source resolution
    # ------------------------------------------------------------------

    def _resolve_source(self) -> Tuple[str, str]:
        """Return ``(kind, value)`` where kind is ``"local"`` or ``"hub"``.

        Falls back to ``DEFAULT_HUB_REPO`` so a bare
        ``--attack_method llm_paraphrase`` Just Works without any extra
        configuration.
        """
        local = self.config.get("local_path") or os.environ.get("LLM_PARAPHRASE_PATH")
        if local:
            return "local", local

        hub = (
            self.config.get("hub_repo")
            or os.environ.get("LLM_PARAPHRASE_REPO")
            or DEFAULT_HUB_REPO
        )
        return "hub", hub

    def _load_rows(self) -> List[Dict[str, Any]]:
        """Pull rows from a local file/dir or a HF Hub repo."""
        kind, value = self._resolve_source()
        if kind == "local":
            return self._load_local(value)
        return self._load_hub(value)

    @staticmethod
    def _load_local(path_str: str) -> List[Dict[str, Any]]:
        p = Path(path_str)
        if p.is_file():
            return list(_iter_jsonl(p))
        if p.is_dir():
            rows: List[Dict[str, Any]] = []
            for jp in sorted(p.glob("*.jsonl")):
                rows.extend(_iter_jsonl(jp))
            if not rows:
                raise FileNotFoundError(
                    f"LLMParaphraseAttack: no .jsonl files found under directory {p}"
                )
            return rows
        raise FileNotFoundError(f"LLMParaphraseAttack: source path not found: {p}")

    def _load_hub(self, repo_id: str) -> List[Dict[str, Any]]:
        from datasets import load_dataset  # lazy: optional dependency at import time

        split = self.config.get("hub_split", "test")
        revision = self.config.get("hub_revision")
        kwargs: Dict[str, Any] = {"split": split}
        if revision is not None:
            kwargs["revision"] = revision

        rows: List[Dict[str, Any]] = []
        for cfg in _KNOWN_DATASETS:
            try:
                ds = load_dataset(repo_id, cfg, **kwargs)
            except Exception:  # noqa: BLE001 - config may not exist; try next
                continue
            rows.extend(dict(r) for r in ds)

        if not rows:
            raise RuntimeError(
                f"LLMParaphraseAttack: no rows loaded from HF repo {repo_id!r}; "
                f"expected at least one of configs {_KNOWN_DATASETS} on split {split!r}."
            )
        return rows

    # ------------------------------------------------------------------
    # Lookup construction
    # ------------------------------------------------------------------

    def _build_lookup(
        self, rows: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Build prompt-keyed and task_id-keyed dicts from raw rows.

        Per-row dataset is taken from ``row["dataset"]`` when present, otherwise
        inferred from the ``task_id`` prefix. Canitedit rows are remasked so
        their key matches the masked prompt the framework passes in.
        """
        from src.utils.content_protection import mask_protected_content

        lookup: Dict[str, str] = {}
        task_lookup: Dict[str, str] = {}

        for row in rows:
            try:
                orig = row["original_prompt"]
                adv = row["adversarial_prompt"]
            except KeyError as exc:
                raise ValueError(
                    f"LLMParaphraseAttack: malformed row missing required field {exc}"
                ) from exc

            task_id = row.get("task_id", "")
            ds = row.get("dataset") or _infer_dataset_from_task_id(task_id)

            if ds == "canitedit":
                masked, _ = mask_protected_content(orig)
                lookup[masked] = adv
            else:
                lookup[orig] = adv

            if task_id:
                task_lookup[task_id] = adv

        return lookup, task_lookup

    # ------------------------------------------------------------------
    # BaseAttack API
    # ------------------------------------------------------------------

    def generate_adversarial_example(
        self,
        input_code: str,
        target_label: Optional[Any] = None,
    ) -> str:
        """Return the precomputed adversarial prompt for *input_code*.

        Falls back to ``input_code`` unchanged when ``strict`` is False and no
        match is found; raises otherwise.
        """
        if input_code in self._lookup:
            return self._lookup[input_code]
        if self.strict:
            raise KeyError(
                "LLMParaphraseAttack: no adversarial prompt found for the given input. "
                "Make sure the precomputed adversarial set was generated against the "
                "same dataset version."
            )
        return input_code

    # Convenience: callers that already know the task_id can bypass the
    # text-keyed lookup. Not used by the framework but useful for tests.
    def get_by_task_id(self, task_id: str) -> Optional[str]:
        """Return the adversarial prompt for *task_id*, or None if absent."""
        return self._task_lookup.get(task_id)
