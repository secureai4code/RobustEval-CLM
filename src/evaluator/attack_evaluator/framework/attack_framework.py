import gzip
import json
import os
import tempfile
from typing import Any, Dict, List, Optional

from evalplus.data import get_human_eval_plus, get_mbpp_plus

from src.core.datasets.dataset_wrapper import AdversarialDatasetWrapper
from src.core.models.base_model import BaseModel
from src.evaluator.attack_evaluator.attack_registry import AttackRegistry
from src.evaluator.attack_evaluator.attacks.base_attack import BaseAttack
from src.evaluator.utils.evaluation import canitedit_evaluator, evaluator
from src.utils.content_protection import mask_protected_content, restore
from src.utils.function_extractor import extract_code_from_markdown, extract_functions

# ---------------------------------------------------------------------------
# CanItEdit helper utilities
# ---------------------------------------------------------------------------


def _build_edit_prompt(old: str, instr: str) -> str:
    """Build the zero-shot edit prompt for CanItEdit."""
    return (
        "You are PythonEditGPT. You will be provided the original code snippet "
        "and an instruction that specifies the changes you need to make. You will "
        "produce the changed code, based on the original code and the instruction "
        "given. Only produce the code, do not include any additional prose.\n\n"
        "## Code Before\n```py\n" + old + "\n```\n\n"
        "## Instruction\n" + instr + "\n\n"
        "## Code After\n"
    )


def _write_pass_rates(results: Dict[str, Any], save_results: str) -> None:
    """Write pass_rates.json ({"base": <pass@1>, "plus": <pass@1>}) for a results dict.

    This is the compact summary consumed by the analysis scripts
    (see scripts/generate_all.py and the outputs layout in the README).
    """
    pass_at_k = results.get("pass_at_k") or {}
    rates = {
        variant: scores["pass@1"]
        for variant, scores in pass_at_k.items()
        if isinstance(scores, dict) and "pass@1" in scores
    }
    if rates:
        with open(os.path.join(save_results, "pass_rates.json"), "w") as fh:
            json.dump(rates, fh)


class AttackFramework:
    """Orchestrates adversarial attack generation and model evaluation."""

    def __init__(
        self,
        model: BaseModel,
        attack_method: str = "synonym",
        attack_config: Dict[str, Any] = None,
        dataset: str = "humaneval",
        mini: bool = False,
        attacker: Optional[BaseAttack] = None,
    ):
        """
        Initialize attack framework.

        Args:
            model: Model to attack.
            attack_method: Name of the attack to use.
            attack_config: Attack configuration dictionary.
            dataset: Dataset to use ("humaneval" or "mbpp").
            mini: Whether to use the mini version of the dataset.
            attacker: Pre-instantiated attack object.  When provided, *attack_method*
                and *attack_config* are only used for book-keeping; no new attacker is
                created.  Pass this when the attacker must be initialised *before* the
                main model (e.g. translation attacks that load a HuggingFace model on
                the same GPU).
        """
        self.model = model
        self.attack_method = attack_method
        self.attack_config = attack_config or {}
        self.dataset = dataset.lower()
        self.mini = mini

        # Load dataset and derive input_type
        if self.dataset == "humaneval":
            self.problems = get_human_eval_plus(mini=mini)
            self.attack_config["input_type"] = "code"
            self.concat_prompt = True
        elif self.dataset == "mbpp":
            self.problems = get_mbpp_plus(mini=mini)
            self.attack_config["input_type"] = "prompt"
            self.concat_prompt = False
        elif self.dataset == "canitedit":
            self._canitedit_examples, self.problems = self._load_canitedit_dataset()
            self.attack_config["input_type"] = "instruction"
            self.concat_prompt = False
        else:
            raise ValueError(
                f"Unknown dataset: {dataset}. Choose 'humaneval', 'mbpp', or 'canitedit'"
            )

        # Use the pre-created attacker when one is supplied; otherwise build from registry.
        if attacker is not None:
            self.attacker = attacker
        else:
            attack_class = AttackRegistry.get(attack_method)
            self.attacker = attack_class(self.attack_config)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_solution(self, output: str, prompt: str) -> str:
        """Extract solution code based on dataset type.

        For mbpp/humaneval: uses extract_functions (prompt+output when concat_prompt).
        For canitedit: uses extract_code_from_markdown.
        """
        if self.dataset == "canitedit":
            extracted = extract_code_from_markdown(output)
        elif self.dataset in ("humaneval", "mbpp"):
            code_to_extract = prompt + output if self.concat_prompt else output
            extracted = extract_functions(code_to_extract)
        else:
            extracted = output
        return extracted if extracted is not None else output

    def _apply_noise_to_model(self):
        """Apply noise attack to the model and return the modified model."""
        return self.attacker.apply_noise(self.model)

    def _is_bigcode_family(self) -> bool:
        """Whether the generation model belongs to the bigcode (StarCoder) family."""
        model_path = (getattr(self.model, "model_path", "") or "").lower()
        return any(name in model_path for name in ("bigcode", "starcoder", "santacoder"))

    def _finalize_prompt(self, prompt: str) -> str:
        """Apply model-family-specific prompt fixes before generation.

        The bigcode family (StarCoder/SantaCoder) expects HumanEval prompts without a
        trailing newline; keeping it degrades completion quality. Applied to both the
        original and adversarial batches so they stay on equal footing.
        """
        if self.dataset == "humaneval" and self._is_bigcode_family():
            return prompt.rstrip("\n")
        return prompt

    def _build_adversarial_prompts(self, problems: list) -> dict:
        """Build adversarial prompts for each problem."""
        adversarial_prompts = {}
        for task_id, problem in problems:
            prompt = problem["prompt"]
            if self.dataset == "canitedit":
                masked, placeholders = mask_protected_content(prompt)
                adv = self.attacker.generate_adversarial_example(masked)
                adversarial_prompts[task_id] = restore(adv, placeholders)
            else:
                adversarial_prompts[task_id] = self.attacker.generate_adversarial_example(prompt)
        return adversarial_prompts

    def _build_llm_adversarial_prompts(self, problems: list, generator) -> dict:
        """Build adversarial prompts via an LLM wrapper."""
        index_dict = {}
        prompts = []
        placeholders_by_task = {}

        for task_id, problem in problems:
            prompt = problem["prompt"]
            if self.dataset == "canitedit":
                masked, placeholders = mask_protected_content(prompt)
                prompts.append(masked)
                index_dict[masked] = task_id
                placeholders_by_task[task_id] = placeholders
            else:
                prompts.append(prompt)
                index_dict[prompt] = task_id

        adversarial_generation = generator.generate_dataset(prompts, self.attack_config["attack_type"])
        adversarial_prompts = {}
        for prompt, adv in adversarial_generation:
            task_id = index_dict[prompt]
            if self.dataset == "canitedit":
                adv = restore(adv, placeholders_by_task[task_id])
            adversarial_prompts[task_id] = adv
        return adversarial_prompts

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_attack(
        self,
        sample_indices: Optional[List[int]] = None,
        save_prompts: str = None,
        save_results: str = None,
        gen_ori: bool = False,
    ):
        """
        Run the full attack pipeline on the selected problems.

        Args:
            sample_indices: Optional list of problem indices to attack.
                            If None, all problems are used.
            save_prompts: Directory path for saving generated prompts.
            save_results: Directory path for saving evaluation results.
            gen_ori: When True, also generate outputs for the original (unperturbed) inputs.

        Returns:
            Tuple ``(original_results, adversarial_results)`` when *gen_ori* is True,
            otherwise just ``adversarial_results``.
        """
        is_canitedit = self.dataset == "canitedit"

        def _ori_prompt(problem):
            """Get the generation prompt for the original (unperturbed) input."""
            if is_canitedit:
                return _build_edit_prompt(problem["before"], problem["prompt"])
            return problem["prompt"]

        problems_to_attack = (
            list(self.problems.items())
            if sample_indices is None
            else [(k, v) for i, (k, v) in enumerate(self.problems.items()) if i in sample_indices]
        )

        original_generations_dict: Dict[str, Any] = {}
        adversarial_generations_dict: Dict[str, Any] = {}

        if save_prompts:
            os.makedirs(save_prompts, exist_ok=True)
        adv_prompt_file = os.path.join(save_prompts, "adversarial_prompts.jsonl") if save_prompts else None
        ori_prompt_file = os.path.join(save_prompts, "original_prompts.jsonl") if save_prompts else None

        if ori_prompt_file and os.path.exists(ori_prompt_file):
            try:
                with open(ori_prompt_file, "r") as fh:
                    for line in fh:
                        try:
                            data = json.loads(line)
                            if "task_id" in data:
                                original_generations_dict[data["task_id"]] = data
                        except json.JSONDecodeError:
                            continue
                print(f"Loaded {len(original_generations_dict)} existing original generations")
            except OSError as exc:
                print(f"Error reading {ori_prompt_file}: {exc}")

        if adv_prompt_file and os.path.exists(adv_prompt_file):
            try:
                with open(adv_prompt_file, "r") as fh:
                    for line in fh:
                        try:
                            data = json.loads(line)
                            if "task_id" in data and data.get("solution") is not None:
                                adversarial_generations_dict[data["task_id"]] = data
                        except json.JSONDecodeError:
                            continue
                print(f"Loaded {len(adversarial_generations_dict)} existing adversarial generations")
            except OSError as exc:
                print(f"Error reading {adv_prompt_file}: {exc}")

        # Build adversarial prompts
        if self.attack_method != "llm_attack":
            adversarial_prompts = self._build_adversarial_prompts(problems_to_attack)
        else:
            attack_wrapper = AdversarialDatasetWrapper(attack_model=self.attacker)
            adversarial_prompts = self._build_llm_adversarial_prompts(problems_to_attack, attack_wrapper)
            assert len(adversarial_prompts) == len(problems_to_attack), (
                "Adversarial prompts not generated correctly"
            )

        # For canitedit the attack was applied to the raw instruction text;
        # wrap the attacked instructions in the full edit prompt template. An attacker
        # may override the wrapper (e.g. the destructure attack strips its structure)
        # by exposing a wrap_edit_prompt(before, instruction) method.
        if is_canitedit:
            wrap_edit_prompt = getattr(self.attacker, "wrap_edit_prompt", None)
            build_prompt = wrap_edit_prompt if callable(wrap_edit_prompt) else _build_edit_prompt
            for task_id, problem in problems_to_attack:
                adversarial_prompts[task_id] = build_prompt(
                    problem["before"], adversarial_prompts[task_id]
                )

        ori_prompt_f = None
        adv_prompt_f = None
        original_results = None

        try:
            if save_prompts:
                adv_prompt_f = open(adv_prompt_file, "a")
                if gen_ori:
                    ori_prompt_f = open(ori_prompt_file, "a")

            original_generations: List[Dict[str, Any]] = []
            adversarial_generations: List[Dict[str, Any]] = []

            skipped_orig = skipped_adv = new_orig = new_adv = 0

            # Original (unperturbed) generations
            if gen_ori:
                tasks_to_generate = []
                task_ids_to_generate = []

                for task_id, problem in problems_to_attack:
                    if task_id in original_generations_dict:
                        original_generations.append(original_generations_dict[task_id])
                        skipped_orig += 1
                    else:
                        tasks_to_generate.append(self._finalize_prompt(_ori_prompt(problem)))
                        task_ids_to_generate.append(task_id)

                if tasks_to_generate:
                    print(f"Generating {len(tasks_to_generate)} original outputs in batch...")
                    original_outputs = self.model.batch_generate(tasks_to_generate)
                    for task_id, prompt, output in zip(
                        task_ids_to_generate, tasks_to_generate, original_outputs
                    ):
                        solution = self._extract_solution(output, prompt)
                        entry = {"task_id": task_id, "solution": solution, "prompt": prompt}
                        new_orig += 1
                        if ori_prompt_f:
                            ori_prompt_f.write(json.dumps(entry) + "\n")
                            ori_prompt_f.flush()
                        original_generations.append(entry)

            # Apply noise to model weights (noise attack only)
            if self.attack_method == "noise":
                self.model = self.attacker.apply_noise(self.model)

            # Adversarial generations
            prompts_to_generate = []
            task_ids_to_generate = []

            for task_id, _ in problems_to_attack:
                if task_id in adversarial_generations_dict:
                    adversarial_generations.append(adversarial_generations_dict[task_id])
                    skipped_adv += 1
                else:
                    prompts_to_generate.append(self._finalize_prompt(adversarial_prompts[task_id]))
                    task_ids_to_generate.append(task_id)

            if prompts_to_generate:
                print(f"Generating {len(prompts_to_generate)} adversarial outputs in batch...")
                adversarial_outputs = self.model.batch_generate(prompts_to_generate)
                for task_id, prompt, output in zip(
                    task_ids_to_generate, prompts_to_generate, adversarial_outputs
                ):
                    solution = self._extract_solution(output, prompt)
                    entry = {"task_id": task_id, "solution": solution, "prompt": prompt}
                    new_adv += 1
                    if adv_prompt_f:
                        adv_prompt_f.write(json.dumps(entry) + "\n")
                        adv_prompt_f.flush()
                    adversarial_generations.append(entry)

            if gen_ori:
                print(f"Original outputs: {new_orig} newly generated, {skipped_orig} reused")
            print(f"Adversarial outputs: {new_adv} newly generated, {skipped_adv} reused")

            # Evaluate
            if is_canitedit:
                comp_base = save_results or tempfile.mkdtemp(prefix="canitedit_eval_")
                os.makedirs(comp_base, exist_ok=True)

                if gen_ori:
                    ori_dir = os.path.join(comp_base, "original_completions")
                    self._save_canitedit_completions(original_generations, ori_dir)
                    original_results = canitedit_evaluator(ori_dir)
                    if save_results:
                        with open(os.path.join(save_results, "original_results.json"), "w") as fh:
                            json.dump(original_results, fh)

                adv_dir = os.path.join(comp_base, "adversarial_completions")
                self._save_canitedit_completions(adversarial_generations, adv_dir)
                adversarial_results = canitedit_evaluator(adv_dir)
                if save_results:
                    with open(os.path.join(save_results, "adversarial_results.json"), "w") as fh:
                        json.dump(adversarial_results, fh)
                    _write_pass_rates(adversarial_results, save_results)
            else:
                if gen_ori and save_results:
                    original_results = evaluator(self.dataset, original_generations)
                    os.makedirs(save_results, exist_ok=True)
                    with open(os.path.join(save_results, "original_results.json"), "w") as fh:
                        json.dump(original_results, fh)

                adversarial_results = evaluator(self.dataset, adversarial_generations)
                if save_results:
                    os.makedirs(save_results, exist_ok=True)
                    with open(os.path.join(save_results, "adversarial_results.json"), "w") as fh:
                        json.dump(adversarial_results, fh)
                    _write_pass_rates(adversarial_results, save_results)

        finally:
            if ori_prompt_f:
                ori_prompt_f.close()
            if adv_prompt_f:
                adv_prompt_f.close()

        return (original_results, adversarial_results) if gen_ori else adversarial_results

    # ------------------------------------------------------------------
    # CanItEdit-specific methods
    # ------------------------------------------------------------------

    @staticmethod
    def _load_canitedit_dataset():
        """Load the nuprl/CanItEdit dataset and create problem entries.

        Returns:
            Tuple of (examples_dict, problems_dict) where:
            - examples_dict maps full_name -> original dataset row (dict)
            - problems_dict maps task_id -> problem info with 'prompt' set to
              the instruction text (for adversarial attack).
        """
        from datasets import load_dataset

        ds = load_dataset("nuprl/CanItEdit", split="test")
        examples: Dict[str, dict] = {}
        problems: Dict[str, dict] = {}

        for ex in ds:
            full_name = ex["full_name"]
            examples[full_name] = dict(ex)
            for instr_kind in ["instruction_descriptive", "instruction_lazy"]:
                task_id = f"{full_name}/{instr_kind}"
                problems[task_id] = {
                    "prompt": ex[instr_kind],
                    "before": ex["before"],
                    "after": ex["after"],
                    "full_name": full_name,
                    "instr_kind": instr_kind,
                }

        return examples, problems

    def _save_canitedit_completions(
        self,
        generations: List[Dict[str, Any]],
        output_dir: str,
    ) -> None:
        """Save canitedit generations as ``.json.gz`` files for Docker evaluation."""
        os.makedirs(output_dir, exist_ok=True)
        gen_by_id = {g["task_id"]: g for g in generations}

        for task_id, problem in self.problems.items():
            gen = gen_by_id.get(task_id)
            if gen is None or gen.get("solution") is None:
                continue

            ex = self._canitedit_examples[problem["full_name"]]
            result = dict(ex)
            result["instr_kind"] = problem["instr_kind"]
            result["prompt"] = ""
            result["completions"] = [gen["solution"]]
            result["language"] = "py"
            result["temperature"] = 0.0
            result["top_p"] = 1.0
            result["max_tokens"] = 1024
            result["stop_tokens"] = []

            fname = f"{problem['full_name']}_{problem['instr_kind']}.json.gz"
            out_path = os.path.join(output_dir, fname)
            with gzip.open(out_path, "wt") as f:
                json.dump(result, f)


if __name__ == "__main__":
    from src.models import CodeLLaMAModel

    model = CodeLLaMAModel(
        model_path="deepseek-ai/deepseek-coder-1.3b-base",
    )
    attack_config = {
        "attack_model": 'gpt-3.5-turbo',
        "temperature": 0.7,
        "max_tokens": 150,
        "api_path": "/home/sfang9/workshop/project_test/openai/openai-key",
        "attack_type": 'paraphrase',
        "input_type": None,
    }
    attack_framework = AttackFramework(
        model=model, attack_method="llm_attack", attack_config=attack_config, dataset="mbpp")
    save_path = "/home/sfang9/workshop/project_test/test-results"
    _, _ = attack_framework.run_attack(save_prompts=save_path, save_results=save_path)

