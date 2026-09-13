"""De-structuring adversarial attack (the inverse of StructuredFormatAttack).

Strips the structural markers that delineate sub-parts of a prompt so everything
blends into running text, testing whether a model relies on those cues:

- **MBPP** (``input_type="prompt"``): drop the ``assert`` keyword and merge the test
  expression onto the instruction line.
- **HumanEval** (``input_type="code"``): drop ``>>>`` / ``...`` doctest markers, collapse
  each example (call + expected output) onto one line, and bleed the first example into
  the end of the prose.
- **CanItEdit** (``input_type="instruction"``): the instruction is already unstructured,
  so the structure lives in the *wrapping* edit prompt. ``wrap_edit_prompt`` rebuilds that
  wrapper bare -- no markdown headers, code fences, or section labels, just the system
  prompt, the code, and the instruction. The framework calls this hook for the adversarial
  prompt only (the original keeps the structured wrapper as the baseline).

The transform is deterministic (no randomness/seed).
"""

import re
from typing import Any, Dict, List, Optional

from .base_attack import BaseAttack

# Matches a triple-quoted string (single or double quotes), non-greedy.
_DOCSTRING_RE = re.compile(r"'''[\s\S]*?'''|\"\"\"[\s\S]*?\"\"\"")

_VALID_INPUT_TYPES = ("prompt", "code", "instruction")


class DestructureAttack(BaseAttack):
    """Removes structural markers so prompt sub-parts blend into running text."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.input_type: str = self.config["input_type"]

    def validate_config(self) -> None:
        """Validate configuration. Only ``input_type`` is required."""
        if "input_type" not in self.config:
            raise ValueError("DestructureAttack: config missing key 'input_type'")
        if self.config["input_type"] not in _VALID_INPUT_TYPES:
            raise ValueError(
                f"DestructureAttack: input_type must be one of "
                f"{sorted(_VALID_INPUT_TYPES)}, got {self.config['input_type']!r}"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_adversarial_example(
        self,
        input_code: str,
        target_label: Optional[Any] = None,
    ) -> str:
        """De-structure *input_code* based on ``input_type``."""
        if self.input_type == "prompt":
            return self._destructure_mbpp(input_code)
        if self.input_type == "code":
            return self._destructure_humaneval(input_code)
        if self.input_type == "instruction":
            # The instruction itself is already unstructured; the wrapper carries the
            # structure and is rebuilt via wrap_edit_prompt() at generation time.
            return input_code
        raise ValueError(f"Unknown input type: {self.input_type}")

    def wrap_edit_prompt(self, before: str, instruction: str) -> str:
        """Build a bare, de-structured CanItEdit edit prompt.

        Mirrors ``AttackFramework._build_edit_prompt`` but strips all structure: no
        ``## Code Before`` / ``## Instruction`` / ``## Code After`` headers, no
        ```` ```py ```` code fences, and no section labels. Only the system prompt, the
        original code, and the instruction remain, separated by blank lines; the model
        generates from there.
        """
        return (
            "You are PythonEditGPT. You will be provided the original code snippet "
            "and an instruction that specifies the changes you need to make. You will "
            "produce the changed code, based on the original code and the instruction "
            "given. Only produce the code, do not include any additional prose.\n\n"
            + before + "\n\n"
            + instruction + "\n\n"
        )

    # ------------------------------------------------------------------
    # Per-dataset formatters
    # ------------------------------------------------------------------

    def _destructure_mbpp(self, text: str) -> str:
        """Drop ``assert`` and merge the test expression onto the instruction line.

        MBPP prompts are 4 lines: ``\"\"\"`` / instruction / ``assert ...`` / ``\"\"\"``.
        """
        lines = text.splitlines()
        if len(lines) < 4:
            return text  # unknown format; leave untouched

        instruction = lines[1].strip()
        tests = [
            re.sub(r"^\s*assert\s+", "", line).strip()
            for line in lines[2:-1]
            if line.strip()
        ]
        merged = " ".join([instruction, *tests]).strip()
        return '"""\n' + merged + '\n"""'

    def _destructure_humaneval(self, code: str) -> str:
        """Drop ``>>>``/``...`` markers and blend doctests into the docstring prose.

        Only the *last* docstring (the task instruction) is rewritten; helper code,
        signatures, and imports are preserved by slicing around the regex match.
        """
        matches = list(_DOCSTRING_RE.finditer(code))
        if not matches:
            return code

        match = matches[-1]
        docstring = match.group(0)
        quote = docstring[:3]
        content = docstring[3:-3]

        lines = content.splitlines()
        example_start = next((i for i, line in enumerate(lines) if ">>>" in line), None)
        if example_start is None:
            return code  # no doctest markers to remove

        prose_lines = lines[:example_start]
        examples = self._collapse_doctests(lines[example_start:])
        if not examples:
            return code

        indent = self._leading_indent(code, match.start())
        prose_text = "\n".join(prose_lines).rstrip()

        if prose_text.strip():
            body = prose_text + " " + examples[0]
        else:
            body = indent + examples[0]
        for example in examples[1:]:
            body += "\n" + indent + example

        new_docstring = f"{quote}{body}\n{indent}{quote}"
        return code[: match.start()] + new_docstring + code[match.end() :]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collapse_doctests(example_lines: List[str]) -> List[str]:
        """Collapse doctest lines into one flowing line per example (markers removed).

        Each ``>>>`` starts a new example; ``...`` continues the call; any other
        non-blank line is treated as expected output. Blank lines (and the closing-quote
        indent line) are dropped.
        """
        examples: List[str] = []
        current: List[str] = []

        for line in example_lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(">>>"):
                if current:
                    examples.append(" ".join(current))
                    current = []
                current.append(stripped[3:].strip())
            elif stripped.startswith("..."):
                current.append(stripped[3:].strip())
            else:
                current.append(stripped)

        if current:
            examples.append(" ".join(current))
        return [example for example in examples if example]

    @staticmethod
    def _leading_indent(code: str, pos: int) -> str:
        """Return the leading whitespace of the line in *code* that contains *pos*."""
        line_start = code.rfind("\n", 0, pos) + 1
        prefix = code[line_start:pos]
        return prefix[: len(prefix) - len(prefix.lstrip())]


if __name__ == "__main__":
    mbpp_prompt = (
        '"""\n'
        "Write a function to find the shared elements from the given two lists.\n"
        "assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))\n"
        '"""\n'
    )
    humaneval_code = (
        "from typing import List\n\n\n"
        "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
        '    """ Check if in given list of numbers, are any two numbers closer to each other than\n'
        "    given threshold.\n"
        "    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n"
        "    False\n"
        "    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n"
        "    True\n"
        '    """\n'
    )
    before_code = "def add(a, b):\n    return a - b\n"
    instruction = "Fix the `add` function so it returns the sum of `a` and `b`."

    attack = DestructureAttack({"input_type": "prompt"})
    print("===== MBPP =====")
    print(attack.generate_adversarial_example(mbpp_prompt))

    attack = DestructureAttack({"input_type": "code"})
    print("\n===== HumanEval =====")
    print(attack.generate_adversarial_example(humaneval_code))

    attack = DestructureAttack({"input_type": "instruction"})
    print("\n===== CanItEdit (de-structured wrapper) =====")
    print(attack.wrap_edit_prompt(before_code, instruction))
