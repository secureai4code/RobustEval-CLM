"""Structured-format adversarial attack.

Leaves the wording of an instruction untouched but reorganizes it into a labeled,
sectioned layout::

    [Task]
    <sentence 1>

    [Details]                # emitted only when the instruction has >1 sentence
    - <sentence 2>
    - <sentence 3>

    [<Block label>]          # for any preserved test/example/code block
    <block, verbatim>

The instruction is split into sentences: the first becomes the ``[Task]`` line and
the rest become ``[Details]`` bullets. Any structural block that is part of the
prompt (MBPP assert tests, HumanEval ``>>>`` doctests) is preserved verbatim under
its own labeled section. The transform is deterministic (no randomness/seed).
"""

import re
import textwrap
from typing import Any, Dict, List, Optional, Tuple

import nltk
from nltk.tokenize import sent_tokenize

from .base_attack import BaseAttack

# NLTK sentence-tokenizer models, downloaded on first import (idempotent: nltk.download
# short-circuits if the resource is already present locally). Mirrors synonym_attack.py.
_NLTK_RESOURCES = ("punkt", "punkt_tab")
for _resource in _NLTK_RESOURCES:
    try:
        nltk.download(_resource, quiet=True)
    except Exception:  # network off / offline cache present; leave it to runtime
        pass

# Matches a triple-quoted string (single or double quotes), non-greedy.
_DOCSTRING_RE = re.compile(r"'''[\s\S]*?'''|\"\"\"[\s\S]*?\"\"\"")

_VALID_INPUT_TYPES = ("prompt", "code", "instruction")


class StructuredFormatAttack(BaseAttack):
    """Reformats the instructive text of a prompt into labeled sections."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.input_type: str = self.config["input_type"]

    def validate_config(self) -> None:
        """Validate configuration. Only ``input_type`` is required."""
        if "input_type" not in self.config:
            raise ValueError("StructuredFormatAttack: config missing key 'input_type'")
        if self.config["input_type"] not in _VALID_INPUT_TYPES:
            raise ValueError(
                f"StructuredFormatAttack: input_type must be one of "
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
        """Reformat *input_code* into the sectioned layout based on ``input_type``."""
        if self.input_type == "prompt":
            return self._format_mbpp(input_code)
        if self.input_type == "code":
            return self._format_humaneval(input_code)
        if self.input_type == "instruction":
            return self._format_instruction(input_code)
        raise ValueError(f"Unknown input type: {self.input_type}")

    # ------------------------------------------------------------------
    # Section builder
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_list_marker(sentence: str) -> str:
        """Strip a leading list marker (``-``, ``*``, ``•``) so we don't double-bullet."""
        return re.sub(r"^[-*•]\s+", "", sentence)

    @staticmethod
    def _pluralize(base: str, count: int) -> str:
        """Return *base* for a single item, otherwise its plural (e.g. Detail/Details)."""
        return base if count == 1 else base + "s"

    def _split_sentences(self, prose: str) -> List[str]:
        """Split *prose* into a list of non-empty, stripped sentences."""
        if not prose or not prose.strip():
            return []
        return [
            self._strip_list_marker(s.strip())
            for s in sent_tokenize(prose.strip())
            if s.strip()
        ]

    def _format_sections(self, prose: str, blocks: List[Tuple[str, str]]) -> str:
        """Build the sectioned layout from *prose* and preserved *blocks*.

        Args:
            prose: Natural-language instruction to split into Task/Details.
            blocks: Ordered ``(label, text)`` pairs preserved verbatim under
                ``[label]`` sections.

        Returns:
            The reformatted text. ``[Details]`` is omitted when the instruction has
            at most one sentence.
        """
        sentences = self._split_sentences(prose)
        lines: List[str] = []

        if sentences:
            lines.append("[Task]")
            lines.append(sentences[0])
            if len(sentences) > 1:
                details = sentences[1:]
                lines.append("")
                lines.append(f"[{self._pluralize('Detail', len(details))}]")
                lines.extend(f"- {sentence}" for sentence in details)

        for label, text in blocks:
            if not text.strip():
                continue
            if lines:
                lines.append("")
            lines.append(f"[{label}]")
            lines.append(text)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Per-dataset formatters
    # ------------------------------------------------------------------

    def _format_mbpp(self, text: str) -> str:
        """Reformat an MBPP prompt.

        MBPP prompts are 4 lines: ``\"\"\"`` / instruction / ``assert ...`` / ``\"\"\"``.
        The instruction becomes Task/Details; the assert(s) are preserved under
        ``[Examples]`` (matching HumanEval); the whole thing is re-wrapped in a docstring.
        """
        lines = text.splitlines()
        if len(lines) < 4:
            return text  # unknown format; leave untouched

        instruction = lines[1]
        example_block = "\n".join(lines[2:-1]).strip("\n")
        example_count = sum(1 for line in example_block.splitlines() if line.strip())
        label = self._pluralize("Example", example_count)
        body = self._format_sections(instruction, [(label, example_block)])
        return '"""\n' + body + '\n"""'

    def _format_humaneval(self, code: str) -> str:
        """Reformat a HumanEval prompt by rewriting only its task docstring.

        The task instruction is the *last* triple-quoted docstring (compound problems
        prepend a fully-implemented helper function whose docstring must stay verbatim).
        Everything outside that docstring (imports, signatures, helper code) is preserved
        by slicing around the regex match.
        """
        matches = list(_DOCSTRING_RE.finditer(code))
        if not matches:
            return code  # no docstring; leave untouched

        match = matches[-1]
        docstring = match.group(0)
        quote = docstring[:3]
        content = docstring[3:-3]

        prose, example_block = self._split_prose_and_examples(content)
        blocks = []
        if example_block.strip():
            label = self._pluralize("Example", example_block.count(">>>"))
            blocks = [(label, example_block)]
        body = self._format_sections(prose, blocks)

        indent = self._leading_indent(code, match.start())
        body_indented = self._indent_block(body, indent)
        new_docstring = f"{quote}\n{body_indented}\n{indent}{quote}"

        return code[: match.start()] + new_docstring + code[match.end() :]

    def _format_instruction(self, text: str) -> str:
        """Reformat a CanItEdit instruction (pure prose, possibly masked)."""
        return self._format_sections(text, [])

    # ------------------------------------------------------------------
    # HumanEval helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_prose_and_examples(content: str) -> Tuple[str, str]:
        """Split docstring *content* into prose and the trailing ``>>>`` example block.

        Everything from the first line containing ``>>>`` onward is the example block
        (dedented, verbatim); everything before it is prose collapsed to a single space.
        """
        lines = content.splitlines()
        example_start = next((i for i, line in enumerate(lines) if ">>>" in line), None)

        if example_start is None:
            prose = " ".join(line.strip() for line in lines)
            return re.sub(r"\s+", " ", prose).strip(), ""

        prose_lines = lines[:example_start]
        example_lines = lines[example_start:]
        prose = re.sub(r"\s+", " ", " ".join(line.strip() for line in prose_lines)).strip()
        example_block = textwrap.dedent("\n".join(example_lines)).strip("\n")
        return prose, example_block

    @staticmethod
    def _leading_indent(code: str, pos: str) -> str:
        """Return the leading whitespace of the line in *code* that contains *pos*."""
        line_start = code.rfind("\n", 0, pos) + 1
        prefix = code[line_start:pos]
        return prefix[: len(prefix) - len(prefix.lstrip())]

    @staticmethod
    def _indent_block(text: str, indent: str) -> str:
        """Prefix every non-blank line of *text* with *indent* (blank lines stay blank)."""
        return "\n".join(
            (indent + line) if line.strip() else line for line in text.split("\n")
        )


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
        "    given threshold. Return True if so, otherwise return False.\n"
        "    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n"
        "    False\n"
        "    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n"
        "    True\n"
        '    """\n'
    )
    humaneval_no_examples = (
        "\n\ndef encode_cyclic(s: str):\n"
        '    """\n'
        "    returns encoded string by cycling groups of three characters.\n"
        '    """\n'
    )
    instruction = (
        "Fix the calculate function to accept a list of operations. "
        "It should apply each operation in order. Return the final result."
    )

    for label, value in (
        ("MBPP (prompt)", mbpp_prompt),
        ("HumanEval (code, with examples)", humaneval_code),
        ("HumanEval (code, no examples)", humaneval_no_examples),
    ):
        input_type = "prompt" if "MBPP" in label else "code"
        attack = StructuredFormatAttack({"input_type": input_type})
        print(f"===== {label} =====")
        print(attack.generate_adversarial_example(value))
        print()

    attack = StructuredFormatAttack({"input_type": "instruction"})
    print("===== CanItEdit (instruction, multi-sentence) =====")
    print(attack.generate_adversarial_example(instruction))
