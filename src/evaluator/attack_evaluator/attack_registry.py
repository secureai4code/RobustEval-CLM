"""Registry for mapping attack method names to their implementations."""

from typing import Dict, List, Type

from src.evaluator.attack_evaluator.attacks.base_attack import BaseAttack


class AttackRegistry:
    """Registry for attack classes to reduce coupling between framework and implementations."""

    _attacks: Dict[str, Type[BaseAttack]] = {}

    @classmethod
    def register(cls, name: str, attack_class: Type[BaseAttack]) -> None:
        """Register an attack class under the given name."""
        cls._attacks[name] = attack_class

    @classmethod
    def get(cls, name: str) -> Type[BaseAttack]:
        """Return the attack class registered under *name*.

        Raises:
            ValueError: If no attack is registered for *name*.
        """
        if name not in cls._attacks:
            raise ValueError(
                f"Unknown attack method: '{name}'. "
                f"Available: {cls.list_attacks()}"
            )
        return cls._attacks[name]

    @classmethod
    def list_attacks(cls) -> List[str]:
        """Return a sorted list of all registered attack names."""
        return sorted(cls._attacks.keys())

    @classmethod
    def auto_discover(cls) -> None:
        """Import all attack modules to trigger registration."""
        try:
            from src.evaluator.attack_evaluator.attacks import (  # noqa: F401
                CharacterCaseAttack,
                ChatGPTAttack,
                DestructureAttack,
                LLMParaphraseAttack,
                NaturalNoiseAttack,
                NoiseAttack,
                SemanticAttack,
                StructuralAttack,
                StructuredFormatAttack,
                SynonymAttack,
                TranslationAttack,
            )

            cls.register("synonym", SynonymAttack)
            cls.register("char", CharacterCaseAttack)
            cls.register("destructure", DestructureAttack)
            cls.register("translation", TranslationAttack)
            cls.register("translate", TranslationAttack)
            cls.register("chatgpt", ChatGPTAttack)
            cls.register("llm_attack", ChatGPTAttack)
            cls.register("llm_paraphrase", LLMParaphraseAttack)
            cls.register("noise", NoiseAttack)
            cls.register("natural_noise", NaturalNoiseAttack)
            cls.register("semantic", SemanticAttack)
            cls.register("structural", StructuralAttack)
            cls.register("structured", StructuredFormatAttack)

        except ImportError as exc:
            print(f"Warning: Could not import some attack classes: {exc}")


# Populate registry on module load.
AttackRegistry.auto_discover()
