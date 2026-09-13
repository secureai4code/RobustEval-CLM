"""Utilities for protecting quoted content from text transformations.

Used by the CanItEdit attack pipeline to mask strings (backticks, single/double
quotes) before applying adversarial attacks, then restore them afterward.
"""

import re
from typing import Dict, Tuple


def mask_protected_content(text: str) -> Tuple[str, Dict[str, str]]:
    """Mask quoted content in text to protect it from transformations.

    Replaces backtick, single-quoted, and double-quoted strings with unique
    placeholders. Call restore() after transformations to put originals back.

    Args:
        text: Input text containing quoted strings to protect.

    Returns:
        Tuple of (masked_text, placeholders) where placeholders maps
        token -> original content.
    """
    placeholders: Dict[str, str] = {}
    counter = [0]

    def replace(match: re.Match) -> str:
        token = f"@@{counter[0]}@@"
        placeholders[token] = match.group(0)
        counter[0] += 1
        return token

    # Process in order: backticks, single quotes, double quotes
    text = re.sub(r"`[^`]+`", replace, text)
    text = re.sub(r"'[^']+'", replace, text)
    text = re.sub(r'"[^"]+"', replace, text)

    return text, placeholders


def restore(text: str, placeholders: Dict[str, str]) -> str:
    """Restore masked content using the placeholders from mask_protected_content.

    Args:
        text: Text with placeholder tokens (e.g. @@0@@).
        placeholders: Mapping from token to original content.

    Returns:
        Text with all placeholders replaced by their original content.
    """
    # Fix placeholders corrupted by tokenizers that split "@@N@@" into "@ @ N @ @"
    text = re.sub(r'@\s+@\s+(\d+)\s+@\s+@', r'@@\1@@', text)
    for token, original in placeholders.items():
        text = text.replace(token, original)
    return text
