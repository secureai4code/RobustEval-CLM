"""General utility functions."""

from .content_protection import mask_protected_content, restore
from .function_extractor import extract_code_from_markdown, extract_functions

__all__ = [
    "extract_code_from_markdown",
    "extract_functions",
    "mask_protected_content",
    "restore",
]
