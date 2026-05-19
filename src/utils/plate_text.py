import re


def normalize_plate_text(text: str) -> str:
    """Keep only plate-like characters and normalize to uppercase Latin/digits."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())
