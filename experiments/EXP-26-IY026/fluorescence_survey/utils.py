"""Small helpers shared across the parsing modules."""

from typing import Iterable


def dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    """Strip, drop empties, and remove case-insensitive duplicates, keeping first-seen order."""
    seen = set()
    unique = []
    for value in values:
        cleaned = value.strip()
        if not cleaned or cleaned.lower() in seen:
            continue
        seen.add(cleaned.lower())
        unique.append(cleaned)
    return unique
