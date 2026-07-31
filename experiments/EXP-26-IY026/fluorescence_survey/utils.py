"""Small helpers shared across the parsing modules."""

from typing import Generic, Iterable, NamedTuple, TypeVar


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


T = TypeVar("T")


class Parsed(NamedTuple, Generic[T]):
    """A parsed value together with the extractor(s) that produced it.

    Every field in results.csv comes from a chain of fallbacks, and knowing which
    link fired is what tells you how far to trust the value — a TF read off a
    Batgirl group label is not the same evidence as one the LLM inferred from
    prose. `sources` therefore ends up in the `provenance` column.

    First-hit chains name a single source; merged chains name every source that
    contributed, in the order they were consulted.
    """

    value: T
    sources: tuple[str, ...] = ()

    def label(self) -> str:
        """Render the sources for the provenance column, e.g. ``group-labels,details``."""
        return ",".join(self.sources) if self.sources else "none"
