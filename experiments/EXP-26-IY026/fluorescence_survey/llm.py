"""Thin OpenAI wrapper used by the parsers and classifiers.

The LLM is always a *fallback*: every caller here is reached only after the
deterministic regex/database paths have failed to produce an answer.
"""

from typing import TYPE_CHECKING

from . import config

if TYPE_CHECKING:
    import openai


def make_client(api_key: str | None = None) -> "openai.OpenAI":
    """Create an OpenAI client with the project's timeout/retry policy."""
    # Imported here rather than at module level so the deterministic parsers,
    # which reach this module only through `complete_or_none(client=None, ...)`,
    # can be imported and tested without the OpenAI SDK installed.
    import openai

    return openai.OpenAI(
        api_key=api_key or config.OPENAI_API_KEY,
        timeout=config.OPENAI_TIMEOUT_SECONDS,
        max_retries=config.OPENAI_MAX_RETRIES,
    )


class Session:
    """An OpenAI client plus a transcript of every call made through it.

    A dataset can trigger four separate LLM calls (condition, TF identity,
    fluorescence, TF localisation) and previously only the fluorescence reply was
    kept, so a model that answered UNKNOWN or timed out on the other three left no
    trace to diagnose. Wrapping the client rather than threading a recorder
    through every parser means no parser signature has to change: they keep taking
    a "client" and this stands in for one.
    """

    def __init__(self, client: "Session | openai.OpenAI | None" = None):
        # Accepts a Session so that wrapping is idempotent: callers can hand on
        # whatever "client" they were given without checking what it is.
        self.client = client.client if isinstance(client, Session) else client
        self.calls: list[tuple[str, str]] = []

    def record(self, label: str, reply: str | None) -> None:
        self.calls.append((label, reply if reply is not None else "(call failed)"))

    def transcript(self) -> str:
        """Render the calls for the results.csv audit column."""
        return "\n\n".join(f"[{label}]\n{reply}" for label, reply in self.calls)


def _unwrap(client: "Session | openai.OpenAI | None") -> "openai.OpenAI | None":
    """Return the underlying OpenAI client, whether or not it is wrapped in a Session."""
    return client.client if isinstance(client, Session) else client


def complete(
    client: "Session | openai.OpenAI",
    system_prompt: str,
    user_message: str,
    max_tokens: int,
    model: str = config.MODEL,
    label: str = "",
) -> str:
    """Run one chat completion and return the stripped reply text.

    Raises on API failure — used where a failure should mark the dataset as ERROR.
    """
    session = client if isinstance(client, Session) else None
    client = _unwrap(client)
    response = client.chat.completions.create(
        model=model,
        max_completion_tokens=max_tokens,
        timeout=config.OPENAI_TIMEOUT_SECONDS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    reply = response.choices[0].message.content.strip()
    if session is not None:
        session.record(label or "llm", reply)
    return reply


def complete_or_none(
    client: "Session | openai.OpenAI | None",
    system_prompt: str,
    user_message: str,
    max_tokens: int,
    model: str = config.MODEL,
    label: str = "",
) -> str | None:
    """Like `complete`, but returns None instead of raising.

    Used by the optional enrichment steps (condition, TF identity, TF
    localisation) where a failed call should simply fall through to the next
    fallback rather than fail the whole dataset.
    """
    if _unwrap(client) is None:
        return None
    try:
        return complete(client, system_prompt, user_message, max_tokens, model=model, label=label)
    except Exception:
        if isinstance(client, Session):
            client.record(label or "llm", None)
        return None


def parse_labelled_lines(text: str, labels: tuple[str, ...]) -> dict[str, str]:
    """Split a structured LLM reply such as ``FLUORESCENCE: YES`` into a dict.

    Labels are given without the colon; missing labels map to an empty string.
    """
    values = {label: "" for label in labels}
    for line in text.splitlines():
        for label in labels:
            if line.startswith(f"{label}:"):
                values[label] = line.split(":", 1)[1].strip()
                break
    return values
