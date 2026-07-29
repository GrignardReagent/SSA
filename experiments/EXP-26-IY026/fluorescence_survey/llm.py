"""Thin OpenAI wrapper used by the parsers and classifiers.

The LLM is always a *fallback*: every caller here is reached only after the
deterministic regex/database paths have failed to produce an answer.
"""

import openai

from . import config


def make_client(api_key: str | None = None) -> openai.OpenAI:
    """Create an OpenAI client with the project's timeout/retry policy."""
    return openai.OpenAI(
        api_key=api_key or config.OPENAI_API_KEY,
        timeout=config.OPENAI_TIMEOUT_SECONDS,
        max_retries=config.OPENAI_MAX_RETRIES,
    )


def complete(
    client: openai.OpenAI,
    system_prompt: str,
    user_message: str,
    max_tokens: int,
    model: str = config.MODEL,
) -> str:
    """Run one chat completion and return the stripped reply text.

    Raises on API failure — used where a failure should mark the dataset as ERROR.
    """
    response = client.chat.completions.create(
        model=model,
        max_completion_tokens=max_tokens,
        timeout=config.OPENAI_TIMEOUT_SECONDS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content.strip()


def complete_or_none(
    client: "openai.OpenAI | None",
    system_prompt: str,
    user_message: str,
    max_tokens: int,
    model: str = config.MODEL,
) -> str | None:
    """Like `complete`, but returns None instead of raising.

    Used by the optional enrichment steps (condition, TF identity, TF
    localisation) where a failed call should simply fall through to the next
    fallback rather than fail the whole dataset.
    """
    if client is None:
        return None
    try:
        return complete(client, system_prompt, user_message, max_tokens, model=model)
    except Exception:
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
