"""The LLM call transcript that backs the `raw_llm_response` audit column."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from fluorescence_survey import llm  # noqa: E402


class FakeClient:
    """Minimal stand-in for openai.OpenAI: replies with whatever it was primed with."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.chat = self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply

        class Message:
            content = reply

        class Choice:
            message = Message()

        class Response:
            choices = [Choice()]

        return Response()


def test_every_call_is_recorded_under_its_label():
    session = llm.Session(FakeClient("2% glucose to 0%", "Msn2, Dot6"))
    llm.complete_or_none(session, "sys", "user", 40, label="condition")
    llm.complete_or_none(session, "sys", "user", 60, label="tf-identity")
    assert session.calls == [("condition", "2% glucose to 0%"), ("tf-identity", "Msn2, Dot6")]
    assert session.transcript() == "[condition]\n2% glucose to 0%\n\n[tf-identity]\nMsn2, Dot6"


def test_a_failed_call_still_leaves_a_trace():
    """The whole point of the column: a model that errored must not vanish silently."""
    session = llm.Session(FakeClient(RuntimeError("timeout")))
    assert llm.complete_or_none(session, "sys", "user", 40, label="condition") is None
    assert session.calls == [("condition", "(call failed)")]


def test_session_without_a_client_makes_no_calls():
    session = llm.Session(None)
    assert llm.complete_or_none(session, "sys", "user", 40, label="condition") is None
    assert session.calls == []


def test_wrapping_a_session_is_idempotent():
    """`classify` wraps whatever it is handed without inspecting it."""
    client = FakeClient()
    assert llm.Session(llm.Session(client)).client is client
