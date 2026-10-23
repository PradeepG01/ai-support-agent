import json

from app.llm import AnthropicChat, OpenAIChat, ToolCall
from app.retrieval import KnowledgeBase, chunk_markdown
from app.webhooks import sign, verify


def test_chunking_splits_on_headings():
    chunks = chunk_markdown("doc", "# Title\n\nintro\n\n## A\nfirst section\n\n## B\nsecond section")
    assert len(chunks) == 3
    assert all(c.title == "Title" for c in chunks)


def test_long_section_is_windowed_with_overlap():
    text = "## Long\n" + " ".join(f"w{i}" for i in range(300))
    chunks = chunk_markdown("doc", text, max_words=100, overlap=20)
    assert len(chunks) >= 3
    assert chunks[0].text.split()[-20:] == chunks[1].text.split()[:20]


def test_search_ranks_relevant_chunk_first():
    kb = KnowledgeBase()
    kb.add_document("a", "## Pool\nThe pool opens at 6 AM.")
    kb.add_document("b", "## Parking\nValet parking is free.")
    assert kb.search("when does the pool open")[0].chunk.doc_id == "a"


def test_reingest_replaces_document():
    kb = KnowledgeBase()
    kb.add_document("a", "## Pool\nold text about pool")
    kb.add_document("a", "## Pool\nnew text about pool")
    assert len(kb) == 1


def test_webhook_signature_and_replay_window():
    body = b'{"x":1}'
    sig = sign("s", "1000", body)
    assert verify("s", "1000", sig, body, now=1000)
    assert not verify("s", "1000", sig, b'{"x":2}', now=1000)  # tampered body
    assert not verify("s", "1000", sig, body, now=5000)  # replayed too late


def test_openai_wire_format_for_tool_round_trip():
    call = ToolCall("call_1", "lookup_booking", {"reference": "SG1234"})
    wire = OpenAIChat("k", "m", 5)._to_wire(
        [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "call_1", "name": "lookup_booking", "content": "{}"},
        ]
    )
    assert wire[1]["tool_calls"][0]["function"]["arguments"] == json.dumps({"reference": "SG1234"})
    assert wire[2] == {"role": "tool", "tool_call_id": "call_1", "content": "{}"}


def test_anthropic_wire_format_groups_tool_results():
    calls = [ToolCall("t1", "a", {}), ToolCall("t2", "b", {})]
    system, wire = AnthropicChat._to_wire(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": calls},
            {"role": "tool", "tool_call_id": "t1", "name": "a", "content": "1"},
            {"role": "tool", "tool_call_id": "t2", "name": "b", "content": "2"},
        ]
    )
    assert system == "sys"
    assert [b["type"] for b in wire[1]["content"]] == ["tool_use", "tool_use"]
    assert len(wire[2]["content"]) == 2 and wire[2]["role"] == "user"
