from __future__ import annotations

from langchain_core.messages import HumanMessage

from agent_kit.config import ToolResult
from agent_kit.state import append_messages, append_tool_outputs, make_initial_state, merge_dict
from agent_kit.utils.state_utils import apply_delta, state_delta


def test_state_reducers_merge_deterministically() -> None:
    left = make_initial_state(messages=[HumanMessage(content="a")], metadata={"x": 1})
    right_messages = [HumanMessage(content="b")]
    right_outputs = [ToolResult(tool_name="t", call_id="1", status="success", data={"ok": True})]

    merged_messages = append_messages(left["messages"], right_messages)
    merged_outputs = append_tool_outputs(left["tool_outputs"], right_outputs)
    merged_metadata = merge_dict(left["metadata"], {"x": 2, "y": 3})

    assert [m.content for m in merged_messages] == ["a", "b"]
    assert len(merged_outputs) == 1
    assert merged_metadata == {"x": 2, "y": 3}


def test_state_delta_and_apply_round_trip() -> None:
    before = make_initial_state(messages=[HumanMessage(content="hi")], metadata={"a": 1})
    after = make_initial_state(messages=[HumanMessage(content="hi"), HumanMessage(content="there")])
    after["metadata"] = {"a": 2, "b": 3}

    delta = state_delta(before, after)
    merged = apply_delta(before, delta)

    assert [m.content for m in merged["messages"]] == ["hi", "there"]
    assert merged["metadata"] == {"a": 2, "b": 3}
