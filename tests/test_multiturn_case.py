from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from ai_eval.dataset import load_cases
from ai_eval.models import EvalCase, Message


def _case(messages: list[dict[str, str]], **kw: object) -> EvalCase:
    return EvalCase.model_validate({"id": "c1", "input": "And in euros?", "messages": messages, **kw})


def test_single_turn_case_unchanged():
    case = EvalCase(id="c1", input="hi")
    assert case.messages == []
    assert not case.is_multi_turn
    assert case.conversation() == [Message(role="user", content="hi")]
    assert "messages" in case.model_dump()  # defaults to an empty list


def test_multi_turn_case_builds_conversation():
    case = _case(
        [
            {"role": "system", "content": "You are a pricing bot."},
            {"role": "user", "content": "How much is the pro plan?"},
            {"role": "assistant", "content": "$20 per month."},
        ]
    )
    assert case.is_multi_turn
    conv = case.conversation()
    assert [m.role for m in conv] == ["system", "user", "assistant", "user"]
    assert conv[-1].content == "And in euros?"
    assert len(case.messages) == 3  # conversation() does not mutate the case


def test_system_only_history_is_allowed():
    case = _case([{"role": "system", "content": "Be brief."}])
    assert [m.role for m in case.conversation()] == ["system", "user"]


@pytest.mark.parametrize(
    ("messages", "match"),
    [
        ([{"role": "user", "content": "hi"}], "end with an assistant"),
        ([{"role": "assistant", "content": "hello"}], "expected role 'user'"),
        (
            [
                {"role": "user", "content": "a"},
                {"role": "user", "content": "b"},
                {"role": "assistant", "content": "c"},
            ],
            "alternate",
        ),
        (
            [
                {"role": "user", "content": "a"},
                {"role": "system", "content": "b"},
                {"role": "assistant", "content": "c"},
            ],
            "only allowed at the start",
        ),
        ([{"role": "tool", "content": "x"}], "role"),
        ([{"role": "user", "content": ""}], "content"),
        ([{"role": "user", "content": "   "}], "blank"),
        ([{"role": "user", "content": "a", "name": "bob"}], "name"),
    ],
)
def test_invalid_conversations_rejected(messages, match):
    with pytest.raises(ValidationError, match=match):
        _case(messages)


def test_multi_turn_case_loads_from_jsonl(tmp_path):
    path = tmp_path / "cases.jsonl"
    rows = [
        {"id": "single", "input": "hello"},
        {
            "id": "multi",
            "input": "What about tomorrow?",
            "messages": [
                {"role": "user", "content": "Weather today?"},
                {"role": "assistant", "content": "Sunny."},
            ],
        },
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    single, multi = load_cases(path)
    assert not single.is_multi_turn
    assert [m.role for m in multi.conversation()] == ["user", "assistant", "user"]


def test_invalid_conversation_in_jsonl_reports_line(tmp_path):
    path = tmp_path / "cases.jsonl"
    bad = {"id": "x", "input": "q", "messages": [{"role": "user", "content": "a"}]}
    path.write_text(json.dumps({"id": "ok", "input": "q"}) + "\n" + json.dumps(bad) + "\n")
    with pytest.raises(ValueError, match=r"cases.jsonl:2: invalid case"):
        load_cases(path)
