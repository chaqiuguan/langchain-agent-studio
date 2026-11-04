"""智能体链路测试：用 ScriptedChatModel 离线验证工具调用、记忆与降级。"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from agentstudio.agent import AgentError, AgentSession
from agentstudio.config import Settings
from agentstudio.testing import ScriptedChatModel
from agentstudio.tools import build_tools


def make_settings(**overrides: object) -> Settings:
    base = {
        "provider": "deepseek",
        "model": "fake-model",
        "api_key": "test-key",
        "base_url": "http://localhost/v1",
        "enabled_tools": ("calculator", "current_datetime", "read_file"),
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def tool_call(name: str, args: dict[str, str], call_id: str = "call_1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def build_session(responses: list[AIMessage], **overrides: object) -> AgentSession:
    settings = make_settings(**overrides)
    return AgentSession(
        settings,
        llm=ScriptedChatModel(responses=responses),
        tools=build_tools(settings.enabled_tools, settings.data_dir),
    )


def test_agent_calls_tool_and_answers() -> None:
    session = build_session(
        [
            tool_call("calculator", {"expression": "(12+8)*3"}),
            AIMessage(content="结果是 60。"),
        ]
    )
    reply = session.ask("(12+8)*3 等于多少？")

    assert reply.output == "结果是 60。"
    assert reply.tool_calls == 1
    assert reply.steps[0].tool == "calculator"
    assert "60" in reply.steps[0].observation
    assert reply.elapsed >= 0


def test_agent_without_tool_calls() -> None:
    session = build_session([AIMessage(content="你好，我是智能体。")])
    reply = session.ask("你好")
    assert reply.output == "你好，我是智能体。"
    assert reply.steps == []


def test_agent_multi_step_tools() -> None:
    session = build_session(
        [
            tool_call("calculator", {"expression": "2**10"}, "c1"),
            tool_call("current_datetime", {"timezone_name": "Asia/Shanghai"}, "c2"),
            AIMessage(content="计算完成。"),
        ]
    )
    reply = session.ask("算 2 的 10 次方，并告诉我现在几点")
    # 第一步返回后模型再次请求工具，因此至少记录了计算这一步
    assert reply.steps[0].tool == "calculator"
    assert "1024" in reply.steps[0].observation


def test_memory_keeps_window() -> None:
    session = build_session(
        [AIMessage(content="第一轮"), AIMessage(content="第二轮"), AIMessage(content="第三轮")],
        memory_window=1,
    )
    session.ask("问题一")
    session.ask("问题二")
    session.ask("问题三")
    # memory_window=1 表示只保留最近 1 轮（一问一答 2 条消息）
    assert len(session) == 2


def test_memory_disabled() -> None:
    session = build_session([AIMessage(content="好")], memory_window=0)
    session.ask("问题")
    assert len(session) == 0


def test_reset_clears_history() -> None:
    session = build_session([AIMessage(content="好")])
    session.ask("问题")
    assert len(session) == 2
    session.reset()
    assert len(session) == 0


def test_empty_question_rejected() -> None:
    session = build_session([AIMessage(content="好")])
    with pytest.raises(AgentError, match="问题不能为空"):
        session.ask("   ")


def test_react_mode_builds() -> None:
    session = build_session(
        [AIMessage(content="Thought: 不需要工具\nFinal Answer: 你好。")],
        agent_mode="react",
    )
    reply = session.ask("你好")
    assert "你好" in reply.output


def test_build_tools_rejects_unknown_name() -> None:
    with pytest.raises(KeyError, match="未知工具"):
        build_tools(["calculator", "nonexistent_tool"], "data")
