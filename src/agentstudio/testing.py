"""离线测试替身：ScriptedChatModel 按剧本返回预设消息，不联网、不需要 API Key。

它同时服务于两件事：
1. 单元测试里验证 Agent 的工具调用链路；
2. `python -m agentstudio.demo` 的零成本演示。
"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


class ScriptedChatModel(BaseChatModel):
    """按顺序吐出预设 AIMessage 的假模型。

    Example:
        >>> model = ScriptedChatModel(responses=[AIMessage(content="你好")])
        >>> model.invoke("hi").content
        '你好'
    """

    responses: list[AIMessage] = Field(default_factory=list)
    received: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-chat"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChatModel:  # noqa: ANN401
        """假装支持工具绑定，让 create_tool_calling_agent 能正常装配。"""
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.received.append(list(messages))
        index = min(len(self.received) - 1, len(self.responses) - 1)
        message = self.responses[index] if self.responses else AIMessage(content="")
        return ChatResult(generations=[ChatGeneration(message=message)])
