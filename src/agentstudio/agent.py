"""智能体装配层：把「模型 + 工具 + 提示词 + 记忆」组装成一个可复用的会话对象。

对外只暴露两个东西：
- `AgentSession`：持有对话历史，`ask()` 一次即完成"思考 → 调工具 → 回答"；
- `AgentReply`：结构化结果，包含回答、工具调用轨迹与耗时，方便界面渲染。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from langchain.agents import AgentExecutor, create_react_agent, create_tool_calling_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.tools import BaseTool

from .config import Settings
from .llm import build_chat_model
from .tools import build_tools

TOOL_CALLING_TEMPLATE = [
    ("system", "{system_prompt}"),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
]

REACT_TEMPLATE = """{system_prompt}

你可以在需要时使用以下工具：

{tools}

请严格按照以下格式作答：

Question: 需要回答的问题
Thought: 你的思考过程
Action: 工具名，必须是 [{tool_names}] 中的一个
Action Input: 传给该工具的输入
Observation: 工具返回的结果
...（Thought / Action / Action Input / Observation 可以重复多次）
Thought: 我已经知道最终答案
Final Answer: 用中文给出最终回答

不需要工具时，可以直接输出 Thought 与 Final Answer。

Question: {input}
Thought:{agent_scratchpad}"""


class AgentError(RuntimeError):
    """智能体执行过程中的可展示错误。"""


@dataclass
class ToolStep:
    """一次工具调用及其观察结果。"""

    tool: str
    tool_input: Any
    observation: str

    def summary(self, limit: int = 160) -> str:
        text = " ".join(self.observation.split())
        return text[:limit] + ("…" if len(text) > limit else "")


@dataclass
class AgentReply:
    """一次问答的完整结果。"""

    output: str
    steps: list[ToolStep] = field(default_factory=list)
    elapsed: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def tool_calls(self) -> int:
        return len(self.steps)


def _format_history(history: list[BaseMessage]) -> str:
    """react 模式没有 MessagesPlaceholder，把历史对话压成一段文本。"""
    if not history:
        return ""
    lines = []
    for message in history:
        role = "用户" if isinstance(message, HumanMessage) else "助手"
        content = message.content if isinstance(message.content, str) else str(message.content)
        lines.append(f"{role}：{content}")
    return "之前的对话（供参考）：\n" + "\n".join(lines) + "\n\n"


def _to_steps(intermediate_steps: Any) -> list[ToolStep]:
    steps: list[ToolStep] = []
    for action, observation in intermediate_steps or []:
        steps.append(
            ToolStep(
                tool=str(getattr(action, "tool", "unknown")),
                tool_input=getattr(action, "tool_input", ""),
                observation=str(observation),
            )
        )
    return steps


class AgentSession:
    """一次会话：模型、工具、历史消息与执行器都在这里。"""

    def __init__(
        self,
        settings: Settings,
        llm: BaseChatModel | None = None,
        tools: list[BaseTool] | None = None,
    ) -> None:
        # 只有真正要联网时才校验 Key，方便用假模型做离线测试
        self.settings = settings if llm is not None else settings.validated()
        self.llm: BaseChatModel = llm if llm is not None else build_chat_model(self.settings)
        self.tools: list[BaseTool] = (
            tools if tools is not None else build_tools(self.settings.enabled_tools, self.settings.data_dir)
        )
        self.history: list[BaseMessage] = []
        self.executor = self._build_executor()

    # ---------- 内部 ----------

    def _build_executor(self) -> AgentExecutor:
        settings = self.settings
        if settings.agent_mode == "react":
            prompt = PromptTemplate.from_template(REACT_TEMPLATE).partial(
                system_prompt=settings.system_prompt
            )
            agent = create_react_agent(self.llm, self.tools, prompt)
        else:
            prompt = ChatPromptTemplate.from_messages(TOOL_CALLING_TEMPLATE).partial(
                system_prompt=settings.system_prompt
            )
            agent = create_tool_calling_agent(self.llm, self.tools, prompt)

        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=settings.verbose,
            max_iterations=settings.max_iterations,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )

    def _remember(self, question: str, answer: str) -> None:
        window = max(0, self.settings.memory_window)
        if window == 0:
            return
        self.history.append(HumanMessage(content=question))
        self.history.append(AIMessage(content=answer))
        limit = window * 2
        if len(self.history) > limit:
            self.history = self.history[-limit:]

    # ---------- 公开接口 ----------

    def ask(self, question: str, remember: bool = True) -> AgentReply:
        """向智能体提一个问题，返回结构化结果。"""
        text = (question or "").strip()
        if not text:
            raise AgentError("问题不能为空。")

        payload: dict[str, Any] = {"input": text}
        if self.settings.agent_mode == "react":
            payload["input"] = _format_history(self.history) + text
        else:
            payload["chat_history"] = list(self.history)

        started = time.perf_counter()
        try:
            result = self.executor.invoke(payload)
        except Exception as exc:  # noqa: BLE001 - 统一包装成可展示错误
            raise AgentError(f"智能体执行失败：{type(exc).__name__}: {exc}") from exc
        elapsed = time.perf_counter() - started

        reply = AgentReply(
            output=str(result.get("output", "")).strip(),
            steps=_to_steps(result.get("intermediate_steps")),
            elapsed=elapsed,
            raw=result,
        )
        if remember:
            self._remember(text, reply.output)
        return reply

    def reset(self) -> None:
        """清空对话历史。"""
        self.history.clear()

    def __len__(self) -> int:
        return len(self.history)
