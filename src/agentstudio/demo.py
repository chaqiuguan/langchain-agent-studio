"""零成本演示：不需要 API Key、不联网，直接观察 ReAct 工具调用链路。

用法：
    python -m agentstudio.demo
"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from rich.console import Console
from rich.panel import Panel

from .agent import AgentSession
from .config import Settings
from .testing import ScriptedChatModel
from .tools import build_tools

DEMO_QUESTION = "帮我算一下 (12+8)*3 的结果，顺便告诉我今天几号、星期几。"

DEMO_ANSWER = """两件事都办好了：

1. **计算结果**：`(12+8) = 20`，`20 × 3 = 60`。
2. **今天的日期**：见上一步 current_datetime 的返回值，它同时给出了星期与 UTC 偏移。

说明：本次演示使用的是脚本化假模型 ScriptedChatModel，
它不会真的联网推理，目的是在零成本的情况下展示
「模型发起工具调用 → 工具返回观察结果 → 模型汇总回答」的完整链路。
接入真实大模型后，这段回答由模型自己生成。"""


def build_demo_session() -> AgentSession:
    """组装一个只会按剧本调两次工具的会话。"""
    script = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "calculator",
                    "args": {"expression": "(12+8)*3"},
                    "id": "call_calculator",
                    "type": "tool_call",
                },
                {
                    "name": "current_datetime",
                    "args": {"timezone_name": "Asia/Shanghai"},
                    "id": "call_datetime",
                    "type": "tool_call",
                },
            ],
        ),
        AIMessage(content=DEMO_ANSWER),
    ]
    settings = Settings(
        provider="deepseek",
        model="scripted-demo",
        api_key="offline-demo",
        base_url="http://localhost/v1",
        enabled_tools=("calculator", "current_datetime"),
    )
    return AgentSession(
        settings,
        llm=ScriptedChatModel(responses=script),
        tools=build_tools(settings.enabled_tools, settings.data_dir),
    )


def main() -> int:
    console = Console()
    console.rule("[bold cyan]AgentStudio 离线演示 · 不需要 API Key")
    console.print(f"[bold]用户：[/bold]{DEMO_QUESTION}\n")

    session = build_demo_session()
    with console.status("智能体执行中…", spinner="dots"):
        reply = session.ask(DEMO_QUESTION)

    for index, step in enumerate(reply.steps, start=1):
        console.print(
            Panel(
                f"[bold]{step.tool}[/bold]\n输入：{step.tool_input}\n输出：{step.observation}",
                title=f"第 {index} 步 · 工具调用",
                border_style="yellow",
            )
        )

    console.print(Panel(reply.output, title="最终回答", border_style="green"))
    console.print(f"[dim]耗时 {reply.elapsed:.2f}s ｜ 工具调用 {reply.tool_calls} 次[/dim]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
