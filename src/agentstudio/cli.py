"""命令行界面：适合快速验证与脚本化调用。

    agentstudio                       # 进入交互式对话
    agentstudio -q "今天北京天气如何"   # 只问一个问题
    agentstudio --list-tools          # 查看可用工具
"""

from __future__ import annotations

import argparse
import os
import sys

from rich.console import Console
from rich.panel import Panel

from .agent import AgentError, AgentReply, AgentSession
from .config import PROVIDERS, ConfigError, Settings
from .tools import AVAILABLE_TOOLS

BANNER = """[bold cyan]AgentStudio 智能体工坊[/bold cyan] [dim]v0.1.0[/dim]
输入问题开始对话；输入 [bold]exit[/bold] 或按 Ctrl+C 退出。"""


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="agentstudio",
        description="基于 LangChain 的工具调用智能体（命令行版）",
    )
    parser.add_argument("-q", "--question", help="只问一个问题就退出")
    parser.add_argument("-p", "--provider", choices=sorted(PROVIDERS), help="切换模型服务商")
    parser.add_argument("-m", "--model", help="指定模型名称")
    parser.add_argument("--tools", help="启用的工具，逗号分隔")
    parser.add_argument("-v", "--verbose", action="store_true", help="打印 ReAct 全过程")
    parser.add_argument("--list-tools", action="store_true", help="列出可用工具后退出")
    return parser.parse_args(argv)


def _apply_overrides(args: argparse.Namespace, settings: Settings) -> Settings:
    if args.provider and args.provider != settings.provider:
        profile = PROVIDERS[args.provider]
        settings = settings.with_overrides(
            provider=profile.name,
            model=profile.default_model,
            base_url=profile.base_url,
            api_key=os.getenv(profile.key_env, "") or os.getenv("LLM_API_KEY", ""),
        )
    if args.model:
        settings = settings.with_overrides(model=args.model)
    if args.tools:
        settings = settings.with_overrides(
            enabled_tools=tuple(item.strip() for item in args.tools.split(",") if item.strip())
        )
    if args.verbose:
        settings = settings.with_overrides(verbose=True)
    return settings


def _short(text: str, limit: int = 600) -> str:
    """终端里只展开工具输出的开头，避免一次回答刷满屏幕。"""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n…（已省略 {len(text) - limit} 个字符）"


def _print_reply(console: Console, reply: AgentReply) -> None:
    for index, step in enumerate(reply.steps, start=1):
        console.print(
            Panel(
                f"[bold]{step.tool}[/bold]\n输入：{step.tool_input}\n输出：{_short(step.observation)}",
                title=f"第 {index} 步 · 工具调用",
                border_style="yellow",
            )
        )
    console.print(Panel(reply.output or "（模型没有返回内容）", title="回答", border_style="green"))
    console.print(f"[dim]耗时 {reply.elapsed:.2f}s ｜ 工具调用 {reply.tool_calls} 次[/dim]")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    console = Console()

    if args.list_tools:
        console.print("[bold]可用工具：[/bold]")
        for name in AVAILABLE_TOOLS:
            console.print(f"  · {name}")
        return 0

    try:
        settings = _apply_overrides(args, Settings.from_env()).validated()
        session = AgentSession(settings)
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1

    console.print(BANNER)
    console.print(f"[dim]{settings.describe()} ｜ 模式 {settings.agent_mode} ｜ 工具 {', '.join(settings.enabled_tools)}[/dim]\n")

    if args.question:
        try:
            reply = session.ask(args.question)
        except (AgentError, ConfigError) as exc:
            console.print(f"[red]{exc}[/red]")
            return 1
        _print_reply(console, reply)
        return 0

    while True:
        try:
            question = console.input("[bold cyan]你 > [/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]已退出。[/dim]")
            return 0
        if not question:
            continue
        if question.lower() in {"exit", "quit", ":q", "退出"}:
            console.print("[dim]已退出。[/dim]")
            return 0
        try:
            with console.status("[dim]智能体思考中…[/dim]", spinner="dots"):
                reply = session.ask(question)
        except (AgentError, ConfigError) as exc:
            console.print(f"[red]{exc}[/red]")
            continue
        console.print()
        _print_reply(console, reply)
        console.print()


if __name__ == "__main__":
    sys.exit(main())
