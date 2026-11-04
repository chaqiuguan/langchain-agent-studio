"""工具注册表：按名字组装工具列表，新增工具只需在这里登记一行。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from langchain_core.tools import BaseTool

from .calculator import calculator
from .datetime_tool import make_datetime_tool
from .file_reader import make_file_reader_tool
from .web_search import make_web_search_tool

_TOOL_FACTORIES: dict[str, Callable[[Path], BaseTool]] = {
    "calculator": lambda _root: calculator,
    "current_datetime": lambda _root: make_datetime_tool(),
    "web_search": lambda _root: make_web_search_tool(),
    "read_file": lambda root: make_file_reader_tool(root),
}

AVAILABLE_TOOLS: tuple[str, ...] = tuple(_TOOL_FACTORIES)


def build_tools(names: Sequence[str], data_dir: Path | str = "data") -> list[BaseTool]:
    """按名字构建工具列表；遇到未知名字直接报错，避免静默少装工具。"""
    unknown = [name for name in names if name not in _TOOL_FACTORIES]
    if unknown:
        raise KeyError(f"未知工具：{', '.join(unknown)}；可选：{', '.join(AVAILABLE_TOOLS)}")
    root = Path(data_dir)
    return [_TOOL_FACTORIES[name](root) for name in names]


__all__ = ["AVAILABLE_TOOLS", "build_tools", "calculator"]
