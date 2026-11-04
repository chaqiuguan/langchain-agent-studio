"""本地文件读取工具：带目录白名单、体积上限与字符截断。

安全边界
--------
智能体生成的参数来自大模型，属于不可信输入。因此这里做了三件事：
1. 只允许读取 data 目录内的文件（路径解析后做前缀校验，防止 ../ 与软链接逃逸）；
2. 只允许纯文本后缀，单文件不超过 200 KB；
3. 返回内容截断到 6000 字符，避免撑爆上下文窗口。
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import BaseTool, StructuredTool

MAX_BYTES = 200_000
MAX_CHARS = 6_000

TEXT_SUFFIXES = frozenset(
    {
        ".md",
        ".markdown",
        ".txt",
        ".json",
        ".jsonl",
        ".csv",
        ".tsv",
        ".py",
        ".js",
        ".ts",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".conf",
        ".log",
        ".html",
        ".css",
        ".xml",
        ".sql",
        ".java",
        ".c",
        ".h",
        ".cpp",
        ".go",
        ".rs",
        ".sh",
    }
)


def make_file_reader_tool(root: Path | str, max_chars: int = MAX_CHARS) -> BaseTool:
    base = Path(root).expanduser().resolve()

    def _resolve(raw: str) -> Path:
        """把用户给的路径解析到 data 目录内，兼容 "data/xxx.md" 写法。"""
        candidate = (base / raw).resolve()
        if candidate.exists():
            return candidate
        normalized = raw.replace("\\", "/")
        if normalized.lower().startswith("data/"):
            alternative = (base / normalized[len("data/") :]).resolve()
            if alternative.exists():
                return alternative
        return candidate

    def read_text_file(path: str) -> str:
        """读取 data 目录内的纯文本文件，返回其内容（超过上限会截断）。"""
        raw = (path or "").strip().strip('"').strip("'")
        if not raw:
            return "错误：请提供文件路径，例如 data/开源许可证速查.md"
        if not base.is_dir():
            return f"错误：数据目录 {base} 不存在。"

        candidate = _resolve(raw)
        try:
            relative = candidate.relative_to(base)
        except ValueError:
            return f"错误：出于安全考虑，只允许读取 {base} 目录内的文件，拒绝访问 {raw!r}。"

        if not candidate.is_file():
            return f"错误：文件不存在：{relative.as_posix()}"

        if candidate.suffix.lower() not in TEXT_SUFFIXES:
            allowed = "、".join(sorted(TEXT_SUFFIXES))
            return f"错误：只支持读取纯文本文件（支持的后缀：{allowed}）。"

        size = candidate.stat().st_size
        if size > MAX_BYTES:
            return f"错误：文件过大（{size} 字节），单文件上限 {MAX_BYTES} 字节。"

        try:
            text = candidate.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"错误：{relative.as_posix()} 不是 UTF-8 编码的文本文件，无法读取。"

        truncated = len(text) > max_chars
        body = text[:max_chars]
        tail = ""
        if truncated:
            tail = f"\n\n（内容过长，仅显示前 {max_chars} 个字符，全文共 {len(text)} 个字符）"
        return f"文件 {relative.as_posix()} 的内容如下：\n---\n{body}{tail}"

    return StructuredTool.from_function(
        func=read_text_file,
        name="read_file",
        description=(
            "读取 data 目录内的文本文件（md/txt/json/csv 等）。"
            "参数为相对 data 的路径，例如 开源许可证速查.md；写成 data/开源许可证速查.md 也可以。"
            "只读、有白名单限制，无法访问目录外的文件。"
        ),
    )
