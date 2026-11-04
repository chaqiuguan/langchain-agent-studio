"""工具层测试：时间、文件读取（含越权访问）、搜索格式化。"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from agentstudio.tools import web_search as search_module
from agentstudio.tools.datetime_tool import make_datetime_tool, now_in
from agentstudio.tools.file_reader import make_file_reader_tool
from agentstudio.tools.web_search import format_results, make_web_search_tool

# ---------------------------------------------------------------- 时间


def test_now_in_shanghai() -> None:
    text = now_in("Asia/Shanghai")
    assert re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", text)
    assert "星期" in text
    assert "UTC+08:00" in text


def test_now_in_utc() -> None:
    assert "UTC" in now_in("UTC")


def test_now_in_invalid_timezone() -> None:
    result = now_in("Mars/Olympus")
    assert result.startswith("错误：未知时区")


def test_datetime_tool_metadata() -> None:
    tool = make_datetime_tool()
    assert tool.name == "current_datetime"
    assert "时区" in tool.description
    assert "星期" in tool.invoke({"timezone_name": "Asia/Shanghai"})


# ---------------------------------------------------------------- 文件读取


@pytest.fixture()
def data_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    root.mkdir()
    (root / "notes.md").write_text("# 标题\n正文内容", encoding="utf-8")
    (root / "binary.bin").write_bytes(b"\x00\x01\x02")
    (tmp_path / "secret.txt").write_text("机密", encoding="utf-8")
    return root


def test_read_file_ok(data_root: Path) -> None:
    tool = make_file_reader_tool(data_root)
    result = tool.invoke({"path": "notes.md"})
    assert "正文内容" in result
    assert "notes.md" in result


def test_read_file_blocks_traversal(data_root: Path) -> None:
    tool = make_file_reader_tool(data_root)
    result = tool.invoke({"path": "../secret.txt"})
    assert result.startswith("错误：出于安全考虑")


def test_read_file_blocks_absolute_path(data_root: Path) -> None:
    tool = make_file_reader_tool(data_root)
    result = tool.invoke({"path": str(data_root.parent / "secret.txt")})
    assert result.startswith("错误：")

def test_read_file_accepts_data_prefix(data_root: Path) -> None:
    tool = make_file_reader_tool(data_root)
    assert "正文内容" in tool.invoke({"path": "data/notes.md"})


def test_read_file_blocks_traversal_via_data_prefix(data_root: Path) -> None:
    tool = make_file_reader_tool(data_root)
    assert tool.invoke({"path": "data/../secret.txt"}).startswith("错误：出于安全考虑")


def test_read_file_missing(data_root: Path) -> None:
    tool = make_file_reader_tool(data_root)
    assert "不存在" in tool.invoke({"path": "nope.md"})


def test_read_file_rejects_binary_suffix(data_root: Path) -> None:
    tool = make_file_reader_tool(data_root)
    assert "只支持读取纯文本文件" in tool.invoke({"path": "binary.bin"})


def test_read_file_missing_dir(tmp_path: Path) -> None:
    tool = make_file_reader_tool(tmp_path / "not-exist")
    assert "数据目录" in tool.invoke({"path": "a.md"})


def test_read_file_empty_path(data_root: Path) -> None:
    tool = make_file_reader_tool(data_root)
    assert "请提供文件路径" in tool.invoke({"path": "  "})


def test_read_file_truncates(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    (root / "big.txt").write_text("A" * 500, encoding="utf-8")
    tool = make_file_reader_tool(root, max_chars=100)
    result = tool.invoke({"path": "big.txt"})
    assert "仅显示前 100 个字符" in result


# ---------------------------------------------------------------- 搜索


def test_format_results_renders_links() -> None:
    items = [
        {"title": "LangChain 发布 0.3", "href": "https://example.com/a", "body": "摘要 A"},
        {"title": "LangGraph 介绍", "href": "https://example.com/b", "body": "摘要 B"},
    ]
    text = format_results("langchain", items)
    assert "来源：https://example.com/a" in text
    assert "2. LangGraph 介绍" in text


def test_format_results_empty() -> None:
    assert "未搜索到" in format_results("不存在的东西", [])


def test_search_tool_handles_network_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> list[object]:
        raise RuntimeError("network down")

    monkeypatch.setattr(search_module, "_fetch", boom)
    tool = make_web_search_tool()
    result = tool.invoke({"query": "今天新闻"})
    assert result.startswith("错误：联网搜索暂时不可用")


def test_search_tool_uses_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_fetch(
        keyword: str, limit: int, region: str, backend: str, timeout: float = 15.0
    ) -> list[dict[str, str]]:
        calls.append(keyword)
        return [{"title": "T", "href": "https://x.test", "body": "B"}]

    monkeypatch.setattr(search_module, "_fetch", fake_fetch)
    tool = make_web_search_tool(max_results=3)
    result = tool.invoke({"query": "pytest"})
    assert calls == ["pytest"]
    assert "https://x.test" in result


def test_search_breaker_opens_after_repeated_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> list[object]:
        raise RuntimeError("down")

    monkeypatch.setattr(search_module, "_fetch", boom)
    tool = make_web_search_tool()
    first = tool.invoke({"query": "a"})
    tool.invoke({"query": "b"})
    third = tool.invoke({"query": "c"})
    assert first.startswith("错误：联网搜索暂时不可用")
    assert "已连续失败" in third


BING_PAGE = """<html><body>
<li class="b_algo"><h2><a href="https://a.test/x">标题 A</a></h2>
<p class="b_lineclamp">摘要 &amp; 说明</p></li>
<li class="b_algo"><h2><a href="https://b.test/y">标题 B</a></h2></li>
</body></html>"""


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


def test_bing_parser_extracts_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(search_module.requests, "get", lambda *a, **k: _FakeResponse(BING_PAGE))
    items = search_module._bing_results("q", 5, 5.0)
    assert [item["title"] for item in items] == ["标题 A", "标题 B"]
    assert items[0]["href"] == "https://a.test/x"
    assert "摘要" in items[0]["body"]


def test_fetch_prefers_bing_in_auto_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEARCH_PROVIDER", raising=False)
    monkeypatch.setattr(
        search_module, "_bing_results", lambda *a, **k: [{"title": "bing", "href": "https://b", "body": ""}]
    )

    def should_not_call(*args: object, **kwargs: object) -> list[object]:
        raise AssertionError("Bing 成功时不应调用 ddgs")

    monkeypatch.setattr(search_module, "_ddg_results", should_not_call)
    items = search_module._fetch("q", 2, "wt-wt", "auto", 1.0)
    assert items[0]["title"] == "bing"


def test_fetch_falls_back_to_ddg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEARCH_PROVIDER", raising=False)

    def bing_down(*args: object, **kwargs: object) -> list[object]:
        raise ConnectionError("blocked")

    monkeypatch.setattr(search_module, "_bing_results", bing_down)
    monkeypatch.setattr(
        search_module, "_ddg_results", lambda *a, **k: [{"title": "ddg", "href": "https://d", "body": ""}]
    )
    items = search_module._fetch("q", 2, "wt-wt", "auto", 1.0)
    assert items[0]["title"] == "ddg"


def test_search_tool_rejects_empty_query() -> None:
    tool = make_web_search_tool()
    assert tool.invoke({"query": "   "}).startswith("错误：搜索关键词为空")
