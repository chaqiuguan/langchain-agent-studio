"""联网搜索工具：Bing 网页搜索优先（国内可直连、无需 Key），ddgs 兜底。

为什么不用单一后端
------------------
DuckDuckGo 在国内网络基本不可达，一次调用会卡十几秒；而 Bing 的普通网页结果
无需 API Key、国内可直连、解析稳定。因此默认顺序是"Bing → ddgs"，
也可以用环境变量 `SEARCH_PROVIDER=bing|ddg|auto` 强制指定。

稳定性设计
----------
1. 每次搜索都在守护线程里执行并设硬超时，避免底层 HTTP 重试把一次调用拖到几十秒；
2. 连续失败达到阈值后熔断，直接返回"暂不可用"，防止模型反复重试。
"""

from __future__ import annotations

import html as html_lib
import os
import re
import threading
from typing import Any

import requests
from langchain_core.tools import BaseTool, StructuredTool

MAX_RESULTS_LIMIT = 8
SNIPPET_LIMIT = 220

_BING_URL = "https://cn.bing.com/search"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}
_TAG_RE = re.compile(r"<[^>]+>")


# ---------------------------------------------------------------- 结果整理


def _field(item: Any, *names: str) -> str:
    """兼容 dict 与对象两种返回结构，按顺序取第一个非空字段。"""
    for name in names:
        if isinstance(item, dict):
            value = item.get(name)
        else:
            value = getattr(item, name, None)
        if value:
            return str(value)
    return ""


def format_results(query: str, items: list[Any], limit: int = 5) -> str:
    """把搜索结果整理成紧凑的文本，方便模型阅读与引用。"""
    selected = items[:limit]
    if not selected:
        return (
            f"未搜索到与「{query}」相关的结果。"
            "可以换一个更具体的关键词重试，或基于已有知识回答并说明未经核实。"
        )
    lines = [f"关于「{query}」的搜索结果（{len(selected)} 条）："]
    for index, item in enumerate(selected, start=1):
        title = _field(item, "title") or "(无标题)"
        url = _field(item, "href", "url", "link")
        body = " ".join(_field(item, "body", "snippet", "description").split())
        if len(body) > SNIPPET_LIMIT:
            body = body[:SNIPPET_LIMIT] + "…"
        lines.append(f"{index}. {title}")
        if url:
            lines.append(f"   来源：{url}")
        if body:
            lines.append(f"   摘要：{body}")
    lines.append("提示：回答时请标注来源链接，并区分“搜索结果”与“模型推断”。")
    return "\n".join(lines)


# ---------------------------------------------------------------- 后端实现


def _clean(fragment: str) -> str:
    return html_lib.unescape(_TAG_RE.sub("", fragment)).strip()


def _bing_results(keyword: str, limit: int, timeout: float) -> list[dict[str, str]]:
    """解析 Bing 网页结果（<li class="b_algo"> 块），无需 API Key。"""
    response = requests.get(
        _BING_URL,
        params={"q": keyword, "setlang": "zh-CN", "count": limit},
        headers=_HEADERS,
        timeout=timeout,
    )
    response.raise_for_status()

    items: list[dict[str, str]] = []
    for block in re.split(r'(?=<li class="b_algo")', response.text)[1:]:
        match = re.search(r'<h2[^>]*>\s*<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', block, re.S)
        if not match:
            continue
        url, title = html_lib.unescape(match.group(1)), _clean(match.group(2))
        if not url.startswith("http") or not title:
            continue
        snippet = ""
        snippet_match = re.search(r"<p[^>]*>(.*?)</p>", block, re.S)
        if snippet_match:
            snippet = _clean(snippet_match.group(1))
        items.append({"title": title, "href": url, "body": snippet})
        if len(items) >= limit:
            break
    return items


def _ddg_results(
    keyword: str, limit: int, region: str, backend: str, timeout: float
) -> list[Any]:
    from ddgs import DDGS

    try:
        client = DDGS(timeout=timeout)
    except TypeError:
        # 旧版本 DDGS 不接受 timeout 参数
        client = DDGS()
    try:
        return list(client.text(keyword, region=region, max_results=limit, backend=backend))
    except TypeError:
        # 兼容不同版本 ddgs 的关键字参数差异
        return list(client.text(keyword, max_results=limit))


def _fetch(
    keyword: str, limit: int, region: str, backend: str, timeout: float
) -> list[Any]:
    """按 SEARCH_PROVIDER 选择后端；auto 时 Bing 优先、失败退回 ddgs。"""
    provider = (os.getenv("SEARCH_PROVIDER") or "auto").strip().lower()
    if provider == "bing":
        return _bing_results(keyword, limit, timeout)
    if provider in {"ddg", "duckduckgo"}:
        return _ddg_results(keyword, limit, region, backend, timeout)

    try:
        items = _bing_results(keyword, limit, timeout)
        if items:
            return items
    except Exception:  # noqa: BLE001 - 后端失败就换下一个
        pass
    return _ddg_results(keyword, limit, region, backend, timeout)


def _run_with_timeout(
    keyword: str, limit: int, region: str, backend: str, timeout: float
) -> list[Any]:
    """在守护线程里执行搜索，超时就放弃 —— 避免底层重试把一次调用拖到几十秒。"""
    box: dict[str, Any] = {}

    def work() -> None:
        try:
            box["items"] = _fetch(keyword, limit, region, backend, timeout)
        except BaseException as exc:  # noqa: BLE001 - 原样抛回主线程
            box["error"] = exc

    thread = threading.Thread(target=work, name="web-search", daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutError(f"搜索超过 {timeout:g} 秒未返回")
    if "error" in box:
        raise box["error"]
    return box.get("items", [])


# ---------------------------------------------------------------- 工具工厂


def make_web_search_tool(
    max_results: int = 5,
    region: str = "wt-wt",
    backend: str = "auto",
    timeout: float = 12.0,
    max_consecutive_failures: int = 2,
) -> BaseTool:
    limit = max(1, min(int(max_results), MAX_RESULTS_LIMIT))
    state = {"failures": 0}

    def web_search(query: str) -> str:
        """联网搜索。输入简洁的关键词或一句话问题，返回带链接的摘要列表。"""
        keyword = (query or "").strip()
        if not keyword:
            return "错误：搜索关键词为空。"

        # 熔断：连续失败后不再发起请求，避免模型反复重试把一次回答拖到几分钟
        if state["failures"] >= max_consecutive_failures:
            return (
                "错误：联网搜索已连续失败，本次会话内暂停使用。"
                "请不要再调用本工具，直接说明无法获取实时信息，并给出用户可自行核实的入口。"
            )

        try:
            items = _run_with_timeout(keyword, limit, region, backend, timeout)
        except Exception as exc:  # noqa: BLE001 - 网络类异常统一降级
            state["failures"] += 1
            return (
                f"错误：联网搜索暂时不可用（{type(exc).__name__}: {exc}）。"
                "不要重复调用同一关键词；请直接告知用户无法获取实时信息，"
                "并用你已有的知识谨慎作答，同时说明内容未经实时核实。"
            )

        state["failures"] = 0
        return format_results(keyword, items, limit)

    return StructuredTool.from_function(
        func=web_search,
        name="web_search",
        description=(
            "联网搜索最新信息（新闻、价格、天气、赛事、软件版本、人物近况等）。"
            "输入 Query 关键词或一句话问题，返回标题、链接与摘要。"
        ),
    )
