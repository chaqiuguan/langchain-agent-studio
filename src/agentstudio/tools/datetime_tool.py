"""时间工具：让智能体在被问到"今天/现在"时先看表，而不是凭记忆编造。"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import BaseTool, StructuredTool

DEFAULT_TIMEZONE = "Asia/Shanghai"
_WEEKDAYS = "一二三四五六日"


def now_in(timezone_name: str = DEFAULT_TIMEZONE) -> str:
    """返回指定时区的当前时间；时区非法时返回错误文本而不是抛异常。"""
    name = (timezone_name or DEFAULT_TIMEZONE).strip()
    try:
        tz = ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, KeyError):
        return (
            f"错误：未知时区 {name!r}。请使用 IANA 时区名，"
            "例如 Asia/Shanghai、Asia/Tokyo、UTC、America/New_York。"
        )
    moment = datetime.now(tz)
    raw_offset = moment.strftime("%z")
    offset = f"UTC{raw_offset[:3]}:{raw_offset[3:]}" if raw_offset else "UTC"
    return (
        f"{moment.strftime('%Y-%m-%d %H:%M:%S')}（星期{_WEEKDAYS[moment.weekday()]}，"
        f"{name}，{offset}，Unix 时间戳 {int(moment.timestamp())}）"
    )


def make_datetime_tool() -> BaseTool:
    return StructuredTool.from_function(
        func=now_in,
        name="current_datetime",
        description=(
            "查询指定时区的当前日期、时间与星期。涉及“今天/现在/最近/今年”等问题时应先调用本工具，"
            "参数为 IANA 时区名，默认 Asia/Shanghai。"
        ),
    )
