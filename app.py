"""AgentStudio 智能体工坊 —— Streamlit 聊天界面。

启动：
    streamlit run app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st

# 允许"克隆 + 装依赖"直接运行，无需先 pip install -e .
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agentstudio.agent import AgentError, AgentReply, AgentSession  # noqa: E402
from agentstudio.config import PROVIDERS, ConfigError, Settings  # noqa: E402
from agentstudio.tools import AVAILABLE_TOOLS  # noqa: E402

st.set_page_config(
    page_title="AgentStudio 智能体工坊",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

EXAMPLES: tuple[str, ...] = (
    "帮我算一下 (12+8)*3 的结果，顺便告诉我今天星期几。",
    "搜索一下 LangChain 的最新版本，并说说它和 LangGraph 的关系。",
    "读一下 data/开源许可证速查.md，用一句话总结 MIT 与 GPL-3.0 的区别。",
    "现在北京时间几点？同时给我 UTC 时间。",
)


# ---------------------------------------------------------------- 侧边栏


def render_sidebar() -> tuple[Settings, bool]:
    """渲染配置面板，返回 (Settings, 是否显示工具轨迹)。"""
    defaults = Settings.from_env()
    providers = list(PROVIDERS)
    default_index = providers.index(defaults.provider) if defaults.provider in providers else 0

    with st.sidebar:
        st.header("⚙️ 模型与工具")

        provider = st.selectbox(
            "大模型服务商",
            options=providers,
            index=default_index,
            format_func=lambda name: PROVIDERS[name].label,
        )
        profile = PROVIDERS[provider]

        model = st.text_input("模型名称", value=profile.default_model, key=f"model::{provider}")

        api_key = st.text_input(
            "API Key",
            type="password",
            key=f"key::{provider}",
            help=f"留空则读取 .env 里的 {profile.key_env}。申请地址：{profile.signup_url}",
        )
        if not api_key and os.getenv(profile.key_env) == "" and defaults.api_key:
            api_key = defaults.api_key

        with st.expander("高级设置", expanded=False):
            base_url = st.text_input(
                "接口地址（base_url）",
                value=profile.base_url,
                key=f"url::{provider}",
                help="换成任意 OpenAI 兼容网关即可，例如本地 Ollama：http://localhost:11434/v1",
            )
            temperature = st.slider("temperature（越低越稳定）", 0.0, 1.5, float(defaults.temperature), 0.1)
            max_iterations = st.slider("单轮最多调用工具次数", 1, 12, int(defaults.max_iterations))
            memory_window = st.slider("记忆最近几轮对话", 0, 20, int(defaults.memory_window))
            tools = st.multiselect(
                "启用的工具",
                options=list(AVAILABLE_TOOLS),
                default=[name for name in defaults.enabled_tools if name in AVAILABLE_TOOLS],
            )

        st.divider()
        show_trace = st.toggle("显示工具调用过程", value=True)
        if st.button("🧹 清空对话", use_container_width=True):
            st.session_state.pop("session", None)
            st.session_state.pop("settings", None)
            st.session_state.pop("messages", None)
            st.rerun()

        st.caption(f"当前：{profile.label} · {model or '未指定模型'}")
        st.divider()
        st.caption(
            "本项目基于 LangChain 构建，代码以 MIT 协议开源。\n\n"
            "工具调用全部在本地进程内执行，API Key 只保存在会话内存中。"
        )

    settings = Settings(
        provider=provider,
        model=model.strip() or profile.default_model,
        api_key=(api_key or os.getenv(profile.key_env, "") or os.getenv("LLM_API_KEY", "")).strip(),
        base_url=(base_url or profile.base_url).strip(),
        temperature=float(temperature),
        max_iterations=int(max_iterations),
        memory_window=int(memory_window),
        enabled_tools=tuple(tools),
        data_dir=Path("data"),
    )
    return settings, show_trace


# ---------------------------------------------------------------- 渲染


def render_trace(steps: list[dict[str, Any]], elapsed: float) -> None:
    with st.expander(f"🔧 工具调用 · {len(steps)} 步 · 耗时 {elapsed:.1f}s", expanded=True):
        for index, step in enumerate(steps, start=1):
            st.markdown(f"**第 {index} 步 · `{step['tool']}`**")
            st.caption("输入")
            st.code(str(step["input"]), language="text")
            st.caption("观察结果")
            st.code(str(step["observation"])[:1500], language="text")


def render_history(show_trace: bool) -> None:
    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            steps = message.get("steps") or []
            if show_trace and steps:
                render_trace(steps, float(message.get("elapsed", 0.0)))


def ask_and_render(session: AgentSession, question: str, show_trace: bool) -> None:
    messages: list[dict[str, Any]] = st.session_state.setdefault("messages", [])
    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("智能体思考中…"):
            try:
                reply: AgentReply = session.ask(question)
            except AgentError as exc:
                st.error(str(exc))
                messages.append({"role": "assistant", "content": f"⚠️ {exc}"})
                return

        st.markdown(reply.output or "（模型没有返回内容）")
        steps = [
            {"tool": step.tool, "input": step.tool_input, "observation": step.observation}
            for step in reply.steps
        ]
        if show_trace and steps:
            render_trace(steps, reply.elapsed)
        st.caption(f"耗时 {reply.elapsed:.1f}s ｜ 工具调用 {reply.tool_calls} 次")

    messages.append(
        {
            "role": "assistant",
            "content": reply.output or "（模型没有返回内容）",
            "steps": steps,
            "elapsed": reply.elapsed,
        }
    )


# ---------------------------------------------------------------- 主流程


def main() -> None:
    settings, show_trace = render_sidebar()

    st.title("🧠 AgentStudio 智能体工坊")
    st.caption(
        "基于 **LangChain** 的工具调用智能体：会搜索、会计算、会看时间、会读本地文档。"
        "切换服务商即可在 DeepSeek / Kimi / 通义 / GLM 等在线大模型之间切换。"
    )

    if not settings.api_key:
        st.warning(
            "还没有配置 API Key。两种方式任选：\n"
            f"1. 左侧输入框直接粘贴（仅在当前会话内存中保留）；\n"
            f"2. 复制 `.env.example` 为 `.env`，填入 `{PROVIDERS[settings.provider].key_env}` 后刷新页面。"
        )
        st.stop()

    session: AgentSession | None = st.session_state.get("session")
    if session is None or st.session_state.get("settings") != settings:
        try:
            session = AgentSession(settings)
        except ConfigError as exc:
            st.error(str(exc))
            st.stop()
        st.session_state["session"] = session
        st.session_state["settings"] = settings
        st.session_state.setdefault("messages", [])

    render_history(show_trace)

    if not st.session_state.get("messages"):
        st.markdown("#### 试试这些例子")
        columns = st.columns(2)
        for index, example in enumerate(EXAMPLES):
            if columns[index % 2].button(example, key=f"example::{index}", use_container_width=True):
                st.session_state["pending"] = example
                st.rerun()

    pending = st.session_state.pop("pending", None)
    typed = st.chat_input("输入你的问题，例如：帮我查一下今天的 AI 新闻并总结三点")
    question = pending or typed
    if question:
        ask_and_render(session, question, show_trace)


main()
