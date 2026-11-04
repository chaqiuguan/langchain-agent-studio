"""配置层：把 .env、环境变量与界面上的选择收敛成一个不可变的 Settings 对象。

设计要点
--------
1. 所有可调参数都有默认值，开箱只缺一个 API Key；
2. 缺 Key 时抛出的 ConfigError 消息是"给人看"的，可以直接展示在界面上；
3. Settings 是 frozen dataclass，可以在 Streamlit 里直接比较是否发生变化。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录（src/agentstudio/config.py -> 上两级）
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 先加载项目根目录的 .env，再兜底允许从当前工作目录向上查找
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()


class ConfigError(RuntimeError):
    """配置缺失或不合法。消息面向最终用户，可直接展示在界面上。"""


@dataclass(frozen=True)
class ProviderProfile:
    """一家 OpenAI 兼容服务商的静态信息。"""

    name: str
    label: str
    base_url: str
    default_model: str
    key_env: str
    signup_url: str


# 全部厂商都提供 OpenAI 兼容接口，因此可以共用 langchain_openai.ChatOpenAI
PROVIDERS: dict[str, ProviderProfile] = {
    "deepseek": ProviderProfile(
        name="deepseek",
        label="DeepSeek 深度求索",
        base_url="https://api.deepseek.com/v1",
        default_model="deepseek-chat",
        key_env="DEEPSEEK_API_KEY",
        signup_url="https://platform.deepseek.com/api_keys",
    ),
    "openai": ProviderProfile(
        name="openai",
        label="OpenAI",
        base_url="https://api.openai.com/v1",
        default_model="gpt-4o-mini",
        key_env="OPENAI_API_KEY",
        signup_url="https://platform.openai.com/api-keys",
    ),
    "moonshot": ProviderProfile(
        name="moonshot",
        label="Moonshot 月之暗面（Kimi）",
        base_url="https://api.moonshot.cn/v1",
        default_model="moonshot-v1-8k",
        key_env="MOONSHOT_API_KEY",
        signup_url="https://platform.moonshot.cn/console/api-keys",
    ),
    "qwen": ProviderProfile(
        name="qwen",
        label="阿里云百炼（通义千问）",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        default_model="qwen-plus",
        key_env="DASHSCOPE_API_KEY",
        signup_url="https://bailian.console.aliyun.com/",
    ),
    "zhipu": ProviderProfile(
        name="zhipu",
        label="智谱 GLM",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        default_model="glm-4-flash",
        key_env="ZHIPUAI_API_KEY",
        signup_url="https://bigmodel.cn/usercenter/apikeys",
    ),
    "siliconflow": ProviderProfile(
        name="siliconflow",
        label="硅基流动 SiliconFlow",
        base_url="https://api.siliconflow.cn/v1",
        default_model="Qwen/Qwen2.5-7B-Instruct",
        key_env="SILICONFLOW_API_KEY",
        signup_url="https://cloud.siliconflow.cn/account/ak",
    ),
    # 本地 Ollama / vLLM / one-api 等任意 OpenAI 兼容网关：需自己填 LLM_BASE_URL
    "custom": ProviderProfile(
        name="custom",
        label="自定义（OpenAI 兼容网关 / 本地 Ollama）",
        base_url="",
        default_model="qwen2.5:7b",
        key_env="LLM_API_KEY",
        signup_url="https://docs.ollama.com/openai",
    ),
}

DEFAULT_PROVIDER = "deepseek"

DEFAULT_TOOLS: tuple[str, ...] = ("web_search", "calculator", "current_datetime", "read_file")

DEFAULT_SYSTEM_PROMPT = """你是「AgentStudio 智能体工坊」的助手，一个可靠、务实、会使用工具的中文智能体。

可用工具：
- web_search：联网搜索，用于新闻、价格、天气、赛事、版本发布等实时信息；
- calculator：数学计算，支持四则运算、乘方、开方、三角函数、对数；
- current_datetime：查询指定时区的当前时间与星期；
- read_file：读取 data 目录内的文本文件，用于回答与本地资料相关的问题。

工作准则：
1. 需要实时信息时先搜索，不要凭记忆编造事实、数字或链接；
2. 任何数值计算都交给 calculator，不要心算；
3. 涉及"今天、现在、最近、今年"等时间概念时，先调用 current_datetime 确认基准时间；
4. 用户问到资料文件时，先用 read_file 读取，再基于原文回答；
5. 工具调用失败时最多换一种方式重试一次，仍失败就如实说明，不要编造结果；
6. 用中文回答：先给结论，再给必要的解释；引用搜索结果时附上来源链接。"""


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if not raw:
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env(name).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    """一次会话所需的全部配置。"""

    provider: str = DEFAULT_PROVIDER
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.3
    timeout: float = 60.0
    max_iterations: int = 6
    memory_window: int = 8
    agent_mode: str = "tool_calling"
    enabled_tools: tuple[str, ...] = DEFAULT_TOOLS
    verbose: bool = False
    data_dir: Path = field(default_factory=lambda: Path("data"))
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    # ---------- 构造 ----------

    @classmethod
    def from_env(cls) -> Settings:
        """从 .env / 环境变量读取全部配置。"""
        provider = _env("LLM_PROVIDER", DEFAULT_PROVIDER).lower()
        profile = PROVIDERS.get(provider, PROVIDERS[DEFAULT_PROVIDER])
        tools_raw = _env("AGENT_TOOLS", ",".join(DEFAULT_TOOLS))
        tools = tuple(item.strip() for item in tools_raw.split(",") if item.strip())
        return cls(
            provider=provider,
            model=_env("LLM_MODEL") or profile.default_model,
            api_key=_env("LLM_API_KEY") or _env(profile.key_env),
            base_url=_env("LLM_BASE_URL") or profile.base_url,
            temperature=_env_float("LLM_TEMPERATURE", 0.3),
            timeout=_env_float("LLM_TIMEOUT", 60.0),
            max_iterations=_env_int("AGENT_MAX_ITERATIONS", 6),
            memory_window=_env_int("AGENT_MEMORY_WINDOW", 8),
            agent_mode=_env("AGENT_MODE", "tool_calling").lower(),
            enabled_tools=tools,
            verbose=_env_bool("AGENT_VERBOSE", False),
            data_dir=Path(_env("AGENT_DATA_DIR", "data")),
        )

    # ---------- 派生属性 ----------

    @property
    def profile(self) -> ProviderProfile:
        return PROVIDERS.get(self.provider, PROVIDERS[DEFAULT_PROVIDER])

    def describe(self) -> str:
        """给界面用的一行摘要，例如 `DeepSeek 深度求索 · deepseek-chat · T=0.3`。"""
        return f"{self.profile.label} · {self.model or '未指定'} · T={self.temperature:g}"

    # ---------- 校验 ----------

    def validated(self) -> Settings:
        """校验必填项；不合法时抛出带中文提示的 ConfigError。"""
        if self.provider not in PROVIDERS:
            valid = "、".join(PROVIDERS)
            raise ConfigError(f"未知的模型服务商 {self.provider!r}，可选：{valid}")
        if not self.model.strip():
            raise ConfigError("未配置模型名称：请设置 LLM_MODEL，或在界面上填写模型名。")
        if not self.api_key.strip():
            profile = self.profile
            raise ConfigError(
                f"缺少 API Key（{profile.key_env} 或 LLM_API_KEY）。\n"
                f"1) 复制 .env.example 为 .env；\n"
                f"2) 到 {profile.signup_url} 申请 Key；\n"
                f"3) 写入 .env 后重启程序，或在左侧边栏直接粘贴 Key。"
            )
        if not self.base_url.strip():
            raise ConfigError("缺少接口地址：自定义服务商需要在 .env 中设置 LLM_BASE_URL。")
        if not 0.0 <= self.temperature <= 2.0:
            raise ConfigError(f"temperature 必须在 0~2 之间，当前为 {self.temperature}。")
        if self.max_iterations < 1:
            raise ConfigError("AGENT_MAX_ITERATIONS 至少为 1。")
        if self.memory_window < 0:
            raise ConfigError("AGENT_MEMORY_WINDOW 不能为负数。")
        if self.agent_mode not in {"tool_calling", "react"}:
            raise ConfigError("AGENT_MODE 只能是 tool_calling 或 react。")
        if not self.enabled_tools:
            raise ConfigError("至少需要启用一个工具（AGENT_TOOLS）。")
        return self

    def with_overrides(self, **kwargs: object) -> Settings:
        """返回替换了若干字段的新 Settings。"""
        return replace(self, **kwargs)  # type: ignore[arg-type]
