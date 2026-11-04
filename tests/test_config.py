"""配置层测试：环境变量解析、默认值、缺失 Key 的友好报错。"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from agentstudio.config import PROVIDERS, ConfigError, Settings


@pytest.fixture()
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "LLM_PROVIDER",
        "LLM_MODEL",
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "LLM_TEMPERATURE",
        "LLM_TIMEOUT",
        "AGENT_MAX_ITERATIONS",
        "AGENT_MEMORY_WINDOW",
        "AGENT_MODE",
        "AGENT_TOOLS",
        "AGENT_VERBOSE",
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def test_defaults(clean_env: None) -> None:
    settings = Settings.from_env()
    assert settings.provider == "deepseek"
    assert settings.model == "deepseek-chat"
    assert settings.base_url == PROVIDERS["deepseek"].base_url
    assert settings.enabled_tools == ("web_search", "calculator", "current_datetime", "read_file")


def test_reads_prefixed_env(clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.9")
    monkeypatch.setenv("AGENT_TOOLS", "calculator, current_datetime")
    settings = Settings.from_env()
    assert settings.provider == "openai"
    assert settings.model == "gpt-4o-mini"
    assert settings.api_key == "sk-openai"
    assert settings.temperature == pytest.approx(0.9)
    assert settings.enabled_tools == ("calculator", "current_datetime")


def test_llm_api_key_wins(clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "from-provider-env")
    monkeypatch.setenv("LLM_API_KEY", "from-generic-env")
    assert Settings.from_env().api_key == "from-generic-env"


def test_missing_key_raises_friendly_error(clean_env: None) -> None:
    with pytest.raises(ConfigError) as excinfo:
        Settings.from_env().validated()
    message = str(excinfo.value)
    assert "API Key" in message
    assert "platform.deepseek.com" in message


def test_unknown_provider_rejected(clean_env: None) -> None:
    settings = Settings(provider="nope", model="x", api_key="k", base_url="http://x")
    with pytest.raises(ConfigError, match="未知的模型服务商"):
        settings.validated()


def test_temperature_bounds(clean_env: None) -> None:
    settings = Settings(model="m", api_key="k", base_url="http://x", temperature=5.0)
    with pytest.raises(ConfigError, match="temperature"):
        settings.validated()


def test_empty_tools_rejected(clean_env: None) -> None:
    settings = Settings(model="m", api_key="k", base_url="http://x", enabled_tools=())
    with pytest.raises(ConfigError, match="至少需要启用一个工具"):
        settings.validated()


def test_custom_provider_requires_base_url(clean_env: None) -> None:
    settings = Settings(provider="custom", model="qwen2.5:7b", api_key="ollama", base_url="")
    with pytest.raises(ConfigError, match="LLM_BASE_URL"):
        settings.validated()


def test_describe(clean_env: None) -> None:
    settings = Settings(provider="deepseek", model="deepseek-chat", temperature=0.3)
    assert "deepseek-chat" in settings.describe()
    assert "T=0.3" in settings.describe()


def test_frozen(clean_env: None) -> None:
    settings = Settings()
    with pytest.raises(FrozenInstanceError):
        settings.model = "changed"  # type: ignore[misc]
