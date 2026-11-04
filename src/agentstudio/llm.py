"""大模型工厂：所有服务商都通过 OpenAI 兼容协议接入，一个函数搞定。"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from .config import ConfigError, Settings


def build_chat_model(settings: Settings) -> BaseChatModel:
    """按配置创建聊天模型。

    由于 DeepSeek / Kimi / 通义 / GLM / 硅基流动 / 本地 Ollama 都提供
    OpenAI 兼容接口，这里只需要一个 ChatOpenAI，换厂商只是换 base_url。
    """
    settings = settings.validated()
    try:
        return ChatOpenAI(
            model=settings.model,
            api_key=SecretStr(settings.api_key),
            base_url=settings.base_url,
            temperature=settings.temperature,
            timeout=settings.timeout,
            max_retries=2,
            streaming=True,
        )
    except Exception as exc:  # noqa: BLE001 - 统一转成可展示的中文错误
        raise ConfigError(f"初始化模型失败：{type(exc).__name__}: {exc}") from exc
