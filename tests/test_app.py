"""界面冒烟测试：用 Streamlit 官方 AppTest 无头执行 app.py，不需要浏览器与网络。"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture()
def no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("LLM_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_app_shows_title(no_key: None) -> None:
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception
    assert any("AgentStudio" in title.value for title in at.title)


def test_app_warns_without_api_key(no_key: None) -> None:
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert any("API Key" in warning.value for warning in at.warning)


def test_app_renders_chat_input_with_key(no_key: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-only")
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    assert not any("还没有配置 API Key" in warning.value for warning in at.warning)
    assert len(at.chat_input) == 1
