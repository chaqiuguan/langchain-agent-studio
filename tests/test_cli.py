"""命令行与演示脚本测试：只覆盖不需要网络的路径。"""

from __future__ import annotations

import pytest

from agentstudio import cli, demo


def test_list_tools(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--list-tools"]) == 0
    out = capsys.readouterr().out
    for name in ("web_search", "calculator", "current_datetime", "read_file"):
        assert name in out


def test_missing_api_key_returns_error(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("LLM_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert cli.main(["-q", "你好"]) == 1
    assert "API Key" in capsys.readouterr().out


def test_offline_demo_runs(capsys: pytest.CaptureFixture[str]) -> None:
    assert demo.main() == 0
    out = capsys.readouterr().out
    assert "calculator" in out
    assert "工具调用" in out
