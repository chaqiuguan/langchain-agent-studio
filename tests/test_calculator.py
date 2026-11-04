"""计算器与安全求值器测试：重点验证"不可信输入无法越界"。"""

from __future__ import annotations

import pytest

from agentstudio.tools.calculator import calculator, format_number, safe_eval


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("1+1", 2),
        ("(12+8)*3", 60),
        ("2**10", 1024),
        ("10/4", 2.5),
        ("7//2", 3),
        ("7%4", 3),
        ("-5+3", -2),
        ("sqrt(144)", 12),
        ("log(100,10)", 2),
        ("round(3.14159,2)", 3.14),
        ("pi", 3.141592653589793),
    ],
)
def test_safe_eval_valid(expression: str, expected: float) -> None:
    assert safe_eval(expression) == pytest.approx(expected)


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo hacked')",
        "open('/etc/passwd').read()",
        "().__class__.__bases__",
        "eval('1+1')",
        "exec('x=1')",
        "x + 1",
        "[1,2,3]",
        "{'a': 1}",
        "1 if True else 2",
        "lambda: 1",
        "1;2",
        "f'{1}'",
    ],
)
def test_safe_eval_rejects_dangerous_input(expression: str) -> None:
    with pytest.raises(ValueError):
        safe_eval(expression)


def test_expression_length_limit() -> None:
    with pytest.raises(ValueError, match="过长"):
        safe_eval("1+" * 200 + "1")


def test_huge_power_rejected() -> None:
    with pytest.raises(ValueError, match="指数过大"):
        safe_eval("9**999999")


def test_factorial_limit() -> None:
    with pytest.raises(ValueError, match="factorial"):
        safe_eval("factorial(9999)")


def test_division_by_zero_is_friendly() -> None:
    assert calculator.invoke({"expression": "1/0"}).startswith("错误：除数为 0")


def test_tool_returns_equation() -> None:
    assert calculator.invoke({"expression": "(12+8)*3"}) == "(12+8)*3 = 60"


def test_tool_name_and_description() -> None:
    assert calculator.name == "calculator"
    assert "数学" in calculator.description


def test_format_number() -> None:
    assert format_number(60.0) == "60"
    assert format_number(2.5) == "2.5"
    assert format_number(10**12) == "1000000000000"
