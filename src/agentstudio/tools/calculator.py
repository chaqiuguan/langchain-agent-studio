"""安全计算器：用 AST 白名单求值，彻底避免 eval 注入。

为什么不用 eval
---------------
`eval("__import__('os').system('rm -rf /')")` 只需要一行就能让智能体变成
任意代码执行漏洞。大模型的输出不可信，因此这里只放行数值字面量、四则运算、
乘方和少数 math 函数，其余语法一律拒绝。
"""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable
from typing import Any

from langchain_core.tools import tool

MAX_EXPRESSION_LENGTH = 200
MAX_DEPTH = 25
MAX_POWER = 1000
MAX_FACTORIAL = 500

_BIN_OPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type, Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_CONSTANTS: dict[str, float] = {"pi": math.pi, "e": math.e, "tau": math.tau}

_FUNCTIONS: dict[str, Callable[..., Any]] = {
    name: getattr(math, name)
    for name in (
        "sqrt",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "atan2",
        "log",
        "log10",
        "log2",
        "exp",
        "floor",
        "ceil",
        "fabs",
        "degrees",
        "radians",
        "factorial",
        "gcd",
        "hypot",
    )
    if hasattr(math, name)
}
_FUNCTIONS.update({"abs": abs, "round": round, "min": min, "max": max, "pow": pow})


def safe_eval(expression: str) -> float:
    """解析并计算表达式，遇到不支持的语法抛 ValueError。"""
    text = (expression or "").strip()
    if not text:
        raise ValueError("表达式为空")
    if len(text) > MAX_EXPRESSION_LENGTH:
        raise ValueError(f"表达式过长（最多 {MAX_EXPRESSION_LENGTH} 个字符）")
    if any(ch in text for ch in "\"'`;"):
        raise ValueError("表达式只能包含数字与运算符，不能包含引号或分号")
    tree = ast.parse(text, mode="eval")
    return _eval(tree.body, 0)


def _eval(node: ast.AST, depth: int) -> Any:
    if depth > MAX_DEPTH:
        raise ValueError("表达式嵌套过深")

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("只支持数值字面量")
        return node.value

    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise ValueError(f"未知的符号：{node.id}")

    if isinstance(node, ast.BinOp):
        func = _BIN_OPS.get(type(node.op))
        if func is None:
            raise ValueError(f"不支持的运算符：{type(node.op).__name__}")
        left = _eval(node.left, depth + 1)
        right = _eval(node.right, depth + 1)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_POWER and abs(left) > 1:
            raise ValueError(f"指数过大（绝对值上限 {MAX_POWER}），拒绝计算以免耗尽内存")
        result = func(left, right)
        if isinstance(result, complex):
            raise ValueError("结果为复数，超出支持范围")
        return result

    if isinstance(node, ast.UnaryOp):
        func = _UNARY_OPS.get(type(node.op))
        if func is None:
            raise ValueError(f"不支持的一元运算符：{type(node.op).__name__}")
        return func(_eval(node.operand, depth + 1))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("只支持直接调用数学函数，例如 sqrt(2)、log(100, 10)")
        func = _FUNCTIONS.get(node.func.id)
        if func is None:
            raise ValueError(f"不支持的函数：{node.func.id}")
        if node.keywords:
            raise ValueError("不支持关键字参数")
        args = [_eval(arg, depth + 1) for arg in node.args]
        if node.func.id == "factorial":
            if not args or not isinstance(args[0], (int, float)):
                raise ValueError("factorial 需要一个非负整数")
            if args[0] < 0 or args[0] > MAX_FACTORIAL:
                raise ValueError(f"factorial 只支持 0~{MAX_FACTORIAL} 的整数")
        return func(*args)

    raise ValueError(f"不支持的语法：{type(node).__name__}")


def format_number(value: Any) -> str:
    """把计算结果格式化成人类友好的字符串（整数不带小数点）。"""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        return f"{value:.10g}"
    return str(value)


@tool("calculator", description="执行数学计算。输入必须是纯数学表达式，例如 (12+8)*3、sqrt(144)、2**10，不能包含文字或中文。")
def calculator(expression: str) -> str:
    """执行数学计算。

    Args:
        expression: 纯数学表达式字符串，例如 "(12+8)*3"、"sqrt(144)+2**10"、"log(100,10)"。
    """
    try:
        value = safe_eval(expression)
    except ZeroDivisionError:
        return "错误：除数为 0，无法计算。"
    except Exception as exc:  # noqa: BLE001 - 所有解析错误都转成可读文本
        return f"错误：{exc}"
    if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
        return "错误：计算结果不是有限数值。"
    return f"{expression.strip()} = {format_number(value)}"
