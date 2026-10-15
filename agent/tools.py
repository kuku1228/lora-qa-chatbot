"""
agent/tools.py —— ReAct Agent 工具集

所有工具真实可用、无需外部 API Key、纯 Python 实现，
用于展示"工具调用"能力（面试重点：工具封装、安全求值、LLM 与工具的交互）。
"""
import ast
import operator
import re
from datetime import datetime, timedelta, timezone
from urllib.request import Request, urlopen

from langchain_core.tools import tool

# ---------- 安全计算器 ----------
# 用 AST 白名单做表达式求值：只允许数字与基础运算符，
# 天然免疫 eval() 的代码注入风险（面试可讲这一点）。
_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}
_MAX_POW = 1000          # 防止 10**100000 这类超大幂运算打爆 CPU


def _safe_eval(node):
    """AST 白名单求值：只接受 数字常量 + 基础二元/一元运算符。"""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            raise ValueError("不支持布尔常量")
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("仅支持数字常量")
    if isinstance(node, ast.BinOp):
        op = _ALLOWED_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"不支持的运算符：{type(node.op).__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_POW:
            raise ValueError(f"指数过大（上限 {_MAX_POW}）")
        return op(left, right)
    if isinstance(node, ast.UnaryOp):
        op = _ALLOWED_OPS.get(type(node.op))
        if op is None:
            raise ValueError("不支持该一元运算符")
        return op(_safe_eval(node.operand))
    raise ValueError("表达式包含不允许的元素（函数调用/变量/比较等）")


@tool
def calculator(expression: str) -> str:
    """对数学表达式做安全求值。支持 + - * / // % ** 与括号。

    示例：'2*(3+4)'、'(12**2 + 5)/3'、'7 % 3'
    """
    try:
        expr = expression.strip().replace("x", "*").replace("×", "*").replace("÷", "/")
        if not expr:
            return "错误：空表达式"
        tree = ast.parse(expr, mode="eval")
        result = _safe_eval(tree)
        if isinstance(result, float):
            result = round(result, 6)
        return f"{expr} = {result}"
    except Exception as exc:  # 任何解析/求值异常都返回给模型，而不是抛错中断
        return f"计算失败：{exc}"


# ---------- 时间工具 ----------
@tool
def get_current_time() -> str:
    """获取当前日期与时间（北京时间）。当用户问"现在几点/今天几号/星期几"时调用。"""
    tz = timezone(timedelta(hours=8))  # UTC+8
    now = datetime.now(tz)
    weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    return now.strftime(f"%Y-%m-%d %H:%M:%S {weekdays[now.weekday()]}")


# ---------- 网页抓取工具 ----------
@tool
def fetch_url(url: str, max_chars: int = 2000) -> str:
    """抓取指定网页，返回其可读文本（去除标签）的前 max_chars 字符，用于查证外部信息。仅支持 http/https。"""
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0)"})
        with urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        # 去掉脚本/样式与标签，只留文本
        text = re.sub(r"<(script|style)[\s\S]*?</\1>", " ", raw, flags=re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars] if text else "（网页无可读文本内容）"
    except Exception as exc:
        return f"抓取失败：{exc}"


# ---------- 工具 JSON Schema（写进系统提示词，供模型遵循的调用协议） ----------
TOOL_SCHEMAS = [
    {
        "name": "calculator",
        "description": "对数学表达式做安全求值，支持 + - * / // % ** 与括号（例如计算题时使用）",
        "args": {"expression": "字符串，数学表达式，例如 '2*(3+4)'"},
    },
    {
        "name": "get_current_time",
        "description": "获取当前日期与时间（北京时间），用户问现在几点/今天几号/星期几时调用",
        "args": {},
    },
    {
        "name": "fetch_url",
        "description": "仅当用户明确要求查询最新新闻、网页内容或外部资料时调用；对话内已有信息不要用它",
        "args": {"url": "字符串，网页链接", "max_chars": "可选整数，返回文本长度上限，默认 2000"},
    },
]


# ---------- 工具集合 ----------
def build_toolset():
    """Agent 使用的全部工具列表（可扩展：加工具就在这里加一行）。"""
    return [calculator, get_current_time, fetch_url]


if __name__ == "__main__":
    # 本地冒烟测试（无需 GPU / 无需 LLM）
    for expr in ["2*(3+4)", "(12**2 + 5)/3", "7 % 3", "__import__('os')", "10**100000"]:
        print(calculator.invoke({"expression": expr}))
    print(get_current_time.invoke({}))
