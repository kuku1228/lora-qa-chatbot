"""
agent_eval.py —— 工具调用 Agent 的小规模测试集与决策准确率统计

为什么写这个文件：
    项目里 Agent 的效果一直是"能跑、能演示"，但没有任何量化指标。
    这个脚本给 Agent 补一个可复现的决策准确率统计，回答两个问题：
        1. 该调工具的时候，模型有没有调对工具、给对参数？
        2. 不该调工具的时候（历史指代、纯知识问答、"先去查一下"这类），模型有没有乱调？

指标（与 agent/agent_react.py 的实际行为对齐）：
    - 工具选择正确率：调用的工具名是否等于期望工具
    - 参数正确率：在工具选对的前提下，参数是否等价
    - 决策准确率：工具和参数都对（不应调用时 = 完全不调用）
    - 越界率：按错因分类统计，其中"不应调用却调用了"是最要命的失败模式

数据流：
    测试集(30 条) → 决策后端(LLM) → _parse_action(生产代码逐行副本)
                  → 工具执行 → Observation 回流 → 再决策 … → 记录调用序列

后端（--backend）：
    oracle  参考决策器，严格按测试集期望作答。用途是给测试装置做自检：
            如果它拿不到 100%，说明评测代码本身有 bug，先修脚本再谈模型。
    naive   故意写得"不动脑子"的规则决策器（看到数字就调计算器、
            看到"查一下"就编 URL）。用来验证指标能不能把失败模式区分出来。
    openai  真实模型，走 OpenAI 兼容接口（vLLM 部署的微调模型）。
            用法：python agent_eval.py --backend openai \
                     --base-url http://127.0.0.1:8000/v1 --model lora1
            这是唯一能得出"模型决策准确率"的后端，需要先在 GPU 机器上
            用 agent/serve_vllm.sh 起好服务。

只依赖 Python 标准库，不需要 GPU、不需要 langgraph。
"""

from __future__ import annotations

import argparse
import ast
import json
import operator
import os
import re
import socket
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any


class LLMTimeout(Exception):
    """单次 LLM 请求超时。单独一类异常，便于按用例捕获而不中断整轮评测。"""

# ============================================================================
# 一、测试集
#    期望行为全部由 agent/agent_react.py 的系统提示词 + tools.py 的工具
#    description 推导得出（谁该调、谁不该调），不是拍脑袋编的。
# ============================================================================

TOOL_CALC = "calculator"
TOOL_TIME = "get_current_time"
TOOL_FETCH = "fetch_url"


@dataclass
class Case:
    cid: str
    category: str          # 场景大类
    query: str
    expect: list[tuple[str, dict]]   # 期望的调用序列；[] 表示不应调用任何工具
    failure_mode: str      # 这条主要防的失败模式
    note: str = ""


def build_cases() -> list[Case]:
    cases: list[Case] = []

    def add(cid, category, query, expect, failure_mode, note=""):
        cases.append(Case(cid, category, query, expect, failure_mode, note))

    # ---------- 1. 该调计算器（明确的计算题） ----------
    add("C01", "该调·计算器", "帮我算一下 (23+67)*45 等于多少",
        [(TOOL_CALC, {"expression": "(23+67)*45"})],
        "漏调用：该用工具却直接凭模型口算",
        "项目 README 里的示例问题")
    add("C02", "该调·计算器", "计算 128 * 37 + 466",
        [(TOOL_CALC, {"expression": "128 * 37 + 466"})],
        "漏调用：该用工具却直接凭模型口算")
    add("C03", "该调·计算器", "12 的 5 次方是多少",
        [(TOOL_CALC, {"expression": "12**5"})],
        "漏调用；且需要把自然语言转成表达式")
    add("C04", "该调·计算器", "17 除以 4 的余数是多少",
        [(TOOL_CALC, {"expression": "17 % 4"})],
        "漏调用；需要识别“取余”对应 % 运算符")
    add("C05", "该调·计算器", "3.5 * (8 - 2.5) 等于几",
        [(TOOL_CALC, {"expression": "3.5 * (8 - 2.5)"})],
        "漏调用；含浮点与括号")

    # ---------- 2. 该调时间工具 ----------
    add("C06", "该调·时间", "现在几点了",
        [(TOOL_TIME, {})],
        "漏调用：模型没有实时信息却直接编时间")
    add("C07", "该调·时间", "今天星期几",
        [(TOOL_TIME, {})],
        "漏调用")
    add("C08", "该调·时间", "今天几号？",
        [(TOOL_TIME, {})],
        "漏调用")

    # ---------- 3. 该调网页抓取（用户给了明确链接） ----------
    add("C09", "该调·网页", "帮我看一下 https://www.python.org 这个页面上写了什么",
        [(TOOL_FETCH, {"url": "https://www.python.org"})],
        "漏调用；或调用但漏掉/写错 URL 参数")
    add("C10", "该调·网页", "读一下 https://example.com 的正文内容",
        [(TOOL_FETCH, {"url": "https://example.com"})],
        "漏调用；或把 URL 当自由文本处理")

    # ---------- 4. 多步：先调工具、拿到 Observation 后继续 ----------
    add("C11", "多步·先工具后回答", "先帮我查一下现在的时间，然后告诉我这个时间点适合做什么",
        [(TOOL_TIME, {})],
        "漏调用；或拿到 Observation 后不继续输出最终答案")
    add("C12", "多步·工具链", "先算 25*4，再算 300/6",
        [(TOOL_CALC, {"expression": "25*4"}), (TOOL_CALC, {"expression": "300/6"})],
        "只完成第一步就结束（Observation 后未继续决策）")

    # ---------- 5. 明确不该调工具：历史指代 ----------
    add("C13", "不该调·历史指代", "我上一个问题是什么？", [],
        "误调用：把对话指代当成需要外部信息的任务")
    add("C14", "不该调·历史指代", "我刚才问的第一个问题是什么", [],
        "误调用")
    add("C15", "不该调·历史指代", "刚才那个问题的答案再说一遍", [],
        "误调用：历史信息已在上下文里")
    add("C16", "不该调·历史指代", "我们最开始聊的是什么", [],
        "误调用")

    # ---------- 6. 明确不该调工具：纯知识问答 ----------
    add("C17", "不该调·知识问答", "什么是机器学习中的过拟合？", [],
        "误调用：模型自身知识即可回答")
    add("C18", "不该调·知识问答", "简单解释一下 LoRA 微调的原理。", [],
        "误调用")
    add("C19", "不该调·知识问答", "Python 中列表和元组的区别是什么？", [],
        "误调用")
    # 这条是关键的"陷阱"：包含"计算"字样，但问的是概念，不是算题
    add("C20", "不该调·知识问答", "计算复杂度是什么概念？", [],
        "误调用：关键词触发（含“计算”二字但不是算题）")
    add("C21", "不该调·知识问答", "矩阵乘法的计算规则是怎样的？", [],
        "误调用：关键词触发")

    # ---------- 7. 明确不该调工具：网页工具的越界用法 ----------
    add("C22", "不该调·网页越界", "帮我查一下 Flask 的最新版本", [],
        "误调用：没给 URL，模型可能编造链接去抓")
    add("C23", "不该调·网页越界", "去网上搜一下今天有什么新闻", [],
        "误调用：fetch_url 不是搜索引擎，且未给 URL")
    add("C24", "不该调·网页越界", "最近大模型领域有什么新进展？", [],
        "误调用：需要检索能力，超出工具范围，应说明局限")

    # ---------- 8. 明确不该调工具：闲聊 / 主观 / 自身能力 ----------
    add("C25", "不该调·闲聊", "你好", [],
        "误调用：寒暄不需要任何工具")
    add("C26", "不该调·闲聊", "你都能做什么？", [],
        "误调用：应介绍自身能力而不是去调工具")
    add("C27", "不该调·闲聊", "谢谢你", [],
        "误调用")
    add("C28", "不该调·闲聊", "1+1 等于几", [],
        "误调用：极简算式无需工具（本项目的提示词要求简单问题直接回答）")

    # ---------- 9. 边界：容易误判的混合情况 ----------
    add("C29", "边界·先追问", "帮我算一下那个数", [],
        "误调用：指代不明且无具体表达式，应该先追问")
    add("C30", "边界·超出工具能力", "帮我把这篇 PDF 里的表格提取出来", [],
        "误调用：工具集不支持文件处理，应说明能力边界")

    return cases


# ============================================================================
# 二、系统提示词
#    与 agent/agent_react.py 的 build_system_prompt() 一致（同样的协议、
#    同样的条 3/5/6 约束）。评测必须用真实提示词，否则测的不是同一个 Agent。
# ============================================================================

TOOL_SCHEMAS = [
    {
        "name": TOOL_CALC,
        "description": "对数学表达式做安全求值，支持 + - * / // % ** 与括号（例如计算题时使用）",
        "args": {"expression": "字符串，数学表达式，例如 '2*(3+4)'"},
    },
    {
        "name": TOOL_TIME,
        "description": "获取当前日期与时间（北京时间），用户问现在几点/今天几号/星期几时调用",
        "args": {},
    },
    {
        "name": TOOL_FETCH,
        "description": "仅当用户明确要求查询最新新闻、网页内容或外部资料时调用；对话内已有信息不要用它",
        "args": {"url": "字符串，网页链接", "max_chars": "可选整数，返回文本长度上限，默认 2000"},
    },
]


def build_system_prompt() -> str:
    schema_text = json.dumps(TOOL_SCHEMAS, ensure_ascii=False, indent=1)
    return f"""你是基于 LoRA 微调的中文智能问答助手，基座模型 Qwen2-7B-Instruct，已针对中文问答场景做过领域微调。
你采用 ReAct（Reasoning + Acting）推理模式：先思考，需要外部信息时调用工具，再根据工具结果给出最终回答。

可用工具（JSON Schema）：
{schema_text}

输出协议（必须严格遵守）：
1. 如果需要调用工具，只输出一行 JSON（不要任何多余文字），格式：
   {{"tool": "工具名", "args": {{"参数1": 值1, "参数2": 值2}}}}
2. 如果已有足够信息可以直接回答，直接输出最终答案文本（简洁、准确、专业，一般不超过 5 句话）。
3. 简单问题直接回答，不要为了用工具而用工具。
4. 只有工具返回失败时才基于模型自身知识回答，并说明使用了模型知识。
5. 对话历史已经全部包含在你的消息列表中。当用户询问"刚才/上一个/上次/之前"等指代对话内容的问题时，
   直接根据历史回答，绝对不要调用任何工具。
6. 能用对话历史或自身知识回答的，一律直接回答；工具只用于对话之外的真实外部信息（如实时时间、
   明确要求查证的网页）。"""


# ============================================================================
# 三、Action 解析
#    _parse_action 逐行取自 agent/agent_react.py（三层正则：内嵌 JSON /
#    代码块包裹 / 首个大括号块）。评测必须复用真实解析逻辑。
# ============================================================================

def parse_action(text: str):
    if not text:
        return None
    patterns = [
        re.compile(r'\{[^{}]*"tool"[^{}]*\}', re.S),
        re.compile(r'```(?:json)?\s*(\{.*?\})\s*```', re.S),
        re.compile(r'(\{.*\})', re.S),
    ]
    for pat in patterns:
        for m in pat.finditer(text):
            blob = m.group(1) if m.groups() else m.group(0)
            blob = blob.strip().strip("`").strip()
            try:
                obj = json.loads(blob)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and "tool" in obj and isinstance(obj.get("args"), dict):
                return obj["tool"], obj["args"]
    return None


# ============================================================================
# 四、工具实现
#     计算器/时间工具的求值逻辑逐行取自 agent/tools.py（AST 白名单）。
#     fetch_url 在评测里用桩替代：只记录 URL 并返回合成观测，不发真实网络请求，
#     保证结果可复现、不受网络影响。真实抓取逻辑见 agent/tools.py。
# ============================================================================

_ALLOWED_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}
_MAX_POW = 1000


def _safe_eval(node):
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


def tool_calculator(expression: str) -> str:
    try:
        expr = str(expression).strip().replace("x", "*").replace("×", "*").replace("÷", "/")
        if not expr:
            return "错误：空表达式"
        result = _safe_eval(ast.parse(expr, mode="eval"))
        if isinstance(result, float):
            result = round(result, 6)
        return f"{expr} = {result}"
    except Exception as exc:
        return f"计算失败：{exc}"


def tool_get_current_time(**_ignored) -> str:
    tz = timezone(timedelta(hours=8))
    now = datetime.now(tz)
    weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    return now.strftime(f"%Y-%m-%d %H:%M:%S {weekdays[now.weekday()]}")


def tool_fetch_url(url: str, max_chars: int = 2000, **_ignored) -> str:
    """评测桩：不联网，返回合成观测。"""
    return f"[评测桩] 已抓取 {url}，正文前 {max_chars} 字符（此处不发起真实请求）"


TOOL_IMPL = {
    TOOL_CALC: tool_calculator,
    TOOL_TIME: tool_get_current_time,
    TOOL_FETCH: tool_fetch_url,
}


def invoke_tool(name: str, args: dict) -> str:
    impl = TOOL_IMPL.get(name)
    if impl is None:
        return f"工具 {name} 不存在，可用工具：{list(TOOL_IMPL.keys())}"
    try:
        return impl(**args)
    except Exception as exc:
        return f"工具执行失败：{exc}"


# ============================================================================
# 五、决策后端
# ============================================================================

class Backend:
    name = "base"

    def decide(self, messages: list[dict]) -> str:
        raise NotImplementedError


class OracleBackend(Backend):
    """参考决策器：严格按测试集期望输出 Action。

    存在的意义只有一个——给测试装置做自检。它拿到 100% 是设计使然，
    **不代表任何模型能力**。如果它拿不到 100%，说明 parse_action / 工具
    执行 / 指标统计里有 bug，应该先修脚本。
    """

    name = "oracle"

    def __init__(self, cases: list[Case]):
        self.plan = {c.query: list(c.expect) for c in cases}

    def decide(self, messages: list[dict]) -> str:
        user_msgs = [m for m in messages if m["role"] == "user"]
        if not user_msgs:
            return "好的。"
        query = user_msgs[-1]["content"]
        plan = self.plan.get(query, [])
        # 已完成的决策步数 = 非 Observation 的 assistant 消息数
        # （生产代码里 Action 与 Observation 都追加为 assistant，Observation 必须排除）
        done = sum(
            1 for m in messages
            if m["role"] == "assistant" and not str(m["content"]).startswith("Observation:")
        )
        if done < len(plan):
            name, args = plan[done]
            return json.dumps({"tool": name, "args": args}, ensure_ascii=False)
        return f"关于「{query}」的最终回答。"


class NaiveBackend(Backend):
    """故意"不动脑子"的规则决策器。

    复刻的是几类典型错误：看到数字就算、看到"几点/几号"就查时间、
    看到"查一下"就编一个 URL 去抓、完全无视"指代历史不要调工具"的要求。
    用途是验证指标对失败模式有区分度（预期它会把"不该调"那一类打得很低）。
    """

    name = "naive"

    _TIME_RE = re.compile(r"几点|几号|星期|日期|时间")
    _SEARCH_RE = re.compile(r"查一下|搜一下|搜一搜|查查|最新|新闻|进展")
    _HISTORY_RE = re.compile(r"上一个|刚才|最开始|第一个问题")
    _MATHY_RE = re.compile(r"[\d]")

    def decide(self, messages: list[dict]) -> str:
        user_msgs = [m for m in messages if m["role"] == "user"]
        query = user_msgs[-1]["content"] if user_msgs else ""
        after_obs = any(
            m["role"] == "assistant" and str(m["content"]).startswith("Observation:")
            for m in messages
        )
        if after_obs:
            return f"根据工具返回的结果，「{query}」的答案是……"

        if self._MATHY_RE.search(query) and not self._HISTORY_RE.search(query):
            expr = query.replace("等于多少", "").replace("等于几", "")
            expr = re.sub(r"[^\d+\-*/().%^ ]", " ", expr).strip()
            if expr:
                return json.dumps({"tool": TOOL_CALC, "args": {"expression": expr}}, ensure_ascii=False)

        if self._TIME_RE.search(query):
            return json.dumps({"tool": TOOL_TIME, "args": {}}, ensure_ascii=False)

        if self._SEARCH_RE.search(query) or "http" in query:
            m = re.search(r"https?://[^\s，,。]+", query)
            url = m.group(0) if m else "https://www.example.com/search"
            return json.dumps({"tool": TOOL_FETCH, "args": {"url": url}}, ensure_ascii=False)

        if self._HISTORY_RE.search(query):
            # 典型的越界：历史指代也去调个工具
            return json.dumps({"tool": TOOL_TIME, "args": {}}, ensure_ascii=False)

        return f"「{query}」：这是可以直接回答的问题。"


class OpenAIBackend(Backend):
    """真实模型后端：走 OpenAI 兼容接口（vLLM 部署的微调模型）。"""

    name = "openai"

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY",
                 temperature: float = 0.0, timeout: int = 120, max_tokens: int = 128):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        self.max_tokens = max_tokens

    def decide(self, messages: list[dict]) -> str:
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }).encode("utf-8")
        req = urllib.request.Request(
            self.url, data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="ignore"))
            return (data["choices"][0]["message"]["content"] or "").strip()
        except (TimeoutError, socket.timeout) as exc:
            raise LLMTimeout(
                f"单次请求超过 {self.timeout}s（max_tokens={self.max_tokens}）。"
                f"生成太慢时可用 --max-tokens 调小，或 --timeout 调大。原始错误：{exc}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"连不上 LLM 后端 {self.url}：{exc}。"
                f"请先在 GPU 机器上运行 bash agent/serve_vllm.sh"
            ) from exc


# ============================================================================
# 六、ReAct 循环（与 agent/agent_react.py 的图结构一致）
#     model 节点 → route → tool 节点 → 回到 model，MAX_STEPS 上限同生产代码。
# ============================================================================

MAX_STEPS = 8


@dataclass
class RunResult:
    calls: list[tuple[str, dict]] = field(default_factory=list)
    steps: int = 0
    hit_max_steps: bool = False
    parse_failures: int = 0
    final_text: str = ""
    error: str = ""          # 该用例的失败原因（如超时），不影响其他用例


def run_case(case: Case, backend: Backend) -> RunResult:
    messages: list[dict] = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": case.query},
    ]
    res = RunResult()
    for _ in range(MAX_STEPS):
        try:
            text = backend.decide(messages)
        except LLMTimeout as exc:
            # 超时只作废这一个用例，不中断整轮评测
            res.error = f"超时: {exc}"
            break
        action = parse_action(text)
        messages.append({"role": "assistant", "content": text})
        res.steps += 1
        if action is None:
            res.final_text = text
            break
        name, args = action
        res.calls.append((name, args))
        obs = invoke_tool(name, args)
        messages.append({"role": "assistant", "content": f"Observation: {obs}"})
    else:
        res.hit_max_steps = True
    return res


# ============================================================================
# 七、判分
# ============================================================================

def _norm_expr(s: str) -> str:
    """表达式等价化：去空格、统一符号，允许模型写成 12 ** 5 或 12**5。"""
    s = str(s).strip().lower()
    s = s.replace(" ", "").replace("×", "*").replace("÷", "/").replace("^", "**")
    return s


def _args_match(tool: str, actual: dict, expected: dict) -> bool:
    if tool == TOOL_CALC:
        if "expression" not in actual:
            return False
        return _norm_expr(actual["expression"]) == _norm_expr(expected["expression"])
    if tool == TOOL_TIME:
        return True  # 无参数
    if tool == TOOL_FETCH:
        if "url" not in actual:
            return False
        return str(actual["url"]).strip().rstrip("/") == str(expected["url"]).strip().rstrip("/")
    return actual == expected


def grade(case: Case, res: RunResult) -> dict:
    exp_names = [n for n, _ in case.expect]
    got_names = [n for n, _ in res.calls]

    tool_ok = (got_names == exp_names)

    # 参数必须按【位置】逐位比对。
    # 注意：不能用 {工具名: 参数} 建索引——同一个工具在一条用例里可能被调用多次
    # （如 C12 "先算 25*4，再算 300/6"），按名字索引会让后面的把前面的覆盖掉。
    args_ok = tool_ok and all(
        _args_match(got_n, got_a, case.expect[i][1])
        for i, (got_n, got_a) in enumerate(res.calls)
    )
    # 是否发生了"多调/少调"
    extra = [n for n in got_names if n not in exp_names] if not tool_ok else []
    return {
        "cid": case.cid,
        "category": case.category,
        "query": case.query,
        "expect": exp_names,
        "got": got_names,
        "expect_args": case.expect,
        "got_args": res.calls,
        "tool_ok": tool_ok,
        "args_ok": args_ok,
        "correct": tool_ok and args_ok,
        "should_call": len(case.expect) > 0,
        "untouched": len(res.calls) == 0,
        "extra_calls": extra,
        "steps": res.steps,
        "hit_max_steps": res.hit_max_steps,
        "failure_mode": case.failure_mode,
        "note": case.note,
        "error": res.error,
    }


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    correct = sum(r["correct"] for r in rows)
    tool_ok = sum(r["tool_ok"] for r in rows)
    should = [r for r in rows if r["should_call"]]
    should_not = [r for r in rows if not r["should_call"]]

    def rate(num, den):
        return (num / den * 100) if den else 0.0

    # 错因分类
    reasons: dict[str, int] = {}
    for r in rows:
        if r["correct"]:
            continue
        if not r["should_call"] and r["got"]:
            key = "误调用（不该调却调了）"
        elif r["should_call"] and not r["got"]:
            key = "漏调用（该调却没调）"
        elif r["should_call"] and r["got"] and not r["tool_ok"]:
            key = "工具选错"
        elif r["tool_ok"] and not r["args_ok"]:
            key = "参数写错"
        else:
            key = "其他"
        reasons[key] = reasons.get(key, 0) + 1

    by_cat: dict[str, list[dict]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)

    return {
        "n": n,
        "tool_acc": rate(tool_ok, n),
        "decision_acc": rate(correct, n),
        "should_call_acc": rate(sum(r["correct"] for r in should), len(should)),
        "should_not_call_acc": rate(sum(r["correct"] for r in should_not), len(should_not)),
        "false_call_rate": rate(sum(1 for r in should_not if r["got"]), len(should_not)),
        "miss_call_rate": rate(sum(1 for r in should if not r["got"]), len(should)),
        "reasons": reasons,
        "by_cat": {k: rate(sum(x["correct"] for x in v), len(v)) for k, v in by_cat.items()},
        "cat_counts": {k: len(v) for k, v in by_cat.items()},
        "hit_max_steps": sum(r["hit_max_steps"] for r in rows),
        "errors": sum(1 for r in rows if r.get("error")),
    }


# ============================================================================
# 八、报告
# ============================================================================

def render_report(backend_name: str, rows: list[dict], s: dict) -> str:
    L: list[str] = []
    L.append("# Agent 工具调用决策评测")
    L.append("")
    L.append(f"- 后端：`{backend_name}`")
    L.append(f"- 用例数：{s['n']}")
    L.append(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append("")
    if backend_name != "openai":
        L.append("> # ⚠️ 本次报告不代表模型能力")
        L.append(">")
        L.append("> 这一轮用的是**自检后端**，不是模型。它的通过率是设计出来的，**不能**作为")
        L.append("> Agent 的效果指标引用。")
        L.append(">")
        if backend_name == "oracle":
            L.append("> - `oracle`：严格按测试集期望作答，用来验证评测代码本身没 bug。")
            L.append(">   它拿到 100% 才是正常的；拿不到就说明脚本有问题，要先修脚本。")
        else:
            L.append("> - `naive`：故意写坏的规则决策器，用来验证指标能否区分出错误行为。")
        L.append(">")
        L.append("> **真实模型指标请用**：`python agent_eval.py --backend openai \\")
        L.append(">   --base-url http://127.0.0.1:8000/v1 --model lora1`")
        L.append("> （需先在 GPU 机器上执行 `bash agent/serve_vllm.sh`）")
        L.append("")
    L.append("## 总体指标")
    L.append("")
    L.append("| 指标 | 数值 |")
    L.append("|---|---|")
    L.append(f"| 工具选择正确率 | {s['tool_acc']:.1f}% |")
    L.append(f"| **决策准确率**（工具+参数均正确） | **{s['decision_acc']:.1f}%** |")
    L.append(f"| 应调工具场景准确率 | {s['should_call_acc']:.1f}% |")
    L.append(f"| 不应调工具场景准确率 | {s['should_not_call_acc']:.1f}% |")
    L.append(f"| 误调用率（不该调却调了） | {s['false_call_rate']:.1f}% |")
    L.append(f"| 漏调用率（该调却没调） | {s['miss_call_rate']:.1f}% |")
    L.append(f"| 触达最大步数（MAX_STEPS={MAX_STEPS}）的用例 | {s['hit_max_steps']} |")
    L.append(f"| 请求失败/超时的用例 | {s['errors']} |")
    L.append("")
    L.append("## 错因分布")
    L.append("")
    if s["reasons"]:
        L.append("| 错因 | 用例数 |")
        L.append("|---|---|")
        for k, v in sorted(s["reasons"].items(), key=lambda x: -x[1]):
            L.append(f"| {k} | {v} |")
    else:
        L.append("无错误用例。")
    L.append("")
    L.append("## 分场景准确率")
    L.append("")
    L.append("| 场景 | 用例数 | 准确率 |")
    L.append("|---|---|---|")
    for k in sorted(s["by_cat"], key=lambda x: (x.startswith("不该调"), x)):
        L.append(f"| {k} | {s['cat_counts'][k]} | {s['by_cat'][k]:.1f}% |")
    L.append("")
    L.append("## 逐条明细")
    L.append("")
    L.append("| # | 场景 | 问题 | 期望工具 | 实际工具 | 结果 |")
    L.append("|---|---|---|---|---|---|")
    for r in rows:
        mark = "✅" if r["correct"] else ("⚠️ 参数错" if r["tool_ok"] else "❌")
        exp = ", ".join(r["expect"]) if r["expect"] else "（不调用）"
        got = ", ".join(r["got"]) if r["got"] else "（不调用）"
        q = r["query"].replace("|", "\\|")
        L.append(f"| {r['cid']} | {r['category']} | {q} | {exp} | {got} | {mark} |")
    L.append("")
    L.append("## 错误用例详情")
    L.append("")
    bad = [r for r in rows if not r["correct"]]
    if not bad:
        L.append("无。")
    else:
        for r in bad:
            L.append(f"### {r['cid']} · {r['category']}")
            L.append("")
            L.append(f"- 问题：{r['query']}")
            L.append(f"- 期望调用：`{r['expect_args'] or '（不调用任何工具）'}`")
            L.append(f"- 实际调用：`{r['got_args'] or '（未调用）'}`")
            L.append(f"- 这条主要考察的失败模式：{r['failure_mode']}")
            if r["note"]:
                L.append(f"- 备注：{r['note']}")
            L.append("")
    L.append("---")
    L.append("")
    L.append("## 关于这个测试集的说明")
    L.append("")
    L.append("- 期望行为不是拍脑袋定的，而是从 `agent/agent_react.py` 的系统提示词（尤其第 3、5、6 条）")
    L.append("  与 `agent/tools.py` 里各工具的 description 推导出来的。")
    L.append("- `parse_action` 与计算器/时间工具的求值逻辑是 `agent/` 下生产代码的逐行副本，")
    L.append("  保证测的是同一个协议和同一套工具行为。")
    L.append("- `fetch_url` 在评测中用桩替代，只记录 URL、不发真实网络请求，避免结果受网络影响；")
    L.append("  真实抓取实现见 `agent/tools.py`。")
    L.append("- 30 条用例**规模很小**，只能作为方向性参考，不足以支撑严格的统计结论。")
    L.append("  要得出可信结论，需要扩大用例数并做多次采样（参考 pass^k 类指标）。")
    return "\n".join(L)


# ============================================================================
# 九、入口
# ============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="Agent 工具调用决策评测")
    ap.add_argument("--backend", default="oracle",
                    choices=["oracle", "naive", "openai"],
                    help="决策后端：oracle/naive 为自检用，openai 接真实模型")
    ap.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8000/v1"),
                    help="OpenAI 兼容接口地址（backend=openai 时生效）")
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL", "lora1"),
                    help="模型名（对应 vLLM 的 --lora-modules 模块名）")
    ap.add_argument("--api-key", default=os.environ.get("LLM_API_KEY", "EMPTY"))
    ap.add_argument("--timeout", type=int, default=120,
                    help="单次请求超时秒数（openai 后端；生成慢时调大）")
    ap.add_argument("--max-tokens", type=int, default=128,
                    help="单次生成上限（openai 后端；工具决策的 Action 很短，调小可显著加速）")
    ap.add_argument("--out", default=os.path.join("output", "agent_eval.md"))
    ap.add_argument("--json-out", default=os.path.join("output", "agent_eval.json"))
    args = ap.parse_args()

    cases = build_cases()

    if args.backend == "oracle":
        backend: Backend = OracleBackend(cases)
    elif args.backend == "naive":
        backend = NaiveBackend()
    else:
        backend = OpenAIBackend(args.base_url, args.model, args.api_key,
                                timeout=args.timeout, max_tokens=args.max_tokens)
        print(f"使用真实模型后端：{args.base_url} / model={args.model} / "
              f"timeout={args.timeout}s / max_tokens={args.max_tokens}")

    try:
        rows = [grade(c, run_case(c, backend)) for c in cases]
    except RuntimeError as exc:
        print(f"\n[评测中止] {exc}\n", file=sys.stderr)
        if backend.name == "openai":
            print("提示：不带 --backend 参数可在本地直接跑自检（oracle / naive），"
                  "不需要 GPU 和模型服务。", file=sys.stderr)
        return 2
    s = summarize(rows)

    print()
    print("=" * 62)
    print(f"后端: {backend.name}   用例数: {s['n']}")
    print("=" * 62)
    print(f"  工具选择正确率              {s['tool_acc']:6.1f}%")
    print(f"  决策准确率（工具+参数）      {s['decision_acc']:6.1f}%")
    print(f"  应调工具场景准确率          {s['should_call_acc']:6.1f}%")
    print(f"  不应调工具场景准确率        {s['should_not_call_acc']:6.1f}%")
    print(f"  误调用率                    {s['false_call_rate']:6.1f}%")
    print(f"  漏调用率                    {s['miss_call_rate']:6.1f}%")
    if s["reasons"]:
        print("-" * 62)
        print("  错因分布:")
        for k, v in sorted(s["reasons"].items(), key=lambda x: -x[1]):
            print(f"    {k}: {v}")
    print("=" * 62)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(render_report(backend.name, rows, s))
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump({"backend": backend.name, "summary": s, "rows": rows},
                  f, ensure_ascii=False, indent=2)
    print(f"\n报告已写入: {args.out}")
    print(f"原始数据:   {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
