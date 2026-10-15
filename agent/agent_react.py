"""
agent/agent_react.py —— LangGraph ReAct Agent（Prompt 模式，兼容 vLLM 0.8.5）

架构：LangGraph StateGraph 手写 ReAct 图（模型节点 → 工具节点 → 条件循环）
流程：Thought（模型思考/输出）→ JSON Action（工具调用指令）→ Observation（工具结果）→ 循环 → 最终回答
特性：MemorySaver 记忆（多轮对话上下文）、流式输出每步轨迹、最大步数防死循环

为什么不用模型原生 tool calling？
vLLM 0.8.5 的 --tool-call-parser 不提供 qwen 选项，无法解析 Qwen2 的原生
function calling 输出。本实现采用经典 ReAct 的 Prompt 模式：工具 schema 写入
系统提示词，模型输出 JSON 协议文本，代码解析并执行工具——不依赖模型原生
工具调用能力，任何模型版本均可运行，且对 7B 模型输出格式更可控。

运行前提：微调模型已通过 vLLM 部署为 OpenAI 兼容 API（见 serve_vllm.sh / README）
"""
import json
import operator
import re
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.llm import build_chat_model, health_check
from agent.tools import build_toolset, TOOL_SCHEMAS

MAX_STEPS = 8  # 最大推理步数（防死循环）


def _tool_schema_prompt(tools) -> str:
    """把工具集渲染成 JSON Schema 文本，写进系统提示词。"""
    schemas = getattr(tools, "schemas", None) or TOOL_SCHEMAS
    return json.dumps(schemas, ensure_ascii=False, indent=1)


def build_system_prompt(tools) -> str:
    schema_text = _tool_schema_prompt(tools)
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


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]  # 追加 reducer：多轮对话历史自动累积
    steps: int                               # 已执行步数（覆盖式）


# 模块级共享 checkpointer：主 ReAct 图与"禁工具"图共用同一份对话历史
_CHECKPOINTER = MemorySaver()


def build_react_agent(model=None, tools=None, system_prompt=None, checkpointer=None):
    """构建 LangGraph ReAct agent（Prompt 模式）。model/tools 可注入（便于测试）。"""
    model = model or build_chat_model()
    tools = tools if tools is not None else build_toolset()
    tool_map = {t.name: t for t in tools}
    system_prompt = system_prompt or build_system_prompt(tools)
    checkpointer = checkpointer or _CHECKPOINTER

    def call_model(state: AgentState) -> dict:
        """模型节点：system + 全部历史 → 模型输出文本。

        messages 使用 operator.add reducer，这里只返回【新增】的一条消息，
        历史由 LangGraph 自动累积（这就是多轮记忆的机制）。
        """
        messages = [{"role": "system", "content": system_prompt}] + _to_openai(state["messages"])
        resp = model.invoke(messages)
        text = (getattr(resp, "content", None) or "").strip()
        return {"messages": [("assistant", text)], "steps": state["steps"] + 1}

    def call_tool(state: AgentState) -> dict:
        """工具节点：解析最后一条 assistant 消息中的 JSON Action，执行工具，追加 Observation。"""
        last = state["messages"][-1]
        text = last.content if hasattr(last, "content") else str(last[1])
        parsed = _parse_action(text)
        if parsed is None:
            return {"messages": []}  # 无工具调用 → 不追加，直接结束
        name, args = parsed
        tool = tool_map.get(name)
        if tool is None:
            obs = f"工具 {name} 不存在，可用工具：{list(tool_map.keys())}"
        else:
            try:
                obs = tool.invoke(args)
            except Exception as exc:  # 工具执行异常也作为观察结果，让模型自行判断
                obs = f"工具执行失败：{exc}"
        return {"messages": [("assistant", f"Observation: {obs}")]}

    def route(state: AgentState) -> str:
        """条件路由：最后一条消息含 JSON Action → 执行工具；否则 → 结束。"""
        if state["steps"] >= MAX_STEPS:
            return "end"
        last = state["messages"][-1]
        text = last.content if hasattr(last, "content") else str(last[1])
        return "tool" if _parse_action(text) is not None else "end"

    graph = StateGraph(AgentState)
    graph.add_node("model", call_model)
    graph.add_node("tool", call_tool)
    graph.add_edge(START, "model")
    graph.add_conditional_edges(
        "model",
        route,
        {"tool": "tool", "end": END},
    )
    graph.add_edge("tool", "model")
    return graph.compile(checkpointer=checkpointer)


def _to_openai(messages) -> list:
    """把 LangChain 消息/元组列表转成 OpenAI dict 列表（供 ChatOpenAI 使用）。"""
    out = []
    for m in messages:
        if hasattr(m, "type"):  # BaseMessage
            role = "assistant" if m.type == "ai" else ("user" if m.type == "human" else "user")
            content = m.content or ""
        else:  # (role, content) tuple
            role, content = m[0], m[1]
        out.append({"role": role, "content": content})
    return out


def _parse_action(text: str):
    """从模型输出中提取 JSON Action：{"tool": ..., "args": {...}}。

    返回 (tool_name, args_dict)；若为普通回答返回 None。
    兼容：整行 JSON / 文本中内嵌 JSON 块 / 代码块包裹的 JSON。
    """
    if not text:
        return None
    patterns = [
        re.compile(r'\{[^{}]*"tool"[^{}]*\}', re.S),          # 内嵌 JSON 对象
        re.compile(r'```(?:json)?\s*(\{.*?\})\s*```', re.S),  # 代码块包裹
        re.compile(r'(\{.*\})', re.S),                        # 兜底：首个大括号块
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


_APP = None  # 模块级单例：agent 图只构建一次


def get_app():
    global _APP
    if _APP is None:
        _APP = build_react_agent()
    return _APP


def _render_latest(messages):
    """把最新一步消息渲染成可读轨迹文本。"""
    last = messages[-1]
    text = last.content if hasattr(last, "content") else str(last[1])
    text = (text or "").strip()
    if text.startswith("Observation:"):
        return f"📊 **工具返回**：{text[12:][:400]}"
    action = _parse_action(text)
    if action is not None:
        name, args = action
        return f"🔧 **调用工具 `{name}`**，参数：`{args}`"
    if text:
        return f"💡 **思考/回答**：{text}"
    return "💡 思考中…"


_HISTORY_Q_RE = re.compile(
    r"第一个问题|第一个问的|最开始问|最开始的问题|一开始问|一开始的问题|首个问题|"
    r"上一个问题|上一个问的|刚才问|刚才说了|刚才的问题|上次问|之前问|我问你的上?一条|我上?一条.*是什么|上一条消息"
)
_FIRST_Q_RE = re.compile(r"第一个|最开始|一开始|首个")
_NTH_Q_RE = re.compile(r"第\s*([一二三四五六七八九十两\d]+)\s*个\s*问题")
_PREV2_Q_RE = re.compile(r"再上一个|上上个|再之前|倒数第二个|再往前|更早")
_PREV_OF_RE = re.compile(r"“?([^，。？?！!\s]{2,20})”?的上一个(?:问题|问的)")
_CN_NUM = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
           "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _parse_num(s: str):
    """解析 1~99 的阿拉伯/中文数字。"""
    if s.isdigit():
        return int(s)
    if s in _CN_NUM:
        return _CN_NUM[s]
    if s == "十":
        return 10
    if len(s) == 2 and s[0] in _CN_NUM and s[1] == "十":
        return _CN_NUM[s[0]] * 10
    if s.startswith("十") and len(s) == 2 and s[1] in _CN_NUM:
        return 10 + _CN_NUM[s[1]]
    return None


def _question_kind(query: str):
    """解析对话指代问题，返回 (类型, 参数) 或 None。

    类型：first=第一个问题；lastn=倒数第 N 个问题（N>=1）；
          nth=第 N 个问题；prev_of=某个具体问题的上一个问题。
    """
    q = query or ""
    m = _NTH_Q_RE.search(q)
    if m:
        n = _parse_num(m.group(1))
        if n:
            return ("nth", n)
    if _PREV2_Q_RE.search(q):
        return ("lastn", 2)
    m = _PREV_OF_RE.search(q)
    if m:
        return ("prev_of", m.group(1))
    if _FIRST_Q_RE.search(q):
        return ("first", 1)
    if _HISTORY_Q_RE.search(q):
        return ("lastn", 1)
    return None


def _find_user_index(users: list, target: str):
    """在用户历史里定位 target（相等或互相包含），返回下标；找不到返回 -1。"""
    for i, u in enumerate(users):
        if target in u or u in target:
            return i
    return -1


def _user_texts(messages) -> list:
    """从历史消息中提取所有用户消息文本（顺序保持）。"""
    texts = []
    for m in messages:
        if hasattr(m, "type"):  # BaseMessage
            if m.type == "human":
                texts.append((m.content or "").strip())
        else:  # (role, content) tuple
            role, content = m[0], m[1]
            if role == "user":
                texts.append(str(content).strip())
    return texts


def stream_react(query: str, thread_id: str = "t1", max_steps: int = MAX_STEPS):
    """流式执行一轮 ReAct，逐块产出（已渲染的完整轨迹文本）。

    用法：for text in stream_react("1+1等于几"): ...
    最后一块 text 包含最终回答（💡 回答）。
    """
    app = get_app()
    config = {"configurable": {"thread_id": thread_id}}

    # 兜底：对话指代类问题（第一个/上一个/刚才…）直接从历史回答，不消耗模型推理
    kind = _question_kind(query)
    if kind:
        try:
            state = app.get_state(config)
            history_msgs = (state.values or {}).get("messages") or []
            users = _user_texts(history_msgs)
            if kind == "first":
                if users:
                    yield f"💡 **回答**：你问的第一个问题是：“{users[0]}”"
                else:
                    yield f"💡 **回答**：这是我们的第一个问题，你刚问的是：“{query}”"
                return
            if users:
                yield f"💡 **回答**：你上一个问题是：“{users[-1]}”"
                return
        except Exception:
            pass  # 历史不可用时退回正常推理

    rendered, seen = [], set()
    steps = 0
    for chunk in app.stream({"messages": [("user", query)], "steps": 0},
                            config=config, stream_mode="values"):
        msgs = chunk["messages"]
        last_id = id(msgs[-1])
        if last_id in seen:
            continue
        seen.add(last_id)
        steps += 1
        if steps > max_steps:
            rendered.append("⚠️ 达到最大推理步数，已终止。")
            break
        step_text = _render_latest(msgs)
        rendered.append(step_text)
        yield "\n\n".join(rendered)


def run_react(query: str, thread_id: str = "t1") -> str:
    """同步跑完一轮，返回最终回答文本（不含轨迹）。"""
    app = get_app()
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"messages": [("user", query)], "steps": 0}, config=config)
    last = result["messages"][-1]
    text = last.content if hasattr(last, "content") else str(last[1])
    return str(text)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法：python -m agent.agent_react \"你的问题\"")
        print(health_check())
        sys.exit(0)
    question = " ".join(sys.argv[1:])
    print(f"\nQ：{question}\n")
    for piece in stream_react(question):
        print(piece)
        print("---")
