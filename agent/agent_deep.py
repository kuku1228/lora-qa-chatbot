"""
agent/agent_deep.py —— DeepAgents 主代理（LangChain 官方 Deep Agents 框架）

与 LangGraph ReAct 的区别：
  - ReAct：单代理 + 工具调用循环（本仓库 agent_react.py）
  - DeepAgents：主代理 + 任务拆解 + 子代理委派（subagents）+ 文件系统工具 + 规划

DeepAgents 构建在 LangGraph 之上，可用同一微调模型。默认自带：
  - 文件系统工具（write_file / read_file，默认 StateBackend 内存暂存）
  - 子代理能力（复杂子任务自动委派）
  - 长上下文管理（内容卸载/记忆）

运行前提：微调模型已通过 vLLM 部署为 OpenAI 兼容 API（见 serve_vllm.sh）
"""
from langgraph.checkpoint.memory import MemorySaver

from agent.llm import build_chat_model
from agent.tools import build_toolset

DEEP_SYSTEM_PROMPT = """你是基于 LoRA 微调的中文智能问答助手（Deep Agent 版），基座模型 Qwen2-7B-Instruct。
你擅长把复杂任务拆解执行，并善用工具：

可用工具：
- calculator：数学表达式安全求值
- get_current_time：获取当前日期时间
- fetch_url：抓取网页内容查证外部信息
- write_file / read_file：暂存与整理长文本材料（默认保存在对话状态中，不会写磁盘）

执行策略：
1. 先理解任务，需要时先调用工具获取事实，再综合回答；
2. 复杂子任务可拆解并委派子代理处理；
3. 面对需要多轮查证的任务（研究类），先收集材料再输出结论；
4. 最终回答保持简洁、准确、专业（一般不超过 6 句话）。"""

_DEEP_APP = None


def build_deep_agent(model=None, tools=None, system_prompt=DEEP_SYSTEM_PROMPT, subagents=None):
    """构建 DeepAgents 主代理。subagents 可传自定义子代理列表（默认使用内置能力）。"""
    from deepagents import create_deep_agent

    model = model or build_chat_model()
    tools = tools if tools is not None else build_toolset()
    return create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        subagents=subagents or [],
        checkpointer=MemorySaver(),
    )


def get_deep_app():
    global _DEEP_APP
    if _DEEP_APP is None:
        _DEEP_APP = build_deep_agent()
    return _DEEP_APP


def stream_deep(query: str, thread_id: str = "d1", max_steps: int = 12):
    """流式执行一轮 Deep Agent，逐块产出渲染后的轨迹文本。"""
    app = get_deep_app()
    config = {"configurable": {"thread_id": thread_id}}
    rendered, seen = [], set()
    steps = 0
    for chunk in app.stream({"messages": [("user", query)]},
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
        # 复用 ReAct 的渲染逻辑（同为 AIMessage/ToolMessage 结构）
        from agent.agent_react import _render_latest
        rendered.append(_render_latest(msgs))
        yield "\n\n".join(rendered)


def run_deep(query: str, thread_id: str = "d1") -> str:
    """同步跑完一轮，返回最终回答。"""
    app = get_deep_app()
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"messages": [("user", query)]}, config=config)
    return str(result["messages"][-1].content)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法：python -m agent.agent_deep \"你的问题\"")
        sys.exit(0)
    question = " ".join(sys.argv[1:])
    print(f"\nQ：{question}\n")
    for piece in stream_deep(question):
        print(piece)
        print("---")
