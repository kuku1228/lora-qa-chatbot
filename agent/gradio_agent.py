"""
agent/gradio_agent.py —— Gradio Agent 界面（网页版）

展示 Agent 的 ReAct 推理轨迹（Thought / Action / Observation）与工具调用，
可切换「LangGraph ReAct」与「DeepAgents」两种架构。

运行前提：微调模型已通过 vLLM 部署为 OpenAI 兼容 API（见 serve_vllm.sh / README）
启动：python -m agent.gradio_agent  → 浏览器打开 http://127.0.0.1:7861
"""
import gradio as gr

from agent.agent_deep import stream_deep
from agent.agent_react import health_check, stream_react
from agent.llm import health_check as llm_health


def _agent_stream(question, history, mode):
    """按所选架构流式执行，把轨迹渲染成 assistant 回复。

    history 使用 Gradio 6.x 的 messages 格式：list[dict]，每条为
    {"role": "user"/"assistant", "content": "..."}。
    """
    history = history or []
    thread = "web"
    gen = stream_deep(question, thread_id=thread) if mode == "DeepAgents" else stream_react(question, thread_id=thread)
    acc = ""
    try:
        for piece in gen:
            acc = piece
            new_history = list(history) + [
                {"role": "user", "content": question},
                {"role": "assistant", "content": acc},
            ]
            yield new_history
    except Exception as exc:
        acc = f"⚠️ 执行出错：{exc}"
        yield list(history) + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": acc},
        ]
    # 最后确保渲染出最终回答标记（若流里没有，补充）
    if not acc:
        yield list(history) + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": "（未获得回答，请检查 LLM 后端是否已启动）"},
        ]


def build_demo():
    with gr.Blocks(title="LoRA Agent 中文问答助手（Qwen2-7B + LangGraph + DeepAgents）") as demo:
        gr.Markdown(
            "## 🤖 LoRA 微调智能问答助手 · Agent 升级版\n\n"
            "基座 **Qwen2-7B-Instruct**（LoRA 微调）+ **LangGraph** ReAct 推理 + **工具调用**。\n"
            "支持问题：计算题、时间查询、网页查证、领域问答（展示 Thought → Action → Observation 全轨迹）。"
        )
        mode = gr.Radio(
            ["LangGraph ReAct", "DeepAgents"],
            value="LangGraph ReAct",
            label="Agent 架构",
            info="ReAct：单代理工具循环；DeepAgents：主代理 + 子代理委派 + 文件工具",
        )
        chatbot = gr.Chatbot(height=520, label="对话")
        msg = gr.Textbox(placeholder="输入问题，例如：帮我算一下 (23+67)*45 等于多少？", label="问题")
        clear = gr.Button("清空对话")

        def respond(question, history, arch):
            if not question or not question.strip():
                yield history, ""
                return
            history = history or []
            for new_history in _agent_stream(question.strip(), history, arch):
                yield new_history, ""

        submit = msg.submit(respond, [msg, chatbot, mode], [chatbot, msg])
        clear.click(lambda: ([], ""), None, [chatbot, msg])

        # 启动时健康检查提示
        gr.Markdown(f"> 后端状态：{llm_health()}")

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7861)
