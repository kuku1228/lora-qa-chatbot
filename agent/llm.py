"""
agent/llm.py —— LLM 后端工厂（可切换多后端）

生产路径（推荐）：AutoDL 上用 vLLM 把"微调后的 Qwen2-7B + LoRA"部署成
OpenAI 兼容 API，LangChain 通过 ChatOpenAI 接入 —— 这是工业界标准架构。

环境变量：
    LLM_BASE_URL  默认 http://127.0.0.1:8000/v1 （本机 vLLM）
    LLM_MODEL     默认 lora1（vLLM 的 --lora-modules 模块名，必须与 vLLM 启动参数一致；
                  注意不能与 --served-model-name 同名，否则 vLLM 会命中 base 模型导致 LoRA 不生效）
    LLM_API_KEY   默认 EMPTY（本地 vLLM 不需要 key）
"""
import os

from langchain_openai import ChatOpenAI

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "lora1"
DEFAULT_API_KEY = "EMPTY"


def build_chat_model(temperature: float = 0.0, **kwargs) -> ChatOpenAI:
    """构建接入微调模型的 ChatOpenAI 客户端。"""
    base_url = os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL)
    model = os.environ.get("LLM_MODEL", DEFAULT_MODEL)
    api_key = os.environ.get("LLM_API_KEY", DEFAULT_API_KEY)
    return ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=temperature,
        **kwargs,
    )


def health_check() -> str:
    """检查 LLM 后端是否可达（用于 Demo 启动提示）。"""
    try:
        model = build_chat_model()
        resp = model.invoke("你好，请回复两个字：正常")
        return f"LLM 后端已连接：{resp.content[:80]}"
    except Exception as exc:
        return f"LLM 后端未连接：{exc}（请先运行 serve_vllm.sh 启动 vLLM，或设置 LLM_BASE_URL）"
