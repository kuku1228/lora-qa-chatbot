"""
gradio_demo.py —— 网页版问答 Demo（在租的 GPU 实例上运行）
效果：浏览器打开一个聊天界面，可连续对话，适合演示录屏 / 简历配图。
运行：python gradio_demo.py  → 浏览器访问 http://127.0.0.1:7860
依赖：pip install gradio transformers peft bitsandbytes accelerate
"""
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ===== 改成你实例上的实际路径 =====
MODEL_DIR   = "models/Qwen2-7B-Instruct"   # 7B 模型权重目录
ADAPTER_DIR = "output/lora-qwen2/final"    # LoRA adapter 目录（魔搭训练的 7B 版）
# ===================================

SYSTEM_PROMPT = "你是专业的中文问答助手。请直接、准确、简洁地回答问题，不要冗长铺垫。"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("加载 Qwen2-7B（4bit）+ LoRA adapter ...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, quantization_config=bnb_config,
    device_map="auto", trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, ADAPTER_DIR)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("模型就绪，启动网页 Demo ...")


def ask(question, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256,
                             do_sample=True, top_p=0.9, temperature=0.7)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True)


gr.ChatInterface(
    ask,
    title="LoRA 中文问答助手（Qwen2-7B-Instruct 微调）",
    description="基于 2 万条中文指令数据 LoRA 微调，回答更简洁、准确、专业。",
).launch(server_name="0.0.0.0", server_port=7860)
