"""
ablation.py —— 消融对比：隔离 system prompt 与 LoRA 微调各自的作用
用于回答面试核心问题："微调到底起了多大作用？"

4 个组合 × 6 题生成对比回答，输出到 output/ablation.md：
  A = 原始模型（无 system prompt）     ← 基线
  B = 原始模型 + system prompt        ← 检验 prompt 单独的作用
  C = LoRA 微调模型（无 system prompt）← 检验微调单独的作用（最关键）
  D = LoRA 微调模型 + system prompt   ← 最终效果

运行：python ablation.py （需要 GPU，≥12GB 显存；复用 config 的镜像兜底）
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from config import BASE_MODEL, ADAPTER_DIR

SYSTEM_PROMPT = "你是专业的中文问答助手。请直接、准确、简洁地回答问题，不要冗长铺垫。"
QUESTIONS = [
    "什么是机器学习中的过拟合？",
    "解释一下反向传播算法的原理。",
    "Python 中列表和元组的区别是什么？",
    "什么是数据库索引？它有什么优缺点？",
    "如何缓解梯度消失问题？",
    "简单解释一下 LoRA 微调的原理。",
]

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


def load_model(adapter_path=None):
    print(f"加载模型：{BASE_MODEL}" + (f" + LoRA({adapter_path})" if adapter_path else ""))
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model


def generate(model, tokenizer, question, use_sp):
    """生成回答；use_sp 控制是否注入 system prompt（贪心解码，可复现）"""
    messages = ([{"role": "system", "content": SYSTEM_PROMPT}] if use_sp else []) + \
               [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = load_model()                      # 原始 Qwen2-7B-Instruct
    lora_model = load_model(ADAPTER_DIR)           # LoRA 微调模型

    rows = []
    for i, q in enumerate(QUESTIONS, 1):
        print(f"\n[{i}/{len(QUESTIONS)}] Q：{q}")
        a = generate(base_model, tokenizer, q, False)
        b = generate(base_model, tokenizer, q, True)
        c = generate(lora_model, tokenizer, q, False)
        d = generate(lora_model, tokenizer, q, True)
        rows.append({"Q": q, "A_base_no_sp": a, "B_base_sp": b,
                     "C_lora_no_sp": c, "D_lora_sp": d})
        print(f"  A(原始,无prompt): {a[:60]}...")
        print(f"  B(原始,+prompt) : {b[:60]}...")
        print(f"  C(微调,无prompt): {c[:60]}...")
        print(f"  D(微调,+prompt) : {d[:60]}...")

    os.makedirs("output", exist_ok=True)
    with open(os.path.join("output", "ablation.md"), "w", encoding="utf-8") as f:
        f.write("# 消融对比：system prompt × LoRA 微调\n\n")
        f.write(f"模型：{BASE_MODEL} + LoRA(r=8) | 贪心解码 | 4 组合 × {len(QUESTIONS)} 题\n\n")
        labels = {
            "A_base_no_sp": "A · 原始模型（无 prompt）",
            "B_base_sp": "B · 原始模型 + system prompt",
            "C_lora_no_sp": "C · LoRA 微调（无 prompt）",
            "D_lora_sp": "D · LoRA 微调 + system prompt",
        }
        for r in rows:
            f.write(f"**Q：{r['Q']}**\n\n")
            for k in ["A_base_no_sp", "B_base_sp", "C_lora_no_sp", "D_lora_sp"]:
                f.write(f"- **{labels[k]}**：{r[k]}\n\n")
    print("\n完成！结果已保存到 output/ablation.md")
    print("下一步：把 output/ablation.md 发回给豆包，由它完成人工打分与结论整理")


if __name__ == "__main__":
    main()
