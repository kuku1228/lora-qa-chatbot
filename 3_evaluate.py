"""
3_evaluate.py —— 微调前后效果对比
流程：加载原始模型 + 加载 LoRA 模型 → 同一评估集算 PPL → 同一批问题生成对比
运行：python 3_evaluate.py
输出：
  - 终端打印 PPL 对比
  - output/comparison.md：同一问题的原始 vs 微调回答对比（可直接进 README）
"""
import json
import os

# config 必须最先导入：内部会设置 HF 镜像环境变量（需早于 transformers/datasets）
from config import BASE_MODEL, EVAL_PATH, ADAPTER_DIR, MAX_SEQ_LEN

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

SYSTEM_PROMPT = "你是专业的中文问答助手。请直接、准确、简洁地回答问题，不要冗长铺垫。"

# 对比问题集：覆盖"是什么/为什么/怎么用/区别/方案"五类，便于展示风格差异
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
    """加载基座；adapter_path 非空时叠加 LoRA"""
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model


def generate(model, tokenizer, question, max_new=200):
    """生成回答：只 decode 新增部分，跳过模板与特殊符号"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new,
                             do_sample=False)   # 贪心：可复现、方便对比
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True)


def compute_ppl(model, tokenizer, texts):
    """
    困惑度 Perplexity：衡量模型对文本的"意外程度"，越低越好。
    PPL = exp( 平均负对数似然 )，是评估微调是否适配领域的最直接指标。
    """
    model.eval()
    total_nll, total_tokens = 0.0, 0
    with torch.no_grad():
        for t in texts:
            enc = tokenizer(t, return_tensors="pt",
                            truncation=True, max_length=MAX_SEQ_LEN)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            out = model(**enc, labels=enc["input_ids"])   # labels=input_ids → 自回归 loss
            loss = out.loss.item()                         # 平均 NLL（已对 token 归一）
            n_tokens = (enc["input_ids"] != tokenizer.pad_token_id).sum().item()
            total_nll += loss * n_tokens
            total_tokens += n_tokens
    return float(torch.exp(torch.tensor(total_nll / total_tokens)))


def build_eval_texts(tokenizer, n=100):
    """把 eval 集前 n 条拼成模板文本，作为 PPL 评估语料"""
    texts = []
    with open(EVAL_PATH, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            ex = json.loads(line)
            user = ex["instruction"]
            if ex.get("input", "").strip():
                user += "\n\n输入：" + ex["input"]
            msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": ex["output"]}]
            texts.append(tokenizer.apply_chat_template(msgs, tokenize=False))
    return texts


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. PPL 对比
    print("[1/3] 计算原始模型 PPL ...")
    base_model = load_model()
    texts = build_eval_texts(tokenizer, n=100)
    ppl_base = compute_ppl(base_model, tokenizer, texts)

    print("[2/3] 计算 LoRA 微调模型 PPL ...")
    lora_model = load_model(ADAPTER_DIR)
    ppl_lora = compute_ppl(lora_model, tokenizer, texts)

    print("\n======== PPL 对比（评估集 100 条）========")
    print(f"原始 Qwen2-7B-Instruct : {ppl_base:.4f}")
    print(f"LoRA 微调后            : {ppl_lora:.4f}")
    print(f"下降幅度                : {(1 - ppl_lora / ppl_base) * 100:.1f}%")

    # 2. 同一问题生成对比
    print("\n[3/3] 同一问题生成效果对比 ...")
    md = ["# 微调前后效果对比\n",
          f"模型：{BASE_MODEL} + LoRA(r={8}) | 数据：2 万条中文指令\n", ""]
    for q in QUESTIONS:
        a1 = generate(base_model, tokenizer, q)
        a2 = generate(lora_model, tokenizer, q)
        md.append(f"**Q：{q}**\n")
        md.append(f"- **原始模型**：{a1}\n")
        md.append(f"- **LoRA 微调后**：{a2}\n")
        print(f"\nQ：{q}")
        print(f"  原始：{a1[:80]}...")
        print(f"  微调：{a2[:80]}...")

    os.makedirs("output", exist_ok=True)
    with open(os.path.join("output", "comparison.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n对比结果已保存：output/comparison.md")
    print("（人工打分：把两组回答打印出来请同学/自己盲评，统计“微调回答更优”的占比）")


if __name__ == "__main__":
    main()
