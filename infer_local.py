"""
infer_local.py —— 本地 CPU 推理版（Qwen2-1.5B + LoRA）
适用：无 NVIDIA 显卡、16GB 内存的电脑；回答约 3~10 秒。
运行：python infer_local.py（输入 exit 退出）
前置：
  1. 本地有模型目录 models/Qwen2-1.5B-Instruct（从魔搭下载后放项目下）
  2. 本地有 adapter 目录 output_1_5b/final（从魔搭下载）
  3. 已安装依赖：pip install transformers peft accelerate
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ===== 改成你本地实际路径 =====
BASE_MODEL  = "models/Qwen2-1.5B-Instruct"   # 1.5B 模型权重目录
ADAPTER_DIR = "output_1_5b/final"            # LoRA adapter 目录
# ==============================

SYSTEM_PROMPT = "你是专业的中文问答助手。请直接、准确、简洁地回答问题，不要冗长铺垫。"


def main():
    print("加载 Qwen2-1.5B（fp32，CPU，约需 6GB 内存）...")
    # CPU 推理用 fp32 最稳（CPU 原生支持，比 fp16 更快）
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("微调模型就绪！输入问题开始问答（输入 exit 退出）\n")
    while True:
        q = input("Q: ").strip()
        if q.lower() in ("exit", "quit", "退出"):
            break
        if not q:
            continue
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": q}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256,
                                 do_sample=True, top_p=0.9, temperature=0.7)
        ans = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                               skip_special_tokens=True)
        print(f"A: {ans}\n")


if __name__ == "__main__":
    main()
