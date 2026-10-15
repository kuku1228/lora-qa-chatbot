"""
4_infer_demo.py —— 本地问答 Demo
加载 4bit 基座 + LoRA adapter，命令行交互问答。
运行：python 4_infer_demo.py
输入 exit / quit / 退出 结束。
"""
# config 必须最先导入：内部会设置 HF 镜像环境变量（需早于 transformers/datasets）
from config import BASE_MODEL, ADAPTER_DIR

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

SYSTEM_PROMPT = "你是专业的中文问答助手。请直接、准确、简洁地回答问题，不要冗长铺垫。"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


def main():
    print("加载 4bit 基座模型 + LoRA adapter ...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("模型就绪！输入问题开始问答（输入 exit 退出）\n")
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
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256,
                                 do_sample=True, top_p=0.9, temperature=0.7)
        ans = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                               skip_special_tokens=True)
        print(f"A: {ans}\n")


if __name__ == "__main__":
    main()
