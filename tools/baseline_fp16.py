#!/usr/bin/env python3
"""
tools/baseline_fp16.py —— 全量 fp16 微调显存对照（只为拿显存数字）

为什么需要它：
    README 里"较全量 fp16 微调显存下降约 70%"是个估算，没有实测。
    这个脚本不训练到头，只跑 50 步就退出，目的是测出 fp16 全量微调的显存峰值，
    用来和 LoRA + 4bit 的实测峰值做对照。

⚠️ 显存要求高：
    Qwen2-7B 全量 fp16 训练需要 权重 14GB + 梯度 14GB + AdamW 优化器状态 56GB
    ≈ 84GB 以上，单张 24GB 卡**必然 OOM**。所以：
      - 只有 80GB 卡（A100/H100）能真正跑起来；
      - 24GB 卡上跑这个脚本，OOM 本身就是结论：说明全量微调在本机不可行，
        这时可以直接用"理论值 84GB+ vs 实测约 12GB"来表述，并注明理论值来源。
    脚本会把 OOM 也如实记录成实验结果，不视为失败。

用法（在项目根目录）：
    # 终端 A：先开始采样
    python tools/monitor.py --out logs/baseline_fp16_mem.csv --interval 5
    # 终端 B：再跑对照（50 步后自动退出）
    python tools/baseline_fp16.py --steps 50
"""

from __future__ import annotations

import argparse
import os
import sys

# 让脚本无论从项目根目录还是 tools/ 下执行，都能 import 到根目录的 config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# config 必须最先导入：内部会设置 HF 镜像环境变量（需早于 transformers）
from config import (BASE_MODEL, TRAIN_PATH, MAX_SEQ_LEN,
                    BATCH_SIZE, GRAD_ACCUM, LEARNING_RATE, WARMUP_RATIO, LOGGING_STEPS)

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset


SYSTEM_PROMPT = "你是专业的中文问答助手。请直接、准确、简洁地回答问题，不要冗长铺垫。"


def main() -> int:
    ap = argparse.ArgumentParser(description="全量 fp16 微调显存对照")
    ap.add_argument("--steps", type=int, default=50, help="跑多少步后停止")
    ap.add_argument("--bf16", action="store_true",
                    help="用 bf16 而非 fp16（显存占用相同，数值更稳）")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[中止] 没有检测到可用 GPU。")
        return 1

    dev = torch.cuda.get_device_properties(0)
    print("=" * 64)
    print(f"GPU: {dev.name}   显存: {dev.total_memory/1024**3:.1f} GB")
    print(f"对照配置: 全量微调（不量化、不冻结）+ {'bf16' if args.bf16 else 'fp16'}")
    print(f"          batch={BATCH_SIZE} x grad_accum={GRAD_ACCUM} x max_len={MAX_SEQ_LEN}")
    print("=" * 64)

    # 理论显存估算（供报告引用）
    n_params = 7.62e9
    bytes_per = 2
    theory = {
        "权重 fp16": n_params * bytes_per / 1024**3,
        "梯度 fp16": n_params * bytes_per / 1024**3,
        "AdamW 状态 fp32 (m+v)": n_params * 4 * 2 / 1024**3,
    }
    print("\n[理论显存下限]")
    for k, v in theory.items():
        print(f"  {k:24s} {v:6.1f} GB")
    print(f"  {'合计（不含激活值）':24s} {sum(theory.values()):6.1f} GB")
    print("  → 单张 24GB 卡必然 OOM；这一条本身就是有价值的结论。\n")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def format_example(ex):
        user = ex["instruction"]
        if ex.get("input") and ex["input"].strip():
            user += "\n\n输入：" + ex["input"]
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
                {"role": "assistant", "content": ex["output"]}]
        return {"text": tokenizer.apply_chat_template(msgs, tokenize=False)}

    def tokenize_fn(examples):
        tok = tokenizer(examples["text"], truncation=True,
                        max_length=MAX_SEQ_LEN, padding=False)
        tok["labels"] = tok["input_ids"].copy()
        return tok

    print("加载并格式化训练数据 ...")
    ds = load_dataset("json", data_files=TRAIN_PATH, split="train")
    ds = ds.map(format_example)
    ds = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)
    ds = ds.select(range(min(len(ds), 2000)))   # 对照实验不需要全量
    print(f"对照组样本数：{len(ds)}（只取前 2000 条用于测显存）\n")

    torch.cuda.reset_peak_memory_stats()
    try:
        print("加载全量模型（无量化、无 LoRA）...")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            device_map="auto", trust_remote_code=True,
        )
        model.config.use_cache = False
        total = sum(p.numel() for p in model.parameters())
        print(f"模型加载完成，参数量 {total/1e8:.1f} 亿，可训练参数 = 全部参数（全量微调）")
        print(f"加载后已用显存: {torch.cuda.memory_allocated()/1024**3:.2f} GB\n")

        targs = TrainingArguments(
            output_dir="output/_baseline_fp16",
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            max_steps=args.steps,
            learning_rate=LEARNING_RATE,
            lr_scheduler_type="cosine",
            warmup_ratio=WARMUP_RATIO,
            logging_steps=LOGGING_STEPS,
            save_strategy="no",
            fp16=not args.bf16,
            bf16=args.bf16,
            optim="adamw_torch",       # 全量微调的标准优化器，不用 8bit 分页版
            report_to="none",
            remove_unused_columns=False,
        )
        trainer = Trainer(model=model, args=targs, train_dataset=ds,
                          data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True))
        trainer.train()
        peak = torch.cuda.max_memory_allocated() / 1024**3
        reserved = torch.cuda.max_memory_reserved() / 1024**3
        print("\n" + "=" * 64)
        print(f"[对照组结果] 全量 {'bf16' if args.bf16 else 'fp16'} 微调跑通 {args.steps} 步")
        print(f"  显存峰值（allocated）: {peak:.2f} GB")
        print(f"  显存峰值（reserved） : {reserved:.2f} GB")
        print("=" * 64)
    except torch.cuda.OutOfMemoryError as exc:
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print("\n" + "=" * 64)
        print("[对照组结果] 全量微调在本卡 OOM —— 这本身就是结论")
        print(f"  卡显存上限 : {total_mem:.1f} GB")
        print(f"  理论下限   : {sum(theory.values()):.1f} GB（不含激活值）")
        print(f"  报错信息   : {str(exc)[:200]}")
        print("  表述建议   : 「全量 fp16 微调理论显存需求 84GB 以上，24GB 单卡直接 OOM；")
        print("                本项目的 LoRA + 4bit 方案实测峰值 ___GB，因此单卡可训」")
        print("=" * 64)
    finally:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
