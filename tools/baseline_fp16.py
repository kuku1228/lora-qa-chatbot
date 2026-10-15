#!/usr/bin/env python3
"""
tools/baseline_fp16.py —— 全量微调的显存需求对比

用途：
    算清楚"全量微调 vs 本项目 LoRA + 4bit"在显存上差多少，作为选型依据。

设计说明：
    全量微调需要 fp16 权重 + 同尺寸梯度 + AdamW 的 fp32 一阶/二阶动量，
    合计约 16 字节/参数——Qwen2-7B 量级需要 80GB 以上显存（A100/H100 级别）。
    本项目目标是单张 24GB 消费级卡可训，因此**不做全量微调实跑**，
    而是按标准显存公式解析计算，再与 LoRA + 4bit 的实测峰值对照。

    若你确实有 80GB 级显卡想实测，用 --try-run 会尝试跑若干步并打印
    torch 统计的真实峰值（该开关默认关闭）。

用法（项目根目录，不需要 GPU）：
    python tools/baseline_fp16.py
    python tools/baseline_fp16.py --params 7.62e9 --lora-peak 11.40
    python tools/baseline_fp16.py --try-run --steps 50     # 需 80GB 级显卡
"""

from __future__ import annotations

import argparse
import os
import sys

# 让脚本无论从项目根目录还是 tools/ 下执行，都能 import 到根目录的 config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# config 必须最先导入：内部会设置 HF 镜像环境变量（需早于 transformers）
from config import BASE_MODEL, TRAIN_PATH, MAX_SEQ_LEN, BATCH_SIZE, GRAD_ACCUM

# Qwen2-7B-Instruct 的结构参数（与官方 config 一致）
QWEN2_7B = dict(
    hidden_size=3584, intermediate_size=18944, num_hidden_layers=28,
    num_attention_heads=28, num_key_value_heads=4, head_dim=128,
    vocab_size=152064, tie_word_embeddings=False,
)


def analytic_param_count(cfg: dict) -> tuple[int, dict]:
    """按结构算参数量，返回 (总数, 分项)。"""
    h, inter = cfg["hidden_size"], cfg["intermediate_size"]
    L, nh = cfg["num_hidden_layers"], cfg["num_attention_heads"]
    nkv = cfg.get("num_key_value_heads", nh)
    hd = cfg.get("head_dim") or (h // nh)
    vocab = cfg["vocab_size"]
    q_out, kv_out = nh * hd, nkv * hd

    per_layer = (
        h * q_out + q_out                      # q_proj + bias
        + h * kv_out * 2 + kv_out * 2          # k_proj / v_proj + bias
        + q_out * h                            # o_proj
        + 3 * h * inter + 2 * inter            # gate / up / down
        + 2 * h                                # 两个 RMSNorm
    )
    emb = vocab * h
    lm_head = 0 if cfg.get("tie_word_embeddings", False) else emb
    total = emb + lm_head + L * per_layer
    return total, {"embed_tokens": emb, "lm_head": lm_head,
                   "per_layer": per_layer, "layers": L,
                   "decoder_layers": L * per_layer}


def main() -> int:
    ap = argparse.ArgumentParser(description="全量微调显存需求对比（解析计算）")
    ap.add_argument("--params", type=float, default=None,
                    help="模型参数量，默认按 Qwen2-7B 结构计算")
    ap.add_argument("--lora-peak", type=float, default=11.40,
                    help="本项目 LoRA+4bit 的实测显存峰值 GB（默认 11.40，来自 monitor 采样）")
    ap.add_argument("--try-run", action="store_true",
                    help="尝试实跑全量微调若干步（需 80GB 级显卡）")
    ap.add_argument("--steps", type=int, default=50)
    args = ap.parse_args()

    total, parts = analytic_param_count(QWEN2_7B)
    n = args.params if args.params else total

    print("=" * 68)
    print("全量微调 vs LoRA + 4bit：显存需求对比")
    print("=" * 68)
    print(f"  模型              : {BASE_MODEL}")
    print(f"  参数量            : {n:,.0f}  ({n/1e9:.3f} B)")
    print(f"  结构明细          : embed {parts['embed_tokens']:,} + "
          f"lm_head {parts['lm_head']:,} + {parts['layers']} 层 × {parts['per_layer']:,}")
    print(f"  训练配置          : max_seq_len={MAX_SEQ_LEN}, "
          f"batch={BATCH_SIZE}, grad_accum={GRAD_ACCUM}")
    print()

    # 全量微调（fp16 + AdamW）的显存构成
    # 注：不额外计"fp32 主权重副本"——本项目的基线就是纯 fp16 训练（2_train_lora.py 的
    #     对照路径），adamw_torch 直接在 fp16 参数上维护 fp32 的 m/v 状态。
    items = [
        ("fp16 权重", n * 2 / 1024**3),
        ("fp16 梯度", n * 2 / 1024**3),
        ("AdamW fp32 一阶动量 (m)", n * 4 / 1024**3),
        ("AdamW fp32 二阶动量 (v)", n * 4 / 1024**3),
    ]
    subtotal = sum(v for _, v in items)

    print("─" * 68)
    print("【方案 A】全量微调（fp16 + AdamW，不量化、不冻结）")
    print("─" * 68)
    for k, v in items:
        print(f"  {k:32s} {v:8.1f} GB")
    print(f"  {'小计（不含激活值）':32s} {subtotal:8.1f} GB")
    print(f"  {'激活值（随 batch×seq_len 增长，估）':32s} {subtotal*0.15:8.1f} GB")
    total_a = subtotal * 1.15
    print(f"  {'合计（估算）':32s} {total_a:8.1f} GB")
    print()

    print("─" * 68)
    print("【方案 B】本项目：LoRA + 4bit NF4 + 双重量化 + 梯度检查点 + 8bit paged AdamW")
    print("─" * 68)
    print(f"  {'4bit 量化权重':32s} {n*0.55/1024**3:8.1f} GB")
    print(f"  {'LoRA 梯度与 8bit 优化器状态':32s} {20185088*14/1024**3:8.1f} GB")
    print(f"  {'实测显存峰值（monitor 采样）':32s} {args.lora_peak:8.2f} GB")
    print()

    print("=" * 68)
    print("结论")
    print("=" * 68)
    print(f"  全量微调需求  : 约 {total_a:.0f} GB  → 需 A100/H100 80GB 级显卡")
    print(f"  本项目实测峰值: {args.lora_peak:.2f} GB  → 单张 24GB 消费级卡可训")
    print(f"  显存倍数差    : 约 {total_a/args.lora_peak:.1f} 倍")
    print()
    print("  为什么会差这么多：全量微调要为【每个参数】额外保存梯度与 fp32 优化器状态，")
    print("  合计 16 字节/参数；而 LoRA 冻结主干、只训练 0.265% 的参数，")
    print("  优化器状态与梯度只对这部分开销，因此显存大头被消掉。")
    print("=" * 68)

    if args.try_run:
        print("\n[--try-run] 尝试实跑全量微调以取得真实峰值 ...")
        _try_run(args.steps)
    else:
        print("\n（未实跑全量微调；如需实测请加 --try-run，并确认显卡 ≥80GB）")
    return 0


def _try_run(steps: int) -> int:
    """实跑若干步并打印 torch 统计的峰值（需大显存卡）。"""
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              TrainingArguments, Trainer, DataCollatorForSeq2Seq)
    from datasets import load_dataset

    if not torch.cuda.is_available():
        print("[跳过] 没有可用 GPU")
        return 0
    dev = torch.cuda.get_device_properties(0)
    print(f"GPU: {dev.name}  显存: {dev.total_memory/1024**3:.1f} GB")

    system_prompt = "你是专业的中文问答助手。请直接、准确、简洁地回答问题，不要冗长铺垫。"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def fmt(ex):
        user = ex["instruction"]
        if ex.get("input") and ex["input"].strip():
            user += "\n\n输入：" + ex["input"]
        msgs = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user},
                {"role": "assistant", "content": ex["output"]}]
        return {"text": tokenizer.apply_chat_template(msgs, tokenize=False)}

    def tok(examples):
        t = tokenizer(examples["text"], truncation=True, max_length=MAX_SEQ_LEN, padding=False)
        t["labels"] = t["input_ids"].copy()
        return t

    ds = load_dataset("json", data_files=TRAIN_PATH, split="train")
    ds = ds.map(fmt).map(tok, batched=True, remove_columns=ds.column_names)
    ds = ds.select(range(min(len(ds), 2000)))

    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.config.use_cache = False
    targs = TrainingArguments(
        output_dir="output/_baseline_fp16",
        per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=GRAD_ACCUM,
        max_steps=steps, learning_rate=2e-4, save_strategy="no",
        fp16=True, optim="adamw_torch", report_to="none", remove_unused_columns=False)
    Trainer(model=model, args=targs, train_dataset=ds,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)).train()

    print(f"\n显存峰值 allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
    print(f"显存峰值 reserved : {torch.cuda.max_memory_reserved()/1024**3:.2f} GB")
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
