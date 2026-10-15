#!/usr/bin/env python3
"""
tools/data_stats.py —— 只统计长度分布，不重新下载数据

为什么单独拆出一个脚本：
    完整的 1_data_preprocess.py 会重新下载数据集、重跑清洗和划分，
    但 data/train.jsonl 和 data/eval.jsonl 其实已经有了，没必要重来。
    这个脚本直接读现成的 jsonl，只算缺的那几项统计：

      - 字符长度分布（P50 / P90 / P95 / P99 / Max）
      - 套完 chat 模板后的 token 长度分布（真实 tokenizer）
      - 各候选 max_seq_len 下的截断比例
      - input 空值比例、重复样本、噪声样本等数据质量指标

用法（项目根目录）：
    python tools/data_stats.py
    python tools/data_stats.py --data-dir data --out output/data_stats.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

import numpy as np

# 让脚本无论从项目根目录还是 tools/ 下执行，都能 import 到根目录的 config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# config 必须最先导入：内部会设置 HF 镜像环境变量（需早于 transformers）
from config import BASE_MODEL, MAX_SEQ_LEN, TRAIN_PATH, EVAL_PATH

# 与 2_train_lora.py 的 system prompt 保持一致，否则统计的长度对不上训练时的真实输入
SYSTEM_PROMPT = "你是专业的中文问答助手。请直接、准确、简洁地回答问题，不要冗长铺垫。"
CANDIDATE_LENS = (512, 768, 1024, 2048)


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def char_stats(name: str, arr: list[int]) -> dict:
    a = np.array(arr)
    p50, p90, p95, p99 = np.percentile(a, [50, 90, 95, 99])
    print(f"  {name:22s} 均值 {a.mean():7.1f} | P50 {p50:6.0f} | P90 {p90:6.0f} "
          f"| P95 {p95:6.0f} | P99 {p99:6.0f} | Max {a.max():6.0f}")
    return {"mean": float(a.mean()), "p50": float(p50), "p90": float(p90),
            "p95": float(p95), "p99": float(p99), "max": int(a.max())}


def main() -> int:
    ap = argparse.ArgumentParser(description="长度分布统计（不重新下载数据）")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default=os.path.join("output", "data_stats.json"))
    ap.add_argument("--sample", type=int, default=0,
                    help="token 统计只取前 N 条（0 = 全部，约 2-3 分钟）")
    args = ap.parse_args()

    train_path = TRAIN_PATH if os.path.exists(TRAIN_PATH) else os.path.join(args.data_dir, "train.jsonl")
    eval_path = EVAL_PATH if os.path.exists(EVAL_PATH) else os.path.join(args.data_dir, "eval.jsonl")

    if not os.path.exists(train_path):
        print(f"[中止] 找不到 {train_path}。请先运行 1_data_preprocess.py 生成数据。")
        return 1

    print("=" * 66)
    print("长度分布统计（读取现有 jsonl，不重新下载数据集）")
    print("=" * 66)

    train = load_jsonl(train_path)
    ev = load_jsonl(eval_path) if os.path.exists(eval_path) else []
    print(f"训练集 {len(train)} 条   评估集 {len(ev)} 条   合计 {len(train)+len(ev)} 条\n")

    # ---------- 1. 字符长度 ----------
    print("[1/5] 字符长度分布")
    ins = [len(x["instruction"]) for x in train]
    outs = [len(x["output"]) for x in train]
    tot = [i + o for i, o in zip(ins, outs)]
    char_res = {
        "instruction": char_stats("instruction", ins),
        "output": char_stats("output", outs),
        "instruction+output": char_stats("instruction+output", tot),
    }

    # ---------- 2. token 长度 ----------
    print(f"\n[2/5] token 长度分布（用真实 tokenizer 套 chat 模板）")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sample = train[: args.sample] if args.sample > 0 else train
    if args.sample > 0:
        print(f"      （只统计前 {args.sample} 条）")

    lens = []
    for ex in sample:
        user = ex["instruction"]
        if ex.get("input") and ex["input"].strip():
            user += "\n\n输入：" + ex["input"]
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
                {"role": "assistant", "content": ex["output"]}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False)
        lens.append(len(tokenizer(text, truncation=False)["input_ids"]))

    arr = np.array(lens)
    p50, p90, p95, p99 = np.percentile(arr, [50, 90, 95, 99])
    print(f"  {'套模板后 token':22s} 均值 {arr.mean():7.1f} | P50 {p50:6.0f} | P90 {p90:6.0f} "
          f"| P95 {p95:6.0f} | P99 {p99:6.0f} | Max {arr.max():6.0f}")
    token_res = {"n": len(arr), "mean": float(arr.mean()), "p50": float(p50),
                 "p90": float(p90), "p95": float(p95), "p99": float(p99),
                 "max": int(arr.max())}

    # ---------- 3. 截断比例 ----------
    print(f"\n[3/5] 各候选 max_seq_len 下的截断比例（当前配置 = {MAX_SEQ_LEN}）")
    trunc = {}
    for L in CANDIDATE_LENS:
        n_over = int((arr > L).sum())
        ratio = n_over / len(arr) * 100
        trunc[L] = {"n_over": n_over, "ratio": round(ratio, 3)}
        mark = "  ← 当前 config" if L == MAX_SEQ_LEN else ""
        print(f"      max_seq_len={L:5d} → 截断 {n_over:5d} 条 ({ratio:5.2f}%){mark}")

    cur = trunc.get(MAX_SEQ_LEN)
    if cur:
        if cur["n_over"] == 0:
            print(f"      结论：{MAX_SEQ_LEN} 可完整覆盖全部样本，无截断。")
        else:
            print(f"      结论：{MAX_SEQ_LEN} 会截断 {cur['n_over']} 条（{cur['ratio']:.2f}%）。")

    # ---------- 4. 数据质量 ----------
    print(f"\n[4/5] 数据质量")
    empty_input = sum(1 for x in train if not (x.get("input") or "").strip())
    print(f"  input 为空的样本        : {empty_input}/{len(train)} = {empty_input/len(train)*100:.1f}%")

    key = lambda x: (x["instruction"].strip(), (x.get("input") or "").strip())
    dup_key = [k for k, c in Counter(key(x) for x in train).items() if c > 1]
    print(f"  instruction+input 重复组 : {len(dup_key)} 组")

    all_key = Counter(key(x) for x in train + ev)
    dup_all = [k for k, c in all_key.items() if c > 1]
    print(f"  全量重复组               : {len(dup_all)} 组")

    overlap = len(set(key(x) for x in train) & set(key(x) for x in ev))
    print(f"  训练集/评估集重叠        : {overlap} 组")

    noise = sum(1 for x in train + ev
                if not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", x["output"]))
    print(f"  output 无中英数字符      : {noise} 条")

    over_ins = sum(1 for v in ins if v > 512)
    over_out = sum(1 for v in outs if v > 1024)
    under_out = sum(1 for v in outs if v < 10)
    print(f"  超阈值（instruction>512）: {over_ins} 条")
    print(f"  超阈值（output>1024）    : {over_out} 条")
    print(f"  过短（output<10）        : {under_out} 条")

    quality = {
        "empty_input_ratio": round(empty_input / len(train) * 100, 2),
        "dup_train": len(dup_key), "dup_all": len(dup_all), "overlap": overlap,
        "noise": noise, "over_instruction": over_ins,
        "over_output": over_out, "under_output": under_out,
    }

    # ---------- 5. 落盘 ----------
    result = {
        "n_train": len(train), "n_eval": len(ev),
        "char": char_res, "token": token_res,
        "max_seq_len": int(MAX_SEQ_LEN), "truncation": trunc,
        "quality": quality,
        "model_used_for_tokenizer": BASE_MODEL,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n[5/5] 已保存：{args.out}")
    print("      → 这些数字直接填进 docs/结果记录模板.md 的第 1 节。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
