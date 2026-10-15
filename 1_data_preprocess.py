"""
1_data_preprocess.py —— 数据预处理
流程：下载数据集 → 采样 2 万条 → 清洗 → 长度分布统计+可视化 → 划分 train/eval
运行：python 1_data_preprocess.py
首次运行会自动联网下载数据集（约几十 MB，之后走本地缓存）
"""
import json
import re
import random
import os

# config 必须最先导入：内部会设置 HF 镜像环境变量（需早于 datasets/huggingface_hub）
from config import (DATA_REPO, SAMPLE_NUM, EVAL_NUM,
                    TRAIN_PATH, EVAL_PATH,
                    MAX_INSTRUCTION_LEN, MAX_OUTPUT_LEN, MIN_OUTPUT_LEN)

import numpy as np
import matplotlib
matplotlib.use("Agg")          # 无界面后端，纯保存图片
import matplotlib.pyplot as plt
from datasets import load_dataset


def clean_example(ex):
    """单条清洗规则：不合格返回 None，合格返回规范化后的 dict"""
    instruction = (ex.get("instruction") or "").strip()
    output      = (ex.get("output") or "").strip()
    input_      = (ex.get("input") or "").strip()

    # 1) 空值过滤：没有指令或没有回答的样本直接丢弃
    if not instruction or not output:
        return None
    # 2) 过短过滤：回答太短说明没有实质内容（“好的”“不知道”之类）
    if len(output) < MIN_OUTPUT_LEN:
        return None
    # 3) 超长过滤：超长样本会让单条序列超过 max_seq_len，被截断后学习价值低
    if len(instruction) > MAX_INSTRUCTION_LEN or len(output) > MAX_OUTPUT_LEN:
        return None
    # 4) 噪声过滤：回答里没有中英文和数字，全是标点/空格/换行 → 视为脏数据
    if not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", output):
        return None

    return {"instruction": instruction, "input": input_, "output": output}


# ---------- token 长度统计 ----------
# 与 2_train_lora.py 用的 system prompt 保持一致，否则统计出来的长度对不上训练时的实际输入
SYSTEM_PROMPT = "你是专业的中文问答助手。请直接、准确、简洁地回答问题，不要冗长铺垫。"

# 统计截断比例时考察的候选长度
CANDIDATE_LENS = (512, 768, 1024, 2048)


def compute_token_stats(cleaned, max_seq_len=None):
    """用真实 tokenizer 统计"套完 chat 模板后的 token 长度"。

    为什么必须用 token 口径：
        清洗阶段的 512/1024 是【字符】阈值，而 max_seq_len 是【token】上限。
        中文常见 1 字 ≈ 1 token，再叠加 chat 模板本身的开销，字符没超阈值的
        样本仍可能被 tokenizer 截断——只看字符分布会低估真实长度。

    返回 dict，包含各分位数与在各候选长度下的截断比例。
    """
    from config import BASE_MODEL, MAX_SEQ_LEN as CFG_MAX_SEQ_LEN
    from transformers import AutoTokenizer

    if max_seq_len is None:
        max_seq_len = CFG_MAX_SEQ_LEN

    print("[3.1/6] 加载 tokenizer 统计 token 长度（首次会下载 tokenizer，仅几十 MB）...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    lens = []
    for ex in cleaned:
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
    print(f"      套模板后 token  {('均值 %7.1f' % arr.mean())} | P50 {p50:5.0f} | P90 {p90:5.0f} "
          f"| P95 {p95:5.0f} | P99 {p99:5.0f} | Max {arr.max():5.0f}")

    print("      各候选 max_seq_len 下的截断比例：")
    trunc = {}
    for L in CANDIDATE_LENS:
        n_over = int((arr > L).sum())
        ratio = n_over / len(arr) * 100
        trunc[L] = {"n_over": n_over, "ratio": ratio}
        mark = "  ← 当前 config 配置" if L == max_seq_len else ""
        print(f"        max_seq_len={L:5d} → 截断 {n_over:5d} 条 ({ratio:5.2f}%){mark}")

    cur = trunc.get(max_seq_len)
    if cur and cur["n_over"] == 0:
        print(f"      结论：当前 max_seq_len={max_seq_len} 可以完整覆盖全部样本，无截断。")
    elif cur:
        print(f"      结论：当前 max_seq_len={max_seq_len} 会截断 {cur['n_over']} 条"
              f"（{cur['ratio']:.2f}%），可考虑上调或过滤这些长样本。")

    stats = {
        "n": len(arr),
        "mean": float(arr.mean()),
        "p50": float(p50), "p90": float(p90), "p95": float(p95), "p99": float(p99),
        "max": int(arr.max()),
        "max_seq_len": int(max_seq_len),
        "truncation": trunc,
    }
    os.makedirs("output", exist_ok=True)
    with open(os.path.join("output", "data_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print("      统计结果已保存：output/data_stats.json（可直接引用到 README）")
    return stats


def main():
    # 确保输出目录存在（data/ 存放分布图与 train/eval jsonl）
    os.makedirs("data", exist_ok=True)

    # 1. 下载数据集（已缓存则直接读缓存）
    print("[1/6] 加载数据集 ...")
    ds = load_dataset(DATA_REPO, split="train")

    # 2. 打乱后采样 2 万条（固定种子保证可复现）
    ds = ds.shuffle(seed=42).select(range(min(SAMPLE_NUM, len(ds))))
    print(f"      采样完成：{len(ds)} 条")

    # 3. 逐条清洗
    cleaned = [c for c in (clean_example(x) for x in ds) if c is not None]
    print(f"[2/6] 清洗完成：{len(ds)} 条 → {len(cleaned)} 条（丢弃 {len(ds) - len(cleaned)} 条）")

    # 4. 长度分布统计（分位数 + 套 chat 模板后的 token 长度）
    ins_lens = [len(c["instruction"]) for c in cleaned]
    out_lens = [len(c["output"]) for c in cleaned]
    print("[3/6] 长度分布统计（字符）：")
    for label, arr in (("instruction", ins_lens), ("output", out_lens)):
        p50, p90, p95, p99 = np.percentile(arr, [50, 90, 95, 99])
        print(f"      {label:12s} 均值 {np.mean(arr):7.1f} | P50 {p50:5.0f} | P90 {p90:5.0f} "
              f"| P95 {p95:5.0f} | P99 {p99:5.0f} | Max {max(arr):5.0f}")

    # 4.1 token 长度分布：这才是决定 max_seq_len 的口径
    #     字符长度 ≠ token 长度（中文分词 + chat 模板都有额外开销），
    #     所以必须用真实 tokenizer 套完模板后再统计，否则阈值定不准。
    token_stats = compute_token_stats(cleaned)

    # 5. 画分布图（保存到 data/length_distribution.png）
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(ins_lens, bins=50, color="#4C9BE8")
    axes[0].set_title("Instruction Length Distribution")
    axes[0].set_xlabel("chars")
    axes[0].set_ylabel("count")
    axes[1].hist(out_lens, bins=50, color="#E87A90")
    axes[1].set_title("Output Length Distribution")
    axes[1].set_xlabel("chars")
    axes[1].set_ylabel("count")
    plt.tight_layout()
    plt.savefig("data/length_distribution.png", dpi=150)
    print("[4/6] 分布图已保存：data/length_distribution.png")

    # 6. 划分 train / eval 并存成 jsonl（每行一个 JSON）
    random.Random(42).shuffle(cleaned)
    train = cleaned[: len(cleaned) - EVAL_NUM]
    eval_ = cleaned[-EVAL_NUM:]
    with open(TRAIN_PATH, "w", encoding="utf-8") as f:
        for c in train:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(EVAL_PATH, "w", encoding="utf-8") as f:
        for c in eval_:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"[5/6] 训练集 {len(train)} 条 → {TRAIN_PATH}")
    print(f"[6/6] 评估集 {len(eval_)} 条 → {EVAL_PATH}")
    print("\n完成！下一步运行：python 2_train_lora.py")


if __name__ == "__main__":
    main()
