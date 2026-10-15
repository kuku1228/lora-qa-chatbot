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

    # 4. 长度分布统计（对 2 万条数据做长度分布分析）
    ins_lens = [len(c["instruction"]) for c in cleaned]
    out_lens = [len(c["output"]) for c in cleaned]
    print(f"[3/6] 长度统计：")
    print(f"      instruction  均值 {np.mean(ins_lens):.1f} | 中位数 {np.median(ins_lens):.0f} | 最长 {max(ins_lens)}")
    print(f"      output       均值 {np.mean(out_lens):.1f} | 中位数 {np.median(out_lens):.0f} | 最长 {max(out_lens)}")

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
