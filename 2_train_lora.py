"""
2_train_lora.py —— LoRA 微调主脚本
流程：4bit 量化加载基座 → 冻结主干+梯度检查点 → 注入 LoRA → 训练 → 保存 adapter + loss 曲线
运行：python 2_train_lora.py
显存参考：Qwen2-7B + 4bit + LoRA(r=8) + batch=1 + 梯度累积 ≈ 12GB（3060 可跑，T4 更稳）
"""
import os

# config 必须最先导入：内部会设置 HF 镜像环境变量（需早于 transformers/datasets）
from config import (BASE_MODEL, TRAIN_PATH,
                    LOAD_IN_4BIT, QUANT_TYPE, DOUBLE_QUANT, MAX_SEQ_LEN,
                    LORA_R, LORA_ALPHA, LORA_DROPOUT, TARGET_MODULES,
                    LEARNING_RATE, NUM_EPOCHS, BATCH_SIZE, GRAD_ACCUM,
                    WARMUP_RATIO, LOGGING_STEPS, OUTPUT_DIR, ADAPTER_DIR)

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, TrainingArguments, Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset


# ---------- 1. 4bit 量化配置 ----------
# 4bit 把每个权重压到 4 位（0.5 字节），7B 模型权重 ≈ 3.5GB，
# 相比 fp16 的 14GB，显存直降约 70%+，这就是"训练显存开销降低 70%"的来源
bnb_config = BitsAndBytesConfig(
    load_in_4bit=LOAD_IN_4BIT,
    bnb_4bit_quant_type=QUANT_TYPE,          # nf4 比 fp4 精度更好
    bnb_4bit_compute_dtype=torch.bfloat16,   # 计算仍走 bf16，保证训练精度
    bnb_4bit_use_double_quant=DOUBLE_QUANT,  # 二次量化，再省约 0.4 字节/参数
)

# ---------- 2. 加载基座模型与 tokenizer ----------
print("加载基座模型（4bit 量化）...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",           # 自动分配到可用 GPU
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
# Qwen2 默认没有 pad_token，训练 padding 需要它，用 eos 代替（不会参与学习）
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ---------- 3. 冻结主干 + 梯度检查点 ----------
# prepare_model_for_kbit_training：冻结原始权重、把输入层转成计算 dtype（省显存）
model = prepare_model_for_kbit_training(model)
# 必须关闭 cache，否则与 gradient checkpointing 冲突报错
model.config.use_cache = False
print("已冻结主干参数，开启梯度检查点")

# ---------- 4. 注入 LoRA ----------
# LoRA 核心 = 冻结 W，训练低秩分解 W + B·A（B∈R^{d×r}, A∈R^{r×k}, r<<min(d,k)）
# 训练后权重 = W0 + (alpha/r)·B·A，几乎不增加推理耗时
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",                 # 不训练偏置
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)


# ---------- 4.1 精确统计可训练参数量 ----------
def print_exact_trainable(model):
    """逐参数统计可训练量，并按目标模块汇总。

    为什么不直接用 PEFT 的 print_trainable_parameters()：
        它给的是估算值，而"可训练参数占比"是会被追问的数字，必须能按
        Qwen2 的真实维度复算。这里直接遍历参数张量求和，结果是精确值，
        并且能列出每个目标层各自贡献多少。
    """
    total = _true_param_count(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    per_module = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        module = next((t for t in TARGET_MODULES if f".{t}." in name), "其他")
        per_module[module] = per_module.get(module, 0) + p.numel()

    print("\n" + "=" * 62)
    print("可训练参数统计（精确值）")
    print("=" * 62)
    print(f"  模型总参数   : {total:,}  ({total/1e9:.3f} B)   ← 按模型结构计算")
    print(f"  可训练参数   : {trainable:,}  ({trainable/1e4:.0f} 万)")
    print(f"  可训练占比   : {trainable/total*100:.4f}%")
    print(f"  冻结参数     : {total-trainable:,}")
    if per_module:
        n_layers = _count_layers(model)
        print("-" * 62)
        print("  各目标模块可训练参数明细：")
        for m in sorted(per_module, key=lambda x: -per_module[x]):
            print(f"    {m:12s} {per_module[m]:>12,}")
        print(f"    {'单层合计':12s} {sum(per_module.values())/max(1, n_layers):>12,.0f}"
              f"   （共 {n_layers} 层）")
    print("=" * 62)
    print("→ 记下『可训练参数』与『可训练占比』，用于替换文档中的估算值。\n")
    return total, trainable


def _true_param_count(model) -> int:
    """按模型结构计算真实参数量（不受 4bit 量化影响）。

    为什么不能用 sum(p.numel())：
        4bit 量化后，量化层返回的是打包后的低比特张量，numel() 不再等于
        逻辑参数量。实测会把 7.62B 的模型算成 4.37B，导致占比偏大 1.74 倍
        （0.4616% 而非正确的 0.265%）。

    实现要点：
        1. 用某个真正的 nn.Linear 的权重张量推断 dtype，把量化张量的 numel
           还原成逻辑参数量（weight.numel() * (packed_numel / logical_elem_count)）；
        2. 输出头是否与词嵌入共享【按张量名判断】，不靠 config 里
           tie_word_embeddings 这个容易记错的字段——Qwen2-7B 实际是不共享的
           （多一份 152064 x 3584 的 lm_head）。
    """
    import torch.nn as nn

    named = dict(model.named_parameters())

    # 1) 从某个被量化的 nn.Linear 推断逻辑/物理元素比
    #    （4bit 打包后 numel 约为逻辑量的一半，ratio ≈ 2）
    ratio = 1.0
    for m in model.modules():
        if isinstance(m, nn.Linear) and getattr(m.weight, "quant_state", None) is not None:
            logical = int(m.weight.shape[0]) * int(m.weight.shape[1])
            packed = int(m.weight.numel())
            if packed > 0 and logical > packed:
                ratio = logical / packed
            break

    # 2) 逐参数累加；LoRA 权重是 fp16 真实张量，本身就是逻辑值，不能乘 ratio
    total = 0
    n_quantized = 0
    for name, p in named.items():
        if "lora_" in name:
            total += int(p.numel())
            continue
        if getattr(p, "quant_state", None) is not None:
            total += int(p.numel() * ratio)
            n_quantized += 1
        else:
            total += int(p.numel())

    # 3) 输出头与词嵌入共享时不能重复计（按张量名判断，不猜 config）
    emb_name = next((n for n in named if n.endswith("embed_tokens.weight")), None)
    has_lm_head = any(n.endswith("lm_head.weight") for n in named)
    if emb_name and not has_lm_head:
        total -= int(named[emb_name].numel())

    print(f"      [诊断] 量化元素比 ratio={ratio:.3f}，量化参数张量 {n_quantized} 个，"
          f"lm_head 独立={has_lm_head}")
    return total


def _count_layers(model) -> int:
    """数模型层数（Qwen2-7B 为 28 层），用于算单层参数量。"""
    for attr in ("model", "base_model"):
        m = getattr(model, attr, None)
        if m is None:
            continue
        inner = getattr(m, "model", m)
        layers = getattr(inner, "layers", None)
        if layers is not None:
            return len(layers)
    return 28


print_exact_trainable(model)
# PEFT 自带估算值一并打印，方便对照两者差异
model.print_trainable_parameters()

# ---------- 5. 数据格式化（Qwen2 Chat 模板） ----------
SYSTEM_PROMPT = "你是专业的中文问答助手。请直接、准确、简洁地回答问题，不要冗长铺垫。"

def format_example(ex):
    """把 instruction/input/output 拼成 Qwen2 的 chat 模板文本"""
    user_content = ex["instruction"]
    if ex.get("input") and ex["input"].strip():      # 部分样本带额外输入
        user_content += "\n\n输入：" + ex["input"]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": ex["output"]},
    ]
    # apply_chat_template 输出形如 <|im_start|>system...<|im_end|>... 的完整文本
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

def tokenize_fn(examples):
    """文本 → input_ids/attention_mask/labels"""
    tok = tokenizer(examples["text"], truncation=True,
                    max_length=MAX_SEQ_LEN, padding=False)
    tok["labels"] = tok["input_ids"].copy()   # 全序列监督
    # 进阶优化（可选）：把 user/system 部分 mask 成 -100，只算 assistant 的 loss
    return tok

print("加载并格式化训练数据...")
train_ds = load_dataset("json", data_files=TRAIN_PATH, split="train")
train_ds = train_ds.map(format_example)                       # 逐条拼模板
train_ds = train_ds.map(tokenize_fn, batched=True,
                        remove_columns=train_ds.column_names)  # 转 token
train_ds = train_ds.filter(lambda x: len(x["input_ids"]) > 0)  # 丢弃空序列
print(f"训练样本数：{len(train_ds)}")

# ---------- 6. 训练参数 ----------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,   # 1：小显存单卡
    gradient_accumulation_steps=GRAD_ACCUM,   # 16 步累积 → 等效 batch 16
    num_train_epochs=NUM_EPOCHS,              # 默认 1 轮：小数据量下轮次过多易过拟合
    learning_rate=LEARNING_RATE,              # 2e-4：LoRA 的典型学习率
    lr_scheduler_type="cosine",               # 余弦退火，收敛更稳
    warmup_ratio=WARMUP_RATIO,
    logging_steps=LOGGING_STEPS,
    save_strategy="epoch",                    # 每轮存一个 checkpoint
    save_total_limit=2,                       # 只保留最近 2 个，省磁盘
    bf16=True,                                # Ampere 卡（3060/T4）支持 bf16
    optim="paged_adamw_8bit",                 # 8bit 优化器，再省约一半优化器显存
    report_to="none",                         # 不向外部平台上报日志
    remove_unused_columns=False,              # 保留 input_ids/labels 供 collator 使用
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

# ---------- 7. 训练 ----------
print("开始训练（3060 12GB 预计数小时，云端 T4 约 3-4 小时）...")
trainer.train()

# ---------- 7.1 训练显存实测 ----------
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_properties(0).name
    gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
    print("\n" + "=" * 62)
    print("训练显存实测（torch 统计）")
    print("=" * 62)
    print(f"  显卡               : {gpu_name}（共 {gpu_total:.1f} GB）")
    print(f"  显存峰值 allocated : {peak_alloc:.2f} GB")
    print(f"  显存峰值 reserved  : {peak_reserved:.2f} GB")
    print(f"  峰值占卡容量       : {peak_reserved/gpu_total*100:.1f}%")
    print("=" * 62)
    print("→ 记下『显存峰值』，替换文档里的估算值。")
    print("  要给出『较全量 fp16 下降 X%』的对照，另跑 tools/baseline_fp16.py。\n")

# ---------- 8. 保存 LoRA adapter（不含基座权重，约 15MB） ----------
os.makedirs(ADAPTER_DIR, exist_ok=True)
model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print(f"LoRA adapter 已保存到 {ADAPTER_DIR}")

# ---------- 9. 画训练 loss 曲线（简历素材） ----------
log_history = trainer.state.log_history
steps, losses = [], []
for log in log_history:
    if "loss" in log:
        steps.append(log["step"])
        losses.append(log["loss"])
if losses:
    plt.figure(figsize=(8, 4))
    plt.plot(steps, losses, marker="o", ms=3, color="#4C9BE8")
    plt.xlabel("Step")
    plt.ylabel("Training Loss")
    plt.title("LoRA Fine-tuning Loss Curve (Qwen2-7B)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    curve_path = os.path.join(OUTPUT_DIR, "loss_curve.png")
    plt.savefig(curve_path, dpi=150)
    print(f"loss 曲线已保存：{curve_path}")
    print(f"最终训练 loss：{losses[-1]:.4f}（观察是否平稳收敛、无反弹 → 无明显过拟合）")

print("训练完成！下一步运行：python 3_evaluate.py")
