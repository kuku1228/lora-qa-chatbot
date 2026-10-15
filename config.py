# ============================================================
# config.py —— 全局配置中心
# 所有脚本统一从这里读参数，想调超参只改这一个文件
# ============================================================
import os

# ===== 镜像兜底（必须放在所有 import 之前生效）=====
# 检测到未设置 HF_ENDPOINT 时自动走国内镜像，PyCharm / 终端新老窗口通吃
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 静默 Windows 无 symlink 支持的缓存告警（不影响功能）
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# ---------- 基座模型 ----------
# 阿里开源 Qwen2-7B-Instruct，无需申请权限，HuggingFace 直接可下
BASE_MODEL = "Qwen/Qwen2-7B-Instruct"

# ---------- 数据集 ----------
# hfl/alpaca_zh_51k：Chinese-LLaMA-Alpaca 官方中文指令数据（51k 条）
# instruction/input/output 三字段，格式干净，镜像可访问（tliang/alpaca_cn 已失效，勿用）
DATA_REPO   = "hfl/alpaca_zh_51k"
SAMPLE_NUM  = 20000                # 采样 2 万条
EVAL_NUM    = 200                  # 从中留出 200 条做评估集
TRAIN_PATH  = os.path.join("data", "train.jsonl")
EVAL_PATH   = os.path.join("data", "eval.jsonl")

# 清洗阈值（字符长度，中英文一视同仁）
MAX_INSTRUCTION_LEN = 512     # instruction 过长：截断价值低，直接过滤
MAX_OUTPUT_LEN      = 1024    # output 过长：占显存、影响训练稳定
MIN_OUTPUT_LEN      = 10      # output 过短：没有实质回答内容

# ---------- 量化（省显存核心） ----------
LOAD_IN_4BIT = True
QUANT_TYPE   = "nf4"          # nf4 量化精度优于 fp4
DOUBLE_QUANT = True           # 二次量化，再省 ~0.4 字节/参数

# ---------- LoRA 超参（r / alpha 的取值逻辑） ----------
LORA_R       = 8              # 秩：控制低秩矩阵的参数量，8/16 是工业界最常用区间
LORA_ALPHA   = 16             # 缩放系数：实际缩放 = alpha / r = 2，太大容易训崩
LORA_DROPOUT = 0.05           # 低秩矩阵的 dropout，防过拟合
# Qwen2 的注意力与 FFN 全部线性层（LLaMA 系结构通用）
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# ---------- 训练 ----------
MAX_SEQ_LEN     = 1024        # 单样本最大 token 数（可降到 768 应对小显存）
LEARNING_RATE   = 2e-4        # LoRA 常用学习率
NUM_EPOCHS      = 3
BATCH_SIZE      = 1           # 小显存单卡：batch=1
GRAD_ACCUM      = 16          # 梯度累积 16 步 → 等效 batch_size = 16
WARMUP_RATIO    = 0.03
LOGGING_STEPS   = 20
OUTPUT_DIR      = os.path.join("output", "lora-qwen2")
ADAPTER_DIR     = os.path.join("output", "lora-qwen2", "final")   # 最终 LoRA 适配器

# 国内下载模型/数据集慢？设环境变量走镜像（Windows: set HF_ENDPOINT=https://hf-mirror.com）
