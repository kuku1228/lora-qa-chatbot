#!/usr/bin/env bash
# ============================================================
# serve_vllm.sh —— AutoDL 上把"微调后的 Qwen2-7B + LoRA"部署为
# OpenAI 兼容 API（vLLM），供 LangGraph / DeepAgents 通过 ChatOpenAI 接入。
#
# 用法（在项目根目录执行）：
#   bash serve_vllm.sh
# 验证：
#   curl http://127.0.0.1:8000/v1/models
# ============================================================
set -e

# 1) 安装 vLLM（国内源）
# 注意：AutoDL 镜像 GPU 驱动为 CUDA 12.8，vLLM 最新版要求驱动 >=12.9 会报
# "NVIDIA driver too old"；固定装 0.8.x（对应 torch 2.6 + CUDA 12.4，兼容 12.8 驱动）
pip install -q "vllm==0.8.5" -i https://mirrors.aliyun.com/pypi/simple/

# 2) 启动服务（后台运行，日志写 vllm.log）
# 参数说明：
#   --enable-lora / --lora-modules <名>=<adapter路径>  挂载 LoRA 适配器
#   --served-model-name                                base 模型的对外名（qwen2-lora）
#   ⚠️ LoRA 模块名必须与 served-model-name 不同（lora1），否则请求会命中 base 模型、
#      导致 LoRA 微调效果不生效（模型回答仍是微调前行为）
#   --max-model-len                                   4096：7B fp16 + KV cache 在 4090 24G 上的稳妥值
#   --gpu-memory-utilization                          0.92：给 CUDA 预留余量
nohup vllm serve models/Qwen2-7B-Instruct \
    --served-model-name qwen2-lora \
    --enable-lora \
    --lora-modules lora1=output/lora-qwen2/final \
    --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.92 \
    --trust-remote-code \
    > vllm.log 2>&1 &

echo "vLLM 已后台启动，日志：vllm.log"
echo "等待模型加载（约 1-3 分钟）..."
sleep 30
tail -5 vllm.log
echo "验证：curl http://127.0.0.1:8000/v1/models"
