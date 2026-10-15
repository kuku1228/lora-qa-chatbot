# 云端补数据 Runbook（AutoDL 单卡）

> **这份文档的唯一目的**：一次租机把项目里所有"写了但没数据"的地方补齐。
>
> 目标产出 6 个数字，全部来自真实运行：
> PPL 对比、消融结论、训练显存峰值、可训练参数精确值、token 长度分布、Agent 真实决策准确率。
>
> 每步跑完都标了**要记下什么、记到哪里**。跑完照 [`docs/结果记录模板.md`](结果记录模板.md) 填一遍即可。

---

## 0. 时间与预算

| 阶段 | 内容 | 预计 |
|---|---|---|
| 准备 | 租实例、装依赖、下/挂载权重 | 1–1.5 小时 |
| ① | 数据分布 + token 统计 | 10 分钟 |
| ② | LoRA 训练（含显存采集） | 2 小时 10 分 |
| ③ | 全量 fp16 显存对照 | 20–30 分钟 |
| ④ | PPL 评估 | 30–40 分钟 |
| ⑤ | 四组消融 | 30–45 分钟 |
| ⑥ | vLLM 部署 + Agent 评测 | 40–60 分钟 |
| ⑦ | 收尾填数 | 20 分钟 |

**合计约 6–8 小时**，¥2–3/小时 → 一晚约 **¥20–25**。建议直接租 8 小时不关机，避免中途释放。

> 显卡怎么选：训练 1 epoch（1,116 步）在 24GB 卡上约 2 小时、T4 上 3–4 小时。
> 优先 4090（24GB）或 A10；T4 便宜但会把总时长拖长。**显存至少 12GB**。

---

## 1. 上传代码到实例

三种方式，任选：

**方式 A（推荐）：GitHub / Gitee 中转**
```bash
# 本地：先推到远程仓库（本地已 commit，只差 push）
git remote add origin <你的仓库地址>
git push -u origin main

# 实例上：
git clone <你的仓库地址> lora-qa-chatbot && cd lora-qa-chatbot
```

**方式 B：直接上传压缩包**（AutoDL 支持拖拽上传，或用 JupyterLab 的文件面板）
本地先打包（**不要**包含 80MB 的 adapter 和 6.5MB 数据）：
```powershell
# 在项目根目录执行
git archive -o ..\lora-qa-chatbot.zip HEAD
```
> `git archive` 只会打包已提交的文件，天然排除 `.gitignore` 里的东西，最干净。

**方式 C：实例上克隆 + 本地改**
不建议，容易和本地版本分叉。

---

## 2. 环境准备

```bash
cd lora-qa-chatbot

# 镜像（config.py 会自动兜底，但显式设置更稳）
export HF_ENDPOINT=https://hf-mirror.com

# 训练依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 关键：torch 必须是 CUDA 版，否则 bitsandbytes 装不上
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望输出 True；若是 False 或没装 CUDA 版：
# pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**如果系统盘满了导致 pip 失败**（AutoDL 常见），把临时目录挪到共享内存：
```bash
mkdir -p /root/tmp && export TMPDIR=/root/tmp
# 或 export TMPDIR=/dev/shm
```

**提前把 vLLM 装好**（放后台跑，别等训练完再装）：见第 8 步的版本要求。

---

## 3. 准备模型权重

`config.py` 里 `BASE_MODEL = "Qwen/Qwen2-7B-Instruct"`，首次运行会自动从 HuggingFace 下载（约 15GB）。
国内网络建议直接从魔搭拉：

```bash
pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
mkdir -p models
python - <<'PY'
from modelscope import snapshot_download
p = snapshot_download("qwen/Qwen2-7B-Instruct", local_dir="models/Qwen2-7B-Instruct")
print("模型已下载到:", p)
PY
```

然后用环境变量或改 `config.py` 指向本地路径：
```bash
# 最省事的做法：把 config.py 第 16 行改成
# BASE_MODEL = "models/Qwen2-7B-Instruct"
sed -i 's|BASE_MODEL = "Qwen/Qwen2-7B-Instruct"|BASE_MODEL = "models/Qwen2-7B-Instruct"|' config.py
grep BASE_MODEL config.py
```

> ⚠️ 改完 `config.py` 不要提交回本地仓库（这是实例上的临时改动）。

---

## 4. ① 数据分布 + token 长度统计（10 分钟）

```bash
python 1_data_preprocess.py 2>&1 | tee logs_01_preprocess.log
```

**要记下的东西**（终端里会直接打印）：
- 清洗后总条数、丢弃条数、保留率
- `instruction` / `output` 的 **P50 / P90 / P95 / P99 / Max**（字符）
- 套完 chat 模板后的 **token 长度 P50 / P90 / P95 / P99 / Max**
- 各候选 `max_seq_len`（512/768/1024/2048）下的**截断比例**

同时会生成 `output/data_stats.json`，这些数字都在里面，不用手抄。

**这一步回答的问题**：`max_seq_len=1024` 到底截断了多少样本、这个值定得合不合理。

---

## 5. ② LoRA 训练 + 显存采集（2 小时 10 分）

**开两个终端**（AutoDL 的 JupyterLab 可以开多个 Terminal）。

**终端 A —— 先起显存采样**：
```bash
mkdir -p logs
python tools/monitor.py --interval 5 --summary-every 900 --out logs/train_mem.csv
```

**终端 B —— 再开训练**：
```bash
python 2_train_lora.py 2>&1 | tee logs_02_train.log
```

> 顺序不能反：先起 monitor 才能抓到加载模型那一刻的峰值。

**要记下的东西**：
- 训练脚本开头打印的 **`可训练参数`** 与 **`可训练占比`**（精确值，逐参数累加）
- 训练脚本结尾打印的 **`显存峰值 allocated / reserved`**
- `trainer` 日志里的**最终 loss** 与总步数
- `logs/train_mem.csv` 及其同名 `.summary.txt` 里的**显存峰值**

**产出文件**：`output/lora-qwen2/final/`（adapter）、`output/lora-qwen2/loss_curve.png`

> 训练跑完后 monitor 还在采，Ctrl+C 停掉它，记下 `logs/train_mem*.summary.txt` 里的峰值。

---

## 6. ③ 全量 fp16 显存对照（20–30 分钟，可选但推荐）

这一步是为了把"较全量 fp16 下降约 70%"从估算变成实测。

**终端 A**：
```bash
python tools/monitor.py --interval 5 --out logs/baseline_fp16_mem.csv
```

**终端 B**：
```bash
python tools/baseline_fp16.py --steps 50 2>&1 | tee logs_03_baseline.log
```

**要记下的东西**：
- 脚本会先打印**理论显存下限**（权重 14GB + 梯度 14GB + AdamW 状态 56GB ≈ 84GB）
- 然后尝试实跑：
  - **跑通** → 记下 `显存峰值 allocated / reserved`
  - **OOM** → 这本身就是结论，脚本会把 OOM 当结果如实打印出来

> 24GB 卡上**必然 OOM**，这是预期行为，不是脚本坏了。
> 这时表述写成：「全量 fp16 微调理论显存需求 84GB 以上，24GB 单卡直接 OOM；本方案实测峰值 ___GB，因此单卡可训」——比硬套一个"下降 70%"更可信。

---

## 7. ④⑤ 评估与消融（合计 1–1.5 小时）

```bash
# ④ PPL 对比 + 同题生成对比
python 3_evaluate.py 2>&1 | tee logs_04_evaluate.log

# ⑤ 四组消融
python ablation.py 2>&1 | tee logs_05_ablation.log
```

**④ 要记下的东西**：
- `原始 Qwen2-7B-Instruct` 的 PPL
- `LoRA 微调后` 的 PPL
- 下降幅度（%）

> 两个脚本都已改成"**只加载一次模型**"（先测基座，再在同一份模型上叠加 adapter），
> 比原来省掉 2 次 15GB 权重的读取与量化初始化，共约省 20–30 分钟。

**⑤ 要记下的东西**：`output/ablation.md` 里 6 题 × 4 组 = 24 个回答，重点看：
- **B 组（原始 + prompt）**：术语错误（如 "LoRA = 无线通信协议"）有没有被 prompt 修正？
- **C 组（微调，无 prompt）**：同样的错误有没有被修正？
- **结论怎么写**：如果 B 没修好、C 修好了 → "prompt 只改变输出组织方式，术语与领域知识的修正必须由参数更新带来"，这句就**有实验支撑**了。

---

## 8. ⑥ vLLM 部署 + Agent 评测（40–60 分钟）

**先装 vLLM（务必固定版本）**：
```bash
pip install "vllm==0.8.5" -i https://mirrors.aliyun.com/pypi/simple/
```
> 不要装最新版：AutoDL 镜像的 CUDA 12.8 驱动会让新版 vLLM 报 `driver too old`。

**启动服务**：
```bash
bash agent/serve_vllm.sh
# 验证
curl http://127.0.0.1:8000/v1/models
```
> ⚠️ `--served-model-name`（`qwen2-lora`）与 `--lora-modules` 的模块名（`lora1`）**不能同名**，
> 否则请求会命中 base 模型、LoRA 静默失效（表现为模型回退到微调前行为）。

**跑 Agent 决策评测**：
```bash
pip install -r agent/requirements-agent.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python agent_eval.py --backend openai --base-url http://127.0.0.1:8000/v1 --model lora1 \
    2>&1 | tee logs_06_agent_eval.log
```

**要记下的东西**：
- 工具选择正确率
- **决策准确率**（这是要写进简历/README 的核心数字）
- 误调用率、漏调用率
- 错因分布与分场景准确率

**产出**：`output/agent_eval.md`、`output/agent_eval.json`

> 报告开头会自动标注后端名。用 `openai` 后端时不会有"不代表模型能力"的警示，
> 说明这份就是真实的模型指标。

> 如果 vLLM 起不来但你又想先看效果，可以临时改用其它 OpenAI 兼容服务（如 DashScope/OpenAI），
> 但那样测的就不是你微调后的模型了，写文档时必须注明。

---

## 9. ⑦ 收尾：把数字填回去

1. 打开 [`docs/结果记录模板.md`](结果记录模板.md)，把 6 组数字填进去
2. 用填好的内容替换 README 的「运行结果」小节与 `docs/简历项目描述.md` 里的占位
3. 把要留档的产物下载回本地（AutoDL 实例文件面板可直接下载）：
   ```
   logs/*.log  logs/*.csv  logs/*.summary.txt
   output/data_stats.json
   output/comparison.md
   output/ablation.md
   output/loss_curve.png
   output/agent_eval.md  output/agent_eval.json
   output/lora-qwen2/final/adapter_config.json   （只要配置文件，权重 80MB 不用下）
   ```
4. 本地 `git add` → `commit` → 保留这次的真实数据

> ⚠️ **记得在释放实例前把产物下载下来**，AutoDL 释放后数据不可恢复。

---

## 10. 常见坑速查

| 现象 | 原因 | 处理 |
|---|---|---|
| `pip install` 报 No space left | 系统盘满 | `export TMPDIR=/root/tmp` 或 `/dev/shm` |
| `bitsandbytes` 导入失败 | torch 不是 CUDA 版 | `pip install bitsandbytes --prefer-binary`，先确认 `torch.cuda.is_available()` |
| `CUDA out of memory`（训练） | 显存不足 | 把 `config.py` 的 `MAX_SEQ_LEN` 降到 768，或关 `DOUBLE_QUANT` |
| 训练报 `use_cache` 冲突 | gradient checkpointing 与 cache 冲突 | 脚本已处理（`2_train_lora.py` 里设了 `use_cache=False`） |
| vLLM 报 `driver too old (found 12080)` | vLLM 新版与 CUDA 12.8 驱动不兼容 | 固定 `vllm==0.8.5` |
| vLLM 报 `invalid tool call parser: qwen` | 0.8.5 不支持 Qwen 原生工具调用 | 不用管，`agent_eval.py` 走 Prompt 模式协议，不依赖原生调用 |
| 模型回答还是"LoRa 无线通信" | LoRA 没生效：模块名与 served-name 同名 | 确认 `--lora-modules lora1=...`，且 `--model lora1` |
| 下载慢/超时 | 网络 | `export HF_ENDPOINT=https://hf-mirror.com` |

---

## 11. 一句话检查表（跑完对着勾）

- [ ] `output/data_stats.json` 里有 token 长度分位数与截断比例
- [ ] `logs_02_train.log` 里有**精确可训练参数量**与占比
- [ ] `logs_02_train.log` 或 `logs/train_mem*.summary.txt` 里有**显存峰值**
- [ ] `logs_03_baseline.log` 里有全量 fp16 的对照结论（跑通值 或 OOM 记录）
- [ ] `logs_04_evaluate.log` 里有**两个 PPL 值**和下降幅度
- [ ] `output/ablation.md` 有 24 个回答，且能对 B/C 两组下结论
- [ ] `output/agent_eval.md` 的**决策准确率**来自 `openai` 后端（不是 oracle）
- [ ] 所有产物已下载回本地，实例已释放
