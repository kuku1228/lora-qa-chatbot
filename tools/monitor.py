#!/usr/bin/env python3
"""
tools/monitor.py —— 训练/推理期间的显存与内存采样

为什么需要它：
    README 里的"训练占用约 12GB""较全量 fp16 下降约 70%"目前没有实测支撑。
    这个脚本在训练跑起来后并行采样，直接给出显存峰值，用来替换掉估算值。

用法（另开一个终端，在项目根目录执行）：
    # 每 5 秒采一次，每 10 分钟打一个汇总
    python tools/monitor.py --interval 5 --summary-every 600

    # 指定输出文件（默认 logs/monitor_<时间戳>.csv）
    python tools/monitor.py --out logs/train_mem.csv

跑完看这个文件即可：
    logs/ 下同一时间戳的 .csv（逐次采样）与 .summary.txt（峰值/均值）

只依赖标准库 + nvidia-smi，不需要装 torch。
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime


def _sh(cmd: list[str]) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return (r.stdout or "").strip()
    except Exception:
        return ""


def find_nvidia_smi() -> str | None:
    p = shutil.which("nvidia-smi")
    if p:
        return p
    for cand in ("/usr/bin/nvidia-smi", "/usr/local/bin/nvidia-smi"):
        if os.path.exists(cand):
            return cand
    return None


def gpu_sample(smi: str) -> dict:
    """返回 0 号卡的显存/利用率。"""
    out = _sh([smi,
               "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
               "--format=csv,noheader,nounits"])
    if not out:
        return {}
    line = out.splitlines()[0]
    parts = [x.strip() for x in line.split(",")]
    if len(parts) < 7:
        return {}
    return {
        "gpu_index": parts[0],
        "gpu_name": parts[1],
        "mem_used_mb": float(parts[2]),
        "mem_total_mb": float(parts[3]),
        "util_pct": float(parts[4]),
        "temp_c": float(parts[5]),
        "power_w": parts[6],
    }


def cpu_mem_mb() -> float:
    """从 /proc/meminfo 估算已用内存（Linux）。"""
    try:
        info = {}
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                k, _, v = line.partition(":")
                info[k.strip()] = float(v.strip().split()[0])  # kB
        total = info.get("MemTotal", 0.0)
        avail = info.get("MemAvailable", info.get("MemFree", 0.0))
        return (total - avail) / 1024.0
    except Exception:
        return -1.0


def main() -> int:
    ap = argparse.ArgumentParser(description="训练期间显存/内存采样")
    ap.add_argument("--interval", type=float, default=5.0, help="采样间隔秒数")
    ap.add_argument("--summary-every", type=float, default=600.0, help="每隔多少秒打印一次汇总")
    ap.add_argument("--out", default=None, help="CSV 输出路径")
    ap.add_argument("--max-minutes", type=float, default=0.0,
                    help="最长运行分钟数，0 表示不限制（跑到 Ctrl+C）")
    args = ap.parse_args()

    os.makedirs("logs", exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = args.out or os.path.join("logs", f"monitor_{stamp}.csv")
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    smi = find_nvidia_smi()
    if not smi:
        print("[警告] 找不到 nvidia-smi，本次只采 CPU 内存。", file=sys.stderr)

    print(f"开始采样：间隔 {args.interval}s → {out_csv}")
    print("（训练脚本另开一个终端跑；本脚本 Ctrl+C 停止并输出汇总）\n")

    rows: list[dict] = []
    peak_mem = 0.0
    peak_mem_total = 0.0
    t0 = time.time()
    next_summary = t0 + args.summary_every

    f = open(out_csv, "w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow(["timestamp", "elapsed_s", "gpu_name", "mem_used_mb",
                     "mem_total_mb", "util_pct", "temp_c", "power_w", "cpu_mem_mb"])

    try:
        while True:
            now = time.time()
            if args.max_minutes and (now - t0) > args.max_minutes * 60:
                break
            g = gpu_sample(smi) if smi else {}
            cm = cpu_mem_mb()
            elapsed = now - t0
            writer.writerow([
                datetime.now().strftime("%H:%M:%S"), f"{elapsed:.0f}",
                g.get("gpu_name", ""), f"{g.get('mem_used_mb', 0):.0f}",
                f"{g.get('mem_total_mb', 0):.0f}", f"{g.get('util_pct', 0):.0f}",
                f"{g.get('temp_c', 0):.0f}", g.get("power_w", ""),
                f"{cm:.0f}" if cm >= 0 else "",
            ])
            f.flush()
            if g.get("mem_used_mb"):
                rows.append(g)
                peak_mem = max(peak_mem, g["mem_used_mb"])
                peak_mem_total = g.get("mem_total_mb", 0)

            if now >= next_summary:
                next_summary = now + args.summary_every
                _print_summary(rows, peak_mem, peak_mem_total, elapsed)

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，生成汇总 ...")
    finally:
        f.close()

    elapsed = time.time() - t0
    lines = _print_summary(rows, peak_mem, peak_mem_total, elapsed, final=True)
    summary_path = os.path.splitext(out_csv)[0] + ".summary.txt"
    with open(summary_path, "w", encoding="utf-8") as sf:
        sf.write("\n".join(lines) + "\n")
    print(f"\n逐次采样：{out_csv}")
    print(f"汇总结果：{summary_path}")
    print("\n把 summary.txt 里的『显存峰值』填进 README 和 docs。")
    return 0


def _print_summary(rows: list[dict], peak: float, total: float,
                   elapsed: float, final: bool = False) -> list[str]:
    L: list[str] = []
    if not rows:
        L.append("[采样] 还没有有效样本")
    else:
        mems = [r["mem_used_mb"] for r in rows]
        avgs = sum(mems) / len(mems)
        utils = [r["util_pct"] for r in rows]
        active = [u for u in utils if u > 5]
        L.append("─" * 58)
        L.append(f"[采样汇总] 已运行 {elapsed/60:.1f} 分钟，样本 {len(rows)} 个")
        L.append(f"  显存峰值   : {peak:,.0f} MB ({peak/1024:.2f} GB)"
                 + (f" / 共 {total:,.0f} MB" if total else ""))
        L.append(f"  显存均值   : {avgs:,.0f} MB ({avgs/1024:.2f} GB)")
        if active:
            L.append(f"  训练中平均 GPU 利用率: {sum(active)/len(active):.0f}%"
                     f"（仅统计 >5% 的样本，共 {len(active)} 个）")
        else:
            L.append("  GPU 利用率 : 全程接近 0 —— 训练可能还没开始或已结束")
        L.append("─" * 58)
    if final:
        L.append("")
        L.append("→ 要填进文档的数字：「显存峰值」这一行。")
        L.append("→ 若同时想验证『较全量 fp16 下降约 70%』，再跑一次 tools/baseline_fp16.py 取对照值。")
    for line in L:
        print(line)
    return L


if __name__ == "__main__":
    sys.exit(main())
