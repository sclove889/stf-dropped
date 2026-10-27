#!/usr/bin/env bash
# AutoDL 上一次性装好 LLaMA-Factory。
# 用法：bash autodl_setup.sh
#
# 前提：已在 AutoDL 开好实例（推荐 RTX 4090 24GB）。**镜像的 Python 版本不用管**——
#       本脚本自己在数据盘上建 Python 3.11 环境，不碰镜像自带的 base。
#
# 为什么不用镜像自带的 Python：
#   AutoDL 很多镜像是 Python 3.8（连标着 torch_env 的也是），而 LLaMA-Factory
#   要求 >= 3.11。版本不够时 pip 报的是 "No matching distribution found for
#   accelerate>=1.3.0"——看着像镜像源问题，其实是 pip 按 requires-python 把新版本
#   全筛掉了，极易查错方向。
#
# 所有大文件都放 /root/autodl-tmp（数据盘）：系统盘只有 30GB 且不可扩容。

set -e

ENV=/root/autodl-tmp/envs/lf
PY=$ENV/bin/python
LF=/root/autodl-tmp/LLaMA-Factory

# torch 2.6.0 = CUDA 12.4，是 LLaMA-Factory 官方 Docker 镜像用的组合。
# ⚠️ 这个版本必须钉死。理由见下面 4/7 的注释。
TORCH="torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0"

echo "════════ 1/7 查驱动 ════════"
# 无卡模式下 nvidia-smi 读不到设备，但 /proc 里的驱动版本还在
DRV=$(sed -n 's/.*NVRM version:.*Module *\([0-9.]*\).*/\1/p' /proc/driver/nvidia/version 2>/dev/null)
echo "驱动 ${DRV:-未知}"
if [ -n "$DRV" ] && [ "$(printf '%s\n525\n' "${DRV%%.*}" | sort -n | head -1)" != "525" ]; then
    echo "⚠️  驱动 < 525，跑不了 CUDA 12.x。换镜像或换机器。"
    exit 1
fi
echo "✅ 支持 CUDA 12.x（torch 2.6.0 + cu124 的要求）"

echo "════════ 2/7 缓存目录挪到数据盘 ════════"
mkdir -p /root/autodl-tmp/{hf_cache,ms_cache,uv_cache,models,work}
if grep -q '话术项目' ~/.bashrc 2>/dev/null; then
    echo "（.bashrc 里已经有了，跳过）"
else
cat >> ~/.bashrc <<'EOF'

# ── 话术项目 ──
export PATH=/root/autodl-tmp/envs/lf/bin:$PATH   # 默认用数据盘上的 3.11 环境
export HF_HOME=/root/autodl-tmp/hf_cache
export MODELSCOPE_CACHE=/root/autodl-tmp/ms_cache
export UV_CACHE_DIR=/root/autodl-tmp/uv_cache    # 和环境同盘才能硬链接，也不撑爆系统盘
export USE_MODELSCOPE_HUB=1                      # 国内走魔搭，比 HuggingFace 快
export TOKENIZERS_PARALLELISM=false
EOF
fi
export HF_HOME=/root/autodl-tmp/hf_cache
export MODELSCOPE_CACHE=/root/autodl-tmp/ms_cache
export UV_CACHE_DIR=/root/autodl-tmp/uv_cache
export USE_MODELSCOPE_HUB=1
export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
echo "✅ 缓存指向 /root/autodl-tmp"

echo "════════ 3/7 拉 LLaMA-Factory ════════"
# 学术加速只在这一步开。它自己的提示就写着「开启后访问 pip 源等会*更慢*」，
# 所以装依赖前必须关掉——这里用子 shell 隔离，出了括号就失效。
if [ -d "$LF" ]; then
    echo "已存在，跳过 clone"
else
    ( source /etc/network_turbo 2>/dev/null || true
      git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git "$LF" )
fi

echo "════════ 4/7 建 Python 3.11 环境（数据盘）════════"
if [ -x "$PY" ]; then
    echo "已存在：$($PY -V)"
else
    conda create -p "$ENV" python=3.11 -y 2>&1 | tail -3
fi
$PY -m pip install -U pip uv -q 2>&1 | grep -vE 'WARNING: Running pip|recommended to use' || true
echo "✅ $($PY -V)，uv $($PY -m uv --version | awk '{print $2}')"

echo "════════ 5/7 装依赖（3-8 分钟）════════"
# 用 uv 而不是 pip：LLaMA-Factory 的依赖范围开得很宽（transformers>=4.55,<=5.8.0 这种），
# pip 的回溯式解析会下载大量候选 wheel 只为读元数据，实测能卡 20 分钟以上不出声。
# uv 同样的解析 3 分钟出头。LLaMA-Factory 官方也是拿 uv 当包管理器的。
#
# ⚠️ torch 必须和 -e . 写在同一条命令里。
#   pyproject 里是 torch>=2.4.0 没有上限，单独装 -e . 会被升到最新版，
#   而最新版依赖的是 nvidia-*-cu13（CUDA 13），要驱动 >= 580，你的 550 跑不了。
#   更麻烦的是事后补救：cu12 和 cu13 的包会往同一个路径写同名的
#   libcudnn.so.9，卸掉 cu13 会把 cu12 的那个文件一起带走，torch 直接 import 失败。
#   一次解析里就钉住版本，根本不制造这个局面。
cd "$LF"
$PY -m uv pip install --python "$PY" -e . $TORCH
$PY -m uv pip install --python "$PY" -r requirements/metrics.txt bitsandbytes modelscope
echo "✅ 依赖装完"

echo "════════ 6/7 自检 ════════"
$PY - <<'PYEOF'
import torch
print(f"torch        {torch.__version__}  (CUDA {torch.version.cuda})")
print(f"CUDA 可用     {torch.cuda.is_available()}      ← 无卡模式下 False 是正常的")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"显卡         {p.name}  {p.total_memory/1024**3:.1f} GB")
for m in ('transformers', 'peft', 'trl', 'datasets', 'accelerate'):
    print(f"{m:<12} {__import__(m).__version__}")
try:
    import bitsandbytes as bnb
    print(f"bitsandbytes {bnb.__version__}  ✅ QLoRA 可用")
except Exception as e:
    print(f"❌ bitsandbytes 有问题：{e}")
    print("   QLoRA 跑不了，改用 bf16 LoRA（删掉 yaml 里 quantization 两行，需要更大显存）")
# CUDA 13 的包是裸名或 -cu13，CUDA 12 的带 -cu12。混进来就说明 torch 被升过版。
import subprocess, sys, re
pkgs = subprocess.run([sys.executable, '-m', 'pip', 'list'], capture_output=True, text=True).stdout
bad = [l.split()[0] for l in pkgs.splitlines()
       if re.match(r'^nvidia-', l) and not l.split()[0].endswith('-cu12')]
print(f"\nCUDA 13 残留  {bad if bad else '无 ✅'}")
PYEOF
$ENV/bin/llamafactory-cli version 2>&1 | tail -4

echo "════════ 7/7 建工作目录 ════════"
mkdir -p /root/autodl-tmp/work/{data,saves,logs,pred}
$PY -m pip cache purge -q 2>/dev/null || true   # 腾回系统盘
echo "✅ 全部完成"
echo
echo "下一步（都在无卡模式下做，省钱）："
echo "  1. 预下模型： python -c \"from modelscope import snapshot_download as d; print(d('Qwen/Qwen3-8B'))\""
echo "  2. 传 smoke.jsonl 到 /root/autodl-tmp/work/data/"
echo "  3. 注册数据集： cd /root/autodl-tmp/work && python register_dataset.py huashu_smoke \$PWD/data/smoke.jsonl"
echo "  4. 然后才切 GPU 开机，跑训练"
echo
echo "⚠️  新开的终端要先 source ~/.bashrc，否则 python 还是镜像自带的 3.8"
