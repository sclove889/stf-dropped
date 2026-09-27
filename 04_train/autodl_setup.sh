#!/usr/bin/env bash
# AutoDL 上一次性装好 LLaMA-Factory。
# 用法：bash autodl_setup.sh
#
# 前提：已在 AutoDL 开好实例（推荐 RTX 4090 24GB），基础镜像选 PyTorch 2.x + CUDA 12.x
# 注意：系统盘只有 30GB 且不可扩容，所有大文件必须放 /root/autodl-tmp（数据盘）

set -e   # 任何一步失败就停，不要带着半装好的环境往下走

echo "════════ 1/6 开学术加速（不开的话 clone GitHub 会很慢或失败）════════"
if [ -f /etc/network_turbo ]; then
    source /etc/network_turbo
    echo "✅ 已开启"
else
    echo "⚠️  没找到 /etc/network_turbo，可能不是 AutoDL 环境，继续"
fi

echo "════════ 2/6 把缓存目录挪到数据盘 ════════"
# 模型动辄十几 GB，默认下到 ~/.cache 会把 30GB 系统盘撑爆
mkdir -p /root/autodl-tmp/{hf_cache,ms_cache,models,work}
cat >> ~/.bashrc <<'EOF'

# ── LLaMA-Factory 环境变量 ──
export HF_HOME=/root/autodl-tmp/hf_cache
export MODELSCOPE_CACHE=/root/autodl-tmp/ms_cache
export USE_MODELSCOPE_HUB=1          # 国内走魔搭，比 HuggingFace 快得多
export TOKENIZERS_PARALLELISM=false  # 关掉一个没用的警告
EOF
export HF_HOME=/root/autodl-tmp/hf_cache
export MODELSCOPE_CACHE=/root/autodl-tmp/ms_cache
export USE_MODELSCOPE_HUB=1
export TOKENIZERS_PARALLELISM=false
echo "✅ 缓存已指向 /root/autodl-tmp"

echo "════════ 3/6 拉 LLaMA-Factory ════════"
cd /root/autodl-tmp
if [ -d LLaMA-Factory ]; then
    echo "已存在，跳过 clone"
else
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
fi
cd LLaMA-Factory

echo "════════ 4/6 装依赖（5-10 分钟）════════"
pip install -e . -q
pip install -r requirements/metrics.txt -q
pip install bitsandbytes -q        # QLoRA 4bit 量化必需
pip install modelscope -q          # 从魔搭下模型必需
echo "✅ 依赖装完"

echo "════════ 5/6 自检 ════════"
llamafactory-cli version
python - <<'PY'
import torch
print(f"torch      {torch.__version__}")
print(f"CUDA 可用   {torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"显卡        {p.name}  {p.total_memory/1024**3:.1f} GB")
try:
    import bitsandbytes
    print(f"bitsandbytes {bitsandbytes.__version__}  ✅ QLoRA 可用")
except Exception as e:
    print(f"❌ bitsandbytes 有问题：{e}")
    print("   QLoRA 跑不了，改用 bf16 LoRA（需要更大显存）")
PY

echo "════════ 6/6 建工作目录 ════════"
mkdir -p /root/autodl-tmp/work/{data,saves,logs,pred}
echo "✅ 全部完成"
echo
echo "下一步："
echo "  1. 用 scp / AutoDL 网盘 把 huashu.jsonl 传到 /root/autodl-tmp/work/data/"
echo "  2. 把它注册进 LLaMA-Factory/data/dataset_info.json"
echo "  3. 传 qwen3_8b_qlora.yaml 到 /root/autodl-tmp/work/"
echo "  4. llamafactory-cli train /root/autodl-tmp/work/qwen3_8b_qlora.yaml"
echo
echo "⚠️  新开的终端要先 source ~/.bashrc 才有上面那些环境变量"
