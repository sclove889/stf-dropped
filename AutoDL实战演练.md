# AutoDL 实战演练：用公开数据跑通整条链路

> 关联文档：[执行流程.md](执行流程.md)（生产流程）、[选型论证与评测方案.md](选型论证与评测方案.md)（为什么选 LoRA/Qwen3）、[销售会议转写结构化抽取-技术调研.md](销售会议转写结构化抽取-技术调研.md)（文献）
>
> 本文档写于 2026-09-26。所有 AutoDL 卡型/价格、数据集链接都做过实际核实，未能核实的项逐条标注。

---

## 0. 先说清楚：这次演练的目的不是「训出好模型」

用公开数据训出来的模型**不能用在你的业务上**——域不对、标签不对。所以如果目标定成"看看效果多好"，这次演练就是纯烧钱。

演练要产出的是**三个可以直接搬到真实数据上的资产**：

| # | 产出 | 为什么值钱 |
|---|---|---|
| 1 | 一套跑通的 AutoDL + LLaMA-Factory 训练配置 | 真实数据到位时只换 `dataset`，不用再调环境 |
| 2 | **一个自建评测 harness**（Hungarian 对齐 + 三态字段 + 幻觉率） | 这是你现在**最缺**的东西。没有尺子，模型换多大都看不出差别 |
| 3 | **一条真实的「样本量 vs F1」曲线**，含 prompt-only 基线 | 直接回答悬了很久的问题：SFT 要多少样本才反超提示词 |

第 3 条最值钱。之前三份文档里这个问题只有论文证据（而且 [FsPONER 说 ~40 个 in-context 示例就饱和](https://arxiv.org/abs/2407.08035) 和 [Bertsch 说 10→1000 还能 +36.8](https://arxiv.org/abs/2405.00200) **互相矛盾**，都没验证）。演练之后你有自己的数。

**验收标准**：演练结束时，`run_eval.py` 能对任意一个 checkpoint 输出一张表（字段级 P/R/F1 + schema 合规率 + 幻觉率 + 漏抽率），且 prompt-only 和至少 3 个样本量档位的 LoRA 都在同一张表上可比。做不到这个，演练算失败，跟 loss 降到多少无关。

---

## 1. 数据集：CSDS 是主力，VCSum 补长文本

### 1.1 为什么是 CSDS（这是全文最关键的一节）

**CSDS**（Chinese Customer Service Dialogue Summarization，[arXiv:2108.13139](https://arxiv.org/abs/2108.13139)，EMNLP 2021）是唯一在**真实中文商业对话**上把你要的四件套全标好的公开数据集。字段映射几乎 1:1：

| 你的 schema | CSDS 字段 | 实例 |
|---|---|---|
| **所属分类** | `intent` | `保修返修及退换货政策`、`配送周期`、`修改订单`、`价保申请流程` |
| **要解决的痛点问题** | `QueSumm` | `用户询问改密码除手机验证外的其他方法。` |
| **解决方案（专家经验）** | `AnsSummLong` | `客服让用户提供号码，交给专员回电解决。` |
| **证据（evidence）** | `QueSummUttIDs` / `AnsSummLongUttIDs` | utterance 下标数组 |

论文自己的表述就是「用户提出问题，客服给出解决方案」——和你的任务是同一个抽象。

规模：**10,701 对话**（train 9,101 / dev 800 / test 800），基于 JDDC。**下载无需申请**，Google Drive 公开 + 百度网盘（提取码 `5dii`）。仓库：<https://github.com/pengyuan-liu/CSDS>（以 arXiv 页链接为准，见 §3.3）

一条对话含多个 QA 对 → 抽出**多条 records**。这正好让你演练**多记录对齐**（Hungarian 匹配），而不是退化成单条抽取。

**它还顺带演练了痛点 #2（标签体系不稳定）**：test split 里有 **146 个不同 intent**（全集约 210 个），这个粒度太细、必然不稳。把它聚成 8–15 类的过程，就是 [TnT-LLM](https://arxiv.org/abs/2403.12173) 式 taxonomy induction 的真实演练（§5）。

### 1.2 CSDS 的两个真实短板

必须知道，否则会拿错结论：

1. **对话短**：实测 test split 平均 **25.1 轮 / 494 字**。你的真实场景是上万字。**CSDS 完全不演练痛点 #3（长录音转写）**。
2. **无 LICENSE 文件**，上游 JDDC 条款不明。跑通链路和自建评测没问题，**商用前必须解决授权**。

### 1.3 VCSum 补第一个短板

**VCSum**（[arXiv:2305.05280](https://arxiv.org/abs/2305.05280)，ACL 2023 Findings）：<https://github.com/hahahawu/VCSum>

- **239 场真实中文多人会议**，平均 **14,462 字/场**、73 轮、5.6 个说话人、6.7 个话题段
- **来源是飞书妙记 ASR 转写后人工校正（WER 8%）**——文本自带真实转写噪声，这对你是**优点**不是缺点
- **MIT 许可证**，本文提到的所有数据集里唯一无条件可商用
- 标注层：`eos_index`（话题分段边界）、`agenda`（段标题）、`discussion`（段摘要）、`summary`（整体摘要）、`highlights`（**字符级 0/1 掩码**，可直接当抽取式证据监督）

**VCSum 的用途只有一个：跑 §0 里遗留的「话题段 vs 整场会议」粒度对照实验**（见 [执行流程.md](执行流程.md) §0）。它没有痛点/方案标签，要自己蒸馏，所以别拿它算主指标。

> ⚠️ 实测与论文不符：论文写划分 193/25/21，实际文件是 **191/22/26**；段级论文写 1076，实际 **1088**。**以实际文件为准**。另外论文正文没有 `agenda`/`discussion` 这两个字段名（论文叫 headline / segmentation summary），字段名以 README 和实际 JSON 为准。

### 1.4 如果要打通「录音 → 转写 → 抽取」全链路

**AliMeeting4MUG**：<https://modelscope.cn/datasets/modelscope/Alimeeting4MUG>（[arXiv:2303.13939](https://arxiv.org/abs/2303.13939)）

654 场真实普通话商务会议（15–30 分钟，2–4 人），人工转写带标点和说话人标签，平均 10,772 token。**5 层人工标注**：话题分段 / 话题级+会话级抽取式摘要 / 话题标题 / 关键词 / **行动项检测（Action Item Detection）**。`CC-BY-4.0` **可商用**，只需一个免费 ModelScope token。

配 OpenSLR 上的 [AliMeeting 原始音频](https://openslr.org/119/)（118.75h，CC BY-SA 4.0 可商用）就能跑真实 ASR + diarization。

**但第一轮演练不要碰这个。** 加进 ASR 环节会让你无法区分「抽取错了」和「转写错了」——这正是 [选型论证与评测方案.md](选型论证与评测方案.md) §B6 要求解耦的两件事。第一轮用 CSDS 的干净文本。

### 1.5 明确避开的数据集

| 数据集 | 为什么避开 |
|---|---|
| **MagicData-RAMC** | `CC BY-NC-ND 4.0`——NC 禁商用，**ND 连你的加工版都不许分发** |
| **DuIE / DuEE** | 竞赛协议明文「不得出于任何商业目的使用」，且是**文档不是对话** |
| **DialogRE 中文版** | 非商用 + 领域是《老友记》剧本；中文版刻意不标 argument type |
| **CrossNER 中文** | **不存在**（5 个域全是英文，基线用 glove.6B + bert-base-cased） |
| **JDDC 1.0/2.0** | `jddc.jd.com` 已失效（302 跳错误页）。CSDS 已释出其子集原文，多数情况不必申请 |
| 数据堂两个客服集 | ModelScope 标 Apache-2.0 但 README 写「商用数据，版权归数据堂」，**冲突未解** |
| `ICIP/TeleSalesCorpus` | cardData 里**无 license 无语言标签** |
| `introvoyz041/CSDS`（HF） | **同名误配**，里面是 DeepLabV3Plus 图像分割代码 |

**弱监督冷启动可选**：`MultiSense/SaleData`（SalesLLM-10k，Apache-2.0，10,000 条中文销售对话，**自带 `pain_points` 字段**）。但它是 LLM 合成、对话中位仅 6 条消息 / 574 字，**只能做词汇/话术的冷启动，绝不能用来评测真实分布**。

**一句实话**：中文「销售对话 +（分类/痛点/解决方案）」这层标注公开世界里不存在。上面这套组合能免费给你**真实的转写文本质感**和**成熟的输出格式**，但那层标签在真实数据上你必须自己标。

---

## 2. AutoDL 卡型：一个成本阶梯，和一个我必须承认的矛盾

### 2.1 官方价格（2026-09-26 从 autodl.com 首页价格表实测）

| GPU | 显存 | 元/小时 | 能跑什么（LLaMA-Factory 官方显存表） |
|---|---|---|---|
| RTX 3090 | 24GB | 1.32 | 14B QLoRA |
| **RTX 4090** | **24GB** | **1.88** | **14B QLoRA（12GB）** ← 流程演练首选 |
| RTX 5090 | 32GB | 2.78 | 14B LoRA bf16 需 32GB，**刚好等于显存，会 OOM，别选** |
| **A800-80GB** | 80GB | **5.59** | **14B LoRA bf16（32GB）/ 32B LoRA（~68GB）** ← 与生产同精度 |
| RTX PRO 6000 | 96GB | 6.98 | 32B LoRA 更宽松 |
| H800 / H20 | 80/96GB | 9.98 | 同上，但贵 79%，**没理由选** |

**没有 H100/H200。** 官网首页两个列表里完全没有这两个型号。
**多机多卡已取消**（`/docs/distributed_training/`：「因此不再支持开通内网IP进行多机多卡并行」）。

未验证：4090D / A40 / L20 / A100 80GB 出现在官网「GPU算力排名」列表里（说明平台有），但**价格表未列 → 单价未验证**。任何型号的**实时库存**必须登录 `autodl.com/market/list` 才能看。

### 2.2 我必须承认的矛盾：这里得用 QLoRA

我在 [选型论证与评测方案.md](选型论证与评测方案.md) §A3 和 [执行流程.md](执行流程.md) §7.5 里写过「**不要用 QLoRA，量化是小显存的妥协，白损精度**」。那是针对 H200 141GB 说的。

**在 4090 24GB 上，14B 只能 QLoRA。** 这不是我改主意了，是约束变了。但要分清哪个产出受影响：

| 产出 | QLoRA 有影响吗 |
|---|---|
| 产出 1（训练配置 + 环境流程） | ❌ 无影响，只有一个 `quantization_bit` 的差别 |
| 产出 2（评测 harness） | ❌ 无影响，harness 不关心模型怎么训的 |
| 产出 3（样本量 vs F1 曲线） | ✅ **有影响。QLoRA 绝对 F1 更低，"反超提示词"的拐点位置可能偏移** |

所以：

> **流程打通用 4090 + 14B QLoRA（¥1.88/h，便宜）。
> 但产出 3 那条曲线，要在 A800 80GB（¥5.59/h）上用 bf16 LoRA 跑，配置与 H200 生产环境一致。**

否则你带回去的曲线拐点是 QLoRA 的拐点，不是生产配置的拐点，会误导后面的标注预算决策——而标注预算是这个项目最大的一笔钱。

### 2.3 成本估算

**第一阶段（流程打通，4090 24GB ¥1.88/h）**

| 步骤 | 时长 | 费用 |
|---|---|---|
| 无卡模式装环境 + 下 Qwen3-14B（~28GB） | ~1.5h × ¥0.1 | ¥0.15 |
| 14B QLoRA，3000 样本，cutoff 2048，3 epoch | ~4h | ¥7.5 |
| prompt-only 基线推理（200 条） | ~0.5h | ¥1 |
| 调试与返工（一定会有） | ~4h | ¥7.5 |
| **小计** | | **约 ¥16** |

**第二阶段（样本量曲线，A800 80GB ¥5.59/h）**

4 个档位（50 / 200 / 800 / 3000 样本）bf16 LoRA + 每档评测推理，约 8h → **约 ¥45**。

**整轮演练 ¥60 量级。** 这个数量级低到不值得为省钱牺牲方法正确性——别为了省 ¥30 去用 5090 32GB 卡着跑然后 OOM 折腾半天。

如果要顺带验证 32B：A800 80GB 上 32B LoRA（~68GB）可行，再加约 ¥40。**但建议第一轮不做**，理由见 §9。

---

## 3. 环境搭建

### 3.1 租实例时就要做对的三件事

1. **数据盘扩到 100GB+**。系统盘 `/` 只有 **30GB 且不可扩容**，这是 AutoDL 最大的坑。Qwen3-14B bf16 权重约 28GB，32B 约 64GB——放系统盘直接爆。
2. **镜像选 Python 3.11+ / PyTorch 2.x / CUDA 12.x**。LLaMA-Factory 官方 README 要求 **Python ≥ 3.11**，而 AutoDL 不少基础镜像还是 3.8/3.10。选错了要自己建 conda 环境。
3. **不要用 CodeWithGPU 上那个 LLaMA-Factory 社区镜像**。我实际打开了 <https://www.codewithgpu.com/i/hiyouga/LLaMA-Factory/LLaMA-Factory>——它**不是 hiyouga 官方维护的**，是第三方魔改版（带 QQ 群、`模型贩卖机.py` 等脚本），可选模型只到 Qwen1.5 / Llama3-8B / GLM-4-9B，**明显停更在 2024 年中**。参照官方 Docker 镜像的规格自己装：Ubuntu 22.04 + CUDA 12.4 + Python 3.11 + PyTorch 2.6.0 + flash-attn 2.7.4。

### 3.2 立刻切无卡模式（省钱关键）

装环境和下模型**完全不需要 GPU**。

```
租用实例 → 立即关机 → 点「无卡模式」开机（0.5核/2GB/无GPU，￥0.1/小时）
```

下一个 28GB 的模型，¥1.88/h 和 ¥0.1/h 差 19 倍。装环境+下模型 ~1.5h，省下的钱不多（¥2.7），但**装环境时抢不到卡的焦虑是真实成本**——无卡模式下你可以慢慢调。

两个限制：
- 同一主账号**同时只能有一个无卡模式实例**
- 进入无卡模式会把 GPU 释放回空闲池，**正常开机时卡可能已被别人租走**。热门卡型（4090）建议改用包日，别来回切

### 3.3 环境变量（这四行决定你会不会踩坑）

```bash
# 1. 别让 HF 缓存撑爆 30GB 系统盘 —— AutoDL 官方文档唯一给的建议
export HF_HOME=/root/autodl-tmp/cache/

# 2. 国内拉模型走 ModelScope（最稳，不依赖任何代理）
export USE_MODELSCOPE_HUB=1

# 3. 需要访问 github / huggingface 时开学术加速（覆盖 github.com / huggingface.co）
source /etc/network_turbo
# 用完关掉，否则某些国内域名会走代理失败
unset http_proxy && unset https_proxy

# 4. 或者走 HF 镜像（与 network_turbo 二选一）
export HF_ENDPOINT=https://hf-mirror.com
```

把前两行写进 `~/.bashrc`，否则每次新 shell 都要重设，忘一次就爆盘。

> `source /etc/network_turbo` 是 AutoDL 的学术资源加速，官方明确「**不承诺稳定性保证**」，遭攻击时可能随时停。所以**模型走 ModelScope（不依赖它），只有 `git clone` 时才开它**。

### 3.4 安装 LLaMA-Factory

```bash
cd /root/autodl-tmp                      # 装在数据盘
source /etc/network_turbo
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
unset http_proxy && unset https_proxy
cd LLaMA-Factory
pip install -e .                          # 官方原文：Installation is mandatory
pip install -r requirements/metrics.txt
llamafactory-cli version                  # 验证
```

`pip install -e .` 不能跳过（官方强调 "Installation is mandatory"）。

### 3.5 预下模型（无卡模式下做）

```bash
pip install modelscope
modelscope download --model Qwen/Qwen3-14B --local_dir /root/autodl-tmp/models/Qwen3-14B
```

后续 yaml 里 `model_name_or_path` 直接写这个本地路径，避免训练时才发现下载失败。

### 3.6 WebUI 走 SSH 隧道，别用 6006

AutoDL 只有 **6006 和 6008** 两个端口有公网映射，而「开放端口」文档页写着「开放端口需要进行企业认证」（**该限制是否也适用于 6006 自定义服务，文档表述含糊 → 未验证**）。

在本地 Windows 上：

```bash
ssh -L 7860:127.0.0.1:7860 -p <端口> root@<AutoDL主机>
```

然后实例内 `llamafactory-cli webui`，本地浏览器开 `http://127.0.0.1:7860`。

不过说实话：**webui 只适合第一次探索**。样本量阶梯实验要跑 4 个配置，用 yaml + CLI 才可复现。

### 3.7 跑完自动关机

```bash
llamafactory-cli train xxx.yaml 2>&1 | tee /root/autodl-tmp/logs/run1.log; /usr/bin/shutdown
```

`tee` 存日志在**前**，`shutdown` 在**后**。顺序反了日志就丢了。

---

## 4. 数据转换：CSDS → 你的 schema

### 4.1 先下数据（在本地 Windows 做，不在 AutoDL 上）

CSDS 在 Google Drive 和百度网盘，AutoDL 的学术加速**不覆盖这两个**。总共约 69MB（train 58.8MB / val 5.1MB / test 5MB），本地下完上传即可。

- Google Drive：<https://drive.google.com/drive/folders/1IrpEUTR2ZanN0ZKmKcbaXjruTAD9JLkF>
- 百度网盘提取码：`5dii`

上传到 AutoDL：用 JupyterLab 的文件上传，或 `scp -P <端口> train.json root@<主机>:/root/autodl-tmp/data/csds/`

### 4.2 第一步永远是 inspect，不是写转换脚本

**CSDS 的官方 `dataset_introduction.md` 有错**——文档写字段叫 `Topic`，**实际 JSON 里叫 `intent`**。所以不要照文档写脚本，先看真实结构：

```python
# 00_spec/inspect_csds.py
import json, collections, itertools

data = json.load(open('data/csds/test.json', encoding='utf-8'))
print(f'对话数: {len(data)}')
print(f'顶层字段: {list(data[0].keys())}\n')

# 把第一条完整打出来看，别猜
print(json.dumps(data[0], ensure_ascii=False, indent=2)[:3000])

# intent 分布 —— 决定 §5 怎么归并
intents = collections.Counter()
for d in data:
    for qa in d.get('QA', []):
        intents[qa.get('intent') or '<空>'] += 1
print(f'\nintent 种类数: {len(intents)}')
for k, v in intents.most_common(20):
    print(f'{v:5d}  {k}')
print(f'\nintent 为空的 QA 对: {intents["<空>"]}')
```

**跑这一步，然后按真实 key 名改下面的常量。** 实测 test split 有 1,517 个 QA 对、146 个不同 intent、**158 个 QA 对 intent 为空**（后者必须处理，见 §4.4）。

### 4.3 转换脚本

```python
# 06_data/csds_to_alpaca.py
import json, random, argparse
from pathlib import Path

# ── 按 inspect 结果核对这几个 key，错一个整个脚本静默产出空数据 ──
K_DIALOG   = 'Dialogue'              # utterance 列表
K_QA       = 'QA'                    # QA 对列表
K_INTENT   = 'intent'                # ⚠️ 不是文档说的 'Topic'
K_PAIN     = 'QueSumm'
K_SOLUTION = 'AnsSummLong'
K_PAIN_IDS = 'QueSummUttIDs'
K_SOL_IDS  = 'AnsSummLongUttIDs'
K_UTT_TEXT = 'utterance'
K_UTT_ROLE = 'speaker'

INSTRUCTION = """你是销售会议记录分析专家。请从下面的对话转写中抽取结构化记录。

每条记录包含四个字段：
- category: 所属分类，必须从给定标签集中选择
- pain_point: 客户要解决的痛点问题，一句话概括
- solution: 提供的解决方案。若对话中未给出方案，填 null，不要编造
- evidence: 支撑本条记录的发言下标数组

只输出 JSON，不要任何解释。若对话中不存在可抽取的记录，输出 {"records": []}。

可选分类标签：
%s"""


def render_dialogue(utts):
    """带下标和角色渲染，下标要和 evidence 对齐。"""
    lines = []
    for i, u in enumerate(utts):
        role = u.get(K_UTT_ROLE, '?')
        text = (u.get(K_UTT_TEXT) or '').strip()
        lines.append(f'[{i}] {role}: {text}')
    return '\n'.join(lines)


def convert(src, label_map, labels, neg_ratio=0.0, seed=0):
    rng = random.Random(seed)
    data = json.load(open(src, encoding='utf-8'))
    out, dropped_no_intent, dropped_empty = [], 0, 0

    for d in data:
        utts = d.get(K_DIALOG) or []
        records = []
        for qa in d.get(K_QA) or []:
            raw_intent = qa.get(K_INTENT)
            if not raw_intent:                       # 158 条这种，见 §4.4
                dropped_no_intent += 1
                continue
            solution = (qa.get(K_SOLUTION) or '').strip()
            records.append({
                'category':   label_map.get(raw_intent, '其他'),
                'pain_point': (qa.get(K_PAIN) or '').strip(),
                # ⚠️ 空字符串 → null，这是在教模型"没有就说没有"
                'solution':   solution or None,
                'evidence':   sorted(set(
                    (qa.get(K_PAIN_IDS) or []) + (qa.get(K_SOL_IDS) or [])
                )),
            })

        if not records:
            dropped_empty += 1
            # 按比例保留为负样本 —— 不能全扔，见 §4.4
            if rng.random() >= neg_ratio:
                continue

        out.append({
            'instruction': INSTRUCTION % '、'.join(labels),
            'input':  render_dialogue(utts),
            'output': json.dumps({'records': records}, ensure_ascii=False),
        })

    print(f'{src}: 产出 {len(out)} 条 | intent 为空丢弃 {dropped_no_intent} 个QA对 '
          f'| 无记录对话 {dropped_empty} 场')
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--dst', required=True)
    ap.add_argument('--label-map', required=True, help='§5 产出的 210→12 映射')
    ap.add_argument('--neg-ratio', type=float, default=0.25)
    ap.add_argument('--n', type=int, default=0, help='0=全量；否则截断到 n 条（做样本量阶梯）')
    a = ap.parse_args()

    label_map = json.load(open(a.label_map, encoding='utf-8'))
    labels = sorted(set(label_map.values()))
    rows = convert(a.src, label_map, labels, a.neg_ratio)
    if a.n:
        random.Random(42).shuffle(rows)
        rows = rows[:a.n]
    Path(a.dst).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(a.dst, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'→ {a.dst} ({len(rows)} 条)')
```

### 4.4 三个转换环节的决定，比脚本本身重要

**① `solution` 空字符串必须转成 `null`，不能转成 `""`，更不能丢弃这条记录。**

你的痛点 #4 是「字段缺失、幻觉编造」。模型学会「没有方案就填 null」的唯一途径，是训练集里真的有 `null`。全都是非空的训练集会教出一个**永远编方案**的模型。这对应 [选型论证与评测方案.md](选型论证与评测方案.md) §B1 的三态值模型（present / null / MISSING）。

**② 负样本要留 20–30%。**

`--neg-ratio 0.25`。无可抽取记录的对话，输出 `{"records": []}`。全是正样本的训练集会让模型**在任何输入上都硬凑出记录**——这是 [执行流程.md](执行流程.md) §11 排第 3 的坑。

**③ `intent` 为空的 158 个 QA 对：丢弃 QA 对，但保留对话。**

丢 QA 对是因为没有分类标签就无法构造完整记录。保留对话是因为——如果那场对话只剩这些空 intent 的 QA，它就成了一个天然负样本，正好喂给 ② 。别把整场对话删掉。

### 4.5 注册到 LLaMA-Factory

`data/dataset_info.json` 加一项：

```json
{
  "csds_extract": {
    "file_name": "csds_train.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
}
```

文件要放在 LLaMA-Factory 的 `data/` 目录下（或用 `dataset_dir` 指定）。

---

## 5. Taxonomy 归并：146 类 → 12 类

这一步**不需要 GPU，但它是整条链路的前置门**。

[执行流程.md](执行流程.md) §11 排第 1 的坑就是「先训模型再定切分规则」。CSDS 的 146 个 intent 太细（尾部很多只出现 1–2 次），直接当标签集会让模型在相似类之间乱跳——**这正是你的痛点 #2 在公开数据上的复刻**。

做法（[TnT-LLM](https://arxiv.org/abs/2403.12173) 的简化版）：

```python
# 05_kb/induce_taxonomy.py（骨架）
# 1. 取 train split 全部 intent + 频次
# 2. 用 Qwen3-Embedding 编码 intent 字符串，层次聚类到 10-15 簇
# 3. 每个簇把成员 intent + 3 个样例 QA 对喂给 LLM，让它命名这一类
# 4. 人工过一遍簇名和边界 —— 这一步不能省
# 5. 落成 {原intent: 新类名} 的 label_map.json
```

产出 `00_spec/label_map.json`，供 §4.3 使用。

**为什么一定要压到 8–15 类**：分类是闭集任务，类别数直接决定每类的样本量。146 类 × 9101 对话，尾部类一个训练样本都没有；12 类的话每类平均几百条，才够学。

**两个硬性动作**：

1. **人工审簇名，不能全自动。** 聚类会把「配送周期」和「配送范围」合并（合理），也会把「价保申请流程」和「发票退换修改」合并（不合理）。LLM 命名同样会出错。这一步的人工成本是 1–2 小时，省不掉。
2. **把 `其他` 留着。** §4.3 的 `label_map.get(raw_intent, '其他')` 有兜底。别追求 100% 覆盖率去硬塞——`其他` 类的占比本身就是一个有用信号（超过 15% 说明 taxonomy 没归纳好）。

---

## 6. prompt-only 基线：先做这个，别先训模型

**基线是产出 3 的分母。没有它，训完的 F1 是个孤零零的数字，说明不了 SFT 值不值。**

同一张评测集、同一个约束解码后端、同一个 Qwen3-14B（**未微调**）：

```bash
# vLLM 起服务
python -m vllm.entrypoints.openai.api_server \
  --model /root/autodl-tmp/models/Qwen3-14B \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --guided-decoding-backend xgrammar
```

推理时传 `guided_json`（你的 schema），保证输出结构合法。

**基线要跑三档，不是一档**：

| 档位 | 配置 | 回答什么问题 |
|---|---|---|
| B0 | zero-shot + 约束解码 | 你现在的真实起点 |
| B1 | 5-shot + 约束解码 | 提示词工程的合理上限 |
| B2 | 20-shot + 约束解码 | 示例数还有没有边际收益 |

B1 → B2 的差值直接回答 §0 里那个 FsPONER / Bertsch 矛盾。如果 B2 ≈ B1，示例数在你的任务上确实饱和得早；如果 B2 明显更好，那 40 示例饱和的结论在长对话多记录任务上不成立。

> ⚠️ **schema 里绝对不要用 `$defs`**。这是 [执行流程.md](执行流程.md) §11 排第 4 的坑——模型会把 schema 定义本身回显到输出里。全部展平写。

---

## 7. 训练：样本量阶梯

### 7.1 生成四个档位的数据

```bash
for N in 50 200 800 3000; do
  python 06_data/csds_to_alpaca.py \
    --src data/csds/train.json \
    --dst LLaMA-Factory/data/csds_train_${N}.json \
    --label-map 00_spec/label_map.json \
    --neg-ratio 0.25 --n ${N}
done
```

**四个档位要用同一个随机种子的嵌套子集**（`random.Random(42).shuffle` 后截断保证了这点）——50 条必须是 200 条的子集。否则档位间的差异里混进了数据抽样噪声，曲线不可读。

### 7.2 4090 24GB：14B QLoRA 配置

```yaml
# 04_train/qwen3_14b_qlora_csds.yaml
model_name_or_path: /root/autodl-tmp/models/Qwen3-14B
trust_remote_code: true
quantization_bit: 4                 # ⚠️ 只因为 4090 只有 24GB；A800 上删掉这两行
quantization_method: bnb

stage: sft
do_train: true
finetuning_type: lora
lora_rank: 32                        # 小显存下的妥协，A800 上提到 64
lora_alpha: 64
lora_dropout: 0.05
lora_target: all

dataset: csds_extract                # 改 dataset_info.json 里的 file_name 切换档位
template: qwen3_nothink
cutoff_len: 2048                     # CSDS 平均 494 字，2048 够用
max_samples: 100000
preprocessing_num_workers: 8
overwrite_cache: true

output_dir: /root/autodl-tmp/saves/qwen3-14b/qlora/n3000
logging_steps: 5
save_steps: 50
plot_loss: true
overwrite_output_dir: true

per_device_train_batch_size: 1
gradient_accumulation_steps: 8
gradient_checkpointing: true
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
flash_attn: fa2

val_size: 0.1
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 50
```

**小样本档位要调 epoch**。50 条样本跑 3 epoch 只有约 19 个 optimizer step，模型基本没动。50 条建议 10–15 epoch，200 条 8 epoch，800 条 5 epoch，3000 条 3 epoch——**按「总 step 数大致可比」来配**，否则你测的是 step 数的影响，不是样本量的影响。

### 7.3 A800 80GB：改成与生产一致的 bf16 LoRA

```yaml
# 删掉 quantization_bit / quantization_method
lora_rank: 64
lora_alpha: 128
per_device_train_batch_size: 4
gradient_accumulation_steps: 2
```

这份配置与 [执行流程.md](执行流程.md) §7.1 的 H200 生产配置**只差 `cutoff_len` 和 batch size**，所以曲线可迁移。

### 7.4 批量推理

用 LLaMA-Factory 自带的：

```bash
python scripts/vllm_infer.py \
  --model_name_or_path /root/autodl-tmp/models/Qwen3-14B \
  --adapter_name_or_path /root/autodl-tmp/saves/qwen3-14b/qlora/n3000 \
  --dataset csds_eval \
  --template qwen3_nothink \
  --save_name /root/autodl-tmp/preds/n3000.jsonl
```

### 7.5 OOM 处理顺序（4090 24GB）

1. `gradient_checkpointing: true`（已开）
2. `per_device_train_batch_size: 1`（已是）
3. 降 `cutoff_len`（2048 → 1024，CSDS 够用）
4. `lora_rank` **最后才动**

`lora_rank` 放最后是因为它是**唯一限制模型容量的变量**——[LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673) 的核心结论就是 LoRA 的瓶颈在秩。为省显存先砍秩，等于先砍效果。

---

## 8. 评测：这是唯一不能将就的部分

### 8.1 先说一遍最重要的警告

**不要用 LLaMA-Factory 自带的 `eval_bleu_rouge.py` / `ComputeSimilarity` 选模型。**

它算的是 BLEU/ROUGE——对这个任务**是错的**。一个把「客服让用户提供号码，交给专员回电解决」抽成「客服让用户提供号码，无法解决」的输出，BLEU 很高但语义相反。你要的是字段级 P/R/F1、schema 合规率、**漏抽率**、**幻觉率**。

这是 [执行流程.md](执行流程.md) §11 排第 2 的坑，也是自建 harness 的全部理由。

### 8.2 harness 四个模块

按 [执行流程.md](执行流程.md) §3 的设计，直接复用：

```
03_eval/
├── parse.py      # JSON 解析 + schema 校验 → 产出 schema 合规率
├── align.py      # Hungarian 二分匹配，把预测记录对齐到金标记录
├── judge.py      # 字段级判定（分类=精确匹配；痛点/方案=语义+LLM judge）
├── metrics.py    # 三态 P/R/F1 + 漏抽率 + 幻觉率
└── run_eval.py   # 串起来，输出一张可比的表
```

核心是 `align()`——多记录任务不能按下标对齐，必须做二分匹配：

```python
from scipy.optimize import linear_sum_assignment
import numpy as np

def align(gold_records, pred_records, sim_fn, threshold=0.7):
    """会议级对齐。sim_fn 用 Qwen3-Embedding 算 pain_point 语义相似度。"""
    if not gold_records or not pred_records:
        return [], gold_records, pred_records
    sim = np.array([[sim_fn(g, p) for p in pred_records] for g in gold_records])
    rows, cols = linear_sum_assignment(-sim)
    matched, used_g, used_p = [], set(), set()
    for i, j in zip(rows, cols):
        if sim[i][j] >= threshold:
            matched.append((gold_records[i], pred_records[j]))
            used_g.add(i); used_p.add(j)
    missed   = [g for i, g in enumerate(gold_records) if i not in used_g]   # FN = 漏抽
    spurious = [p for j, p in enumerate(pred_records) if j not in used_p]   # FP = 潜在幻觉
    return matched, missed, spurious
```

`missed` 和 `spurious` 要**分开报**，不能合成一个 F1。你的场景里这两种错误代价完全不同：漏抽只是少一条经验，**幻觉是把编造的「专家经验」写进知识库**，会污染后续所有复用。

### 8.3 CSDS 上的额外便利

CSDS 有 `QueSummUttIDs` / `AnsSummLongUttIDs`，所以你可以额外算一个**证据接地准确率**（预测的 `evidence` 下标和金标的重叠度）。这在真实数据上要额外标，在这里是免费的。

它的用途是**区分两种幻觉**：指向了不存在发言的（硬幻觉），和指对了发言但概括错了的（软错误）。前者是 schema/接地问题，后者是理解问题，修法完全不同。

### 8.4 不要用 Qwen3-14B 当自己的判官

[执行流程.md](执行流程.md) §11 排第 5 的坑。LLM-as-judge 环节要用**不同家族**的模型（或闭源 API），并且做 **swap-and-average**（交换 A/B 位置各评一次取平均）+ 报 **Conflict Rate**。自己评自己会系统性偏高。

### 8.5 人类天花板

CSDS 的金标是人标的，但**「一场对话应该抽出几条记录」本身是 ill-posed 的**——[选型论证与评测方案.md](选型论证与评测方案.md) §B5 引的证据是专家在「仅仅数记录条数」上 Krippendorff α 只有 0.25–0.63。

所以：**在 CSDS test 上随机抽 30 条自己重标一遍**，算你和 CSDS 原标注者的一致性。这个数是你的**人类天花板**。模型 F1 接近它就该停了，再调是在拟合标注噪声。

这一步会花你 2–3 小时，但它决定了你后面所有数字怎么读。

---

## 9. 演练完，什么能搬走、什么不能

| 资产 | 能搬到真实数据吗 |
|---|---|
| AutoDL/LLaMA-Factory 环境流程 | ✅ 直接搬 |
| 训练 yaml | ✅ 改 `dataset` + `cutoff_len` 即可 |
| **评测 harness 全部代码** | ✅ 直接搬，这是最大收益 |
| taxonomy 归并脚本 | ✅ 脚本能搬，**label_map 不能**（CSDS 是电商客服，你是销售会议） |
| 「solution 空值 → null」「负样本 25%」这些数据决定 | ✅ 结论能搬 |
| **样本量 vs F1 曲线的拐点** | 🔶 **只能当量级参考**。CSDS 对话 494 字、你的上万字；CSDS 分类体系是别人标的 |
| 模型权重 | ❌ 完全不能用 |

### 建议第一轮不做的三件事

1. **不要跑 32B。** 演练的三个产出没有一个对模型尺寸敏感。等真实数据和 harness 都到位，在 H200 上一次跑 14B/32B 对照，比现在在 AutoDL 上花 ¥40 拿一个域外的尺寸结论有用。
2. **不要接 ASR。** 会让你无法区分抽取错误和转写错误。先用 CSDS 干净文本把抽取链路调对。
3. **不要跑 VCSum 粒度实验。** 那是第二轮的事——第一轮先让 harness 能稳定输出一张可比的表。

### 第二轮再做

- VCSum 上的「话题段 vs 整场会议」粒度对照（[执行流程.md](执行流程.md) §0 那个承重决策，目前只有推演没有数据）
- A800 上 bf16 LoRA 重跑样本量曲线（与生产配置一致）
- AliMeeting4MUG + 原始音频跑全链路，测 ASR 质量对下游抽取的影响

---

## 10. 这次演练最容易踩的坑（按危害排序）

1. **跳过 §5 taxonomy 归并，直接用 146 个 intent 训练。** 尾部类样本数为 0，分类指标会难看得没法解读，你会误以为是模型不行。
2. **跳过 §6 prompt-only 基线，直接训模型。** 训完拿到一个 F1=0.61，然后不知道这是好还是坏。基线必须先跑。
3. **用 BLEU/ROUGE 选 checkpoint。** LLaMA-Factory 默认给你这个，很容易顺手就用了。它对本任务是错的。
4. **HF 缓存没改路径，30GB 系统盘爆掉。** 下 14B 的时候爆，前面装的环境一起受影响。`export HF_HOME=/root/autodl-tmp/cache/` 写进 `.bashrc`。
5. **照 CSDS 官方文档写字段名。** 文档说 `Topic`，实际是 `intent`。脚本会静默产出全是 `其他` 的数据，训完才发现。**先跑 §4.2 的 inspect。**
6. **四个样本量档位用了不同的随机子集。** 曲线里混进抽样噪声，读不出趋势。
7. **小样本档位没调 epoch。** 50 条 × 3 epoch ≈ 19 步，等于没训，你会得出「50 条完全没用」的错误结论。
8. **训练命令没加 `tee` 就 `shutdown`。** 日志丢了，跑了 4 小时白费。

---

## 11. 起步顺序（前四步不用 GPU）

```
□ 1. 本地下 CSDS（69MB），跑 §4.2 inspect，把真实字段名抄下来
□ 2. §5 归并 taxonomy → 00_spec/label_map.json（人工审一遍簇名）
□ 3. §4.3 转换脚本跑通，肉眼检查 20 条产出：null 有没有、负样本有没有、evidence 下标对不对
□ 4. §8 写 harness，先用「金标 vs 金标」自测（P=R=F1=1.0 才算 harness 对）
□ 5. 租 4090，无卡模式装环境下模型（§3）
□ 6. §6 跑 B0/B1/B2 三档 prompt-only 基线
□ 7. §7 训 3000 条那档，确认全链路能出数
□ 8. 补齐 50/200/800 三档，画曲线
□ 9. §8.5 手标 30 条，算人类天花板
```

第 4 步的自测容易被跳过，但它是唯一能验证 harness 本身没写错的办法。**把金标当预测喂进去，如果不是满分，说明 harness 有 bug**——不要等到评测模型时才发现，那时你分不清是模型差还是 harness 错。

前四步全在本地做，**一分钱 GPU 费都不花，但占整个演练一半以上的时间**。这和 [执行流程.md](执行流程.md) §10 的关卡顺序是一致的：定义和评测先行。
