# 销售会议转写 → 结构化表格抽取

从中文销售-客户会议录音转写中抽取结构化表格，并跨会议沉淀为专家知识库。

**目标 schema**

| 列 | 说明 |
|---|---|
| 所属分类 | 问题归属的类别标签 |
| 要解决的痛点问题 | 客户提出的具体问题 |
| 解决方案（沉淀为专家经验） | 对应的解法；没有则为 `null` |

**要解决的四个痛点**

1. 抽取不准 / 漏抽
2. 分类标签体系不稳定，同一问题被打到不同类
3. 长录音转写（上万字、多轮、口语噪声、说话人混淆）处理不了
4. 输出 JSON / schema 不守规矩、字段缺失、幻觉编造

---

## 文档导航

按阅读顺序：

| 文档 | 什么时候看 |
|---|---|
| [概念解释.md](概念解释.md) | **先看这个。** 从零解释 SFT / LoRA / 约束解码 / 为什么不能用 BLEU，全是类比不讲公式 |
| [销售会议转写结构化抽取-技术调研.md](销售会议转写结构化抽取-技术调研.md) | 技术路线调研，每条主张带论文出处（arXiv 号 / 链接） |
| [选型论证与评测方案.md](选型论证与评测方案.md) | 为什么选 Qwen3 + LoRA + LLaMA-Factory；评测指标怎么设计 |
| [执行流程.md](执行流程.md) | 生产环境（H200 141GB）的分阶段落地流程 |
| [AutoDL实战演练.md](AutoDL实战演练.md) | 用公开数据集在 AutoDL 上跑一次完整演练的方案与成本估算 |
| [操作手册.md](操作手册.md) | **动手时看这个。** 20 步逐步可操作清单，含全部脚本代码 |

---

## 技术选型

- **模型**：Qwen3-32B LoRA（r=64）为主，14B 作快速迭代档
- **训练**：LLaMA-Factory，Alpaca 格式，`template: qwen3_nothink`
- **推理**：vLLM + XGrammar 约束解码，`guided_json` 强制 schema 合规
- **评测**：匈牙利二分图匹配对齐 + 字段级 P/R/F1，区分漏抽率与幻觉率

评测**不用** BLEU/ROUGE——"交给专家回电解决"和"无法解决"的 n-gram 重叠很高但语义相反，理由见 [概念解释.md](概念解释.md) 第 8 节。

---

## 目录结构

```
00_spec/     schema 定义、字段名探查、标签体系工具
01_raw/      原始数据集（不入库，见下方下载说明）
02_gold/     人工校对的黄金样本
03_eval/     评测 harness
04_train/    训练配置与产物（产物不入库）
05_kb/       跨会议知识库构建
06_data/     格式转换后的训练数据
logs/        运行日志
```

---

## 数据集下载

原始数据集**不包含在本仓库**，需自行下载到对应目录。

### CSDS（主力代理数据集）

中文客服对话摘要数据集，带真实标注，与目标 schema 近似同构：

| CSDS 字段 | 映射到 |
|---|---|
| `intent` | 所属分类 |
| `QueSumm` | 要解决的痛点问题 |
| `AnsSummLong` | 解决方案 |
| `QueSummUttIDs` / `AnsSummLongUttIDs` | evidence（证据句下标） |

- 论文：[CSDS: A Fine-grained Chinese Dataset for Customer Service Dialogue Summarization](https://arxiv.org/abs/2108.13139) (EMNLP 2021)
- 仓库：<https://github.com/xiaolinAndy/CSDS>
- 下载后放到 `01_raw/csds/`，需要 `train.json` / `val.json` / `test.json`

> ⚠️ CSDS 原仓库未附 LICENSE 文件，使用前请自行确认授权范围。

### VCSum（长文本场景）

用于压测痛点 3（长录音转写），239 场会议、平均 14,462 字：

- 论文：[VCSum: A Versatile Chinese Meeting Summarization Dataset](https://arxiv.org/abs/2305.05280) (ACL 2023)
- 许可：MIT
- 放到 `01_raw/vcsum/`

---

## 快速开始

```bash
# 1. 确认 CSDS 已下载到 01_raw/csds/
# 2. 探查字段名（会自动猜测并写入 00_spec/字段名.txt）
python 00_spec/inspect_csds.py
```

完整的 20 步流程见 [操作手册.md](操作手册.md)。两个**硬关卡**不要跳过：

- **步骤 6**：数据健康度四项指标（负样本占比 20–30%、solution 为 null 超 5%、类别数对得上、输入长度合理）
- **步骤 8**：评测 harness 自检必须 100% 通过

---

## 已知局限

- 演练用的是公开代理数据，**训练出的权重不能直接用在真实业务数据上**
- CSDS 对话偏短（25.1 轮 / 494 字），**不足以压测痛点 3**，长文本要靠 VCSum
- CSDS 官方文档字段名有误（写作 `Topic`，实际是 `intent`），这是 `inspect_csds.py` 存在的原因
