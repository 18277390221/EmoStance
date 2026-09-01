# EmoStance

[English](README.md)

以下论文的官方代码与数据重建资源：

> **EmoStance：基于 Emoji 弱监督的共情回复生成——回复侧情感倾向控制**  
> Ziyuan Jin、Yuxuan Ge、Zheng Tian†  
> 上海科技大学 · †通讯作者

[论文 PDF](paper/EmoStance.pdf) · [论文信息与摘要](docs/PAPER.md) · [复现指南](docs/REPRODUCTION.md)

![EmoStance 方法概览](assets/method_overview.png)

## 项目概述

EmoStance 将多标注模型产生的 Emoji 分布视为弱情感—态度证据，而不是输出符号、金标准情绪标签或金标准倾听者立场标签。该方法构建一个无命名的潜在倾向空间，根据对话上下文和说话人角色预测回复侧的软倾向分布，通过聚类原型重建连续控制向量，并利用学习得到的前缀嵌入控制冻结的指令微调语言模型。推理时只需要对话文本和说话人角色。

本仓库包含：

- 仅使用训练集构建的无命名 Emoji 图与聚类流程；
- 回复倾向数据准备、预测、原型重建和消融实验；
- 面向 `mistralai/Mistral-7B-Instruct-v0.3` 的连续前缀控制训练；
- 基于多个候选回复的倾向一致性重排序；
- 自动评估、人工评估、数据审计和效率测试脚本；
- 已发布的 Emoji—潜在区域隶属度与 Emoji 质心资源；
- 不含文本的 EmojiDialogue 标注元数据及其重建脚本。

模型权重、EmpatheticDialogues 原始文本、私有标注导出文件以及实验运行目录均未包含在仓库中。

## 仓库结构

```text
EmoStance/
├── assets/                         # 方法图（PNG 与源 PDF）
├── configs/                        # 论文实验配置
├── data/annotation_metadata/       # 已发布的无文本 EmojiDialogue 元数据
├── docs/                           # 论文、数据与复现说明
├── examples/                       # 合成的冒烟测试输入
├── human_ablation/                 # 聚焦式盲测成对人工评估脚本
├── human_llm_emoji_audit/          # 人类与 LLM 分布一致性审计脚本
├── data_eval/                      # 弱标签合理性审计脚本
├── rerank_efficiency/              # B=1 与 B=4 的耗时基准测试
├── system_baseline/                # 对齐的系统级评估框架
├── scripts/                        # 数据重建与消融实验工具
├── src/latent_stance_control/      # EmoStance 训练与生成代码
└── src/name_free_emoji_clustering/ # 弱监督目标构建代码
```

## 安装

论文实验环境使用 Python 3.10.14、PyTorch 2.6.0（CUDA 11.8）和 Transformers 4.46.3。请先安装适合本机平台的 PyTorch，然后安装本项目：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[clustering,evaluation]"
```

如只需在 CPU 上进行轻量代码检查：

```bash
python -m pip install "numpy==2.2.6" "networkx==3.4.2" "pytest>=8"
PYTHONPATH=src pytest -q tests src/name_free_emoji_clustering/tests human_llm_emoji_audit/tests
```

## 快速冒烟测试

以下合成示例不包含 EmpatheticDialogues 文本，也不会下载模型：

```bash
python -m latent_stance_control.prepare_data \
  --annotations examples/tiny_annotations.jsonl \
  --clusters examples/tiny_clusters.json \
  --out runs/smoke/prepared
```

命令应在 `runs/smoke/prepared` 下生成 `train.jsonl`、`dev.jsonl`、`meta.json` 和 `prepare_summary.json`。

## 重建 EmojiDialogue

`data/annotation_metadata/` 下发布的文件只包含对话标识符、轮次索引、Emoji 投票和置信度分数。请按照原始许可条款单独获取 EmpatheticDialogues，然后在本地将其与元数据合并：

```bash
python scripts/reconstruct_emojidialogue.py \
  --metadata-root data/annotation_metadata \
  --ed-root /path/to/empatheticdialogues \
  --output-root private_data/reconstructed
```

请勿提交 `private_data/reconstructed`，其中包含原始对话文本。数据格式与发布规则见 [docs/DATA.md](docs/DATA.md)。

## 主要训练流程

完成数据重建后，整体流程如下：

```bash
# 仅使用训练集构建无命名潜在空间。
python -m name_free_emoji_clustering \
  --root private_data/reconstructed \
  --output-dir runs/main/clustering \
  --cluster-splits train

python -m name_free_emoji_clustering.soft_membership \
  --artifact runs/main/clustering/cluster_visualization.html \
  --output-dir runs/main/clustering/soft_membership

# 构建相邻轮次的源倾向/回复倾向目标。
python -m latent_stance_control.prepare_data \
  --annotation-root private_data/reconstructed \
  --clusters runs/main/clustering/soft_membership/emoji_cluster_membership.csv \
  --emoji-vectors runs/main/clustering/tables/emoji_centroids.csv \
  --out runs/main/prepared

# 训练角色感知的倾向预测器。
python -m latent_stance_control.train_role_aware_stance_predictor \
  --prepared runs/main/prepared \
  --out runs/main/stance_role_aware \
  --model microsoft/deberta-v3-base \
  --epochs 3 --batch-size 8 --lr 1.5e-5 --max-length 320 \
  --focal-gamma 0 --class-weight-power 0.25

# 拟合原型重建并评估倾向消融实验。
python -m latent_stance_control.run_ablations \
  --prepared runs/main/prepared \
  --stance-dir runs/main/stance_role_aware \
  --out runs/main/ablations_role_aware
```

生成器训练与解码需要 7B 参数的 Mistral 主干模型及合适的 GPU。完整命令、c7 门控准备、三随机种子生成和重排序流程见 [docs/REPRODUCTION.md](docs/REPRODUCTION.md)。论文使用的超参数记录在 [configs/emostance_main.json](configs/emostance_main.json) 中。

## 数据与资源使用规则

- 代码采用 [MIT License](LICENSE) 发布。
- 已发布的 EmojiDialogue 元数据与派生倾向资源仅限非商业研究使用，具体见与 EmpatheticDialogues 原始限制兼容的[数据专用条款](data/DATA_LICENSE.md)。
- 本标注层属于弱监督信号，不得将其视为金标准情绪标注、诊断结果、受保护属性推断、用户画像或说话人真实内在状态的证据。
- 本仓库不包含 API 密钥、模型权重、原始对话文本或原始 API 日志。

## 引用

```bibtex
@misc{jin2026emostance,
  title  = {EmoStance: Response-Side Affective-Orientation Control for Empathetic Response Generation via Emoji Weak Supervision},
  author = {Jin, Ziyuan and Ge, Yuxuan and Tian, Zheng},
  year   = {2026},
  url    = {https://github.com/18277390221/EmoStance}
}
```

机器可读的引用信息见 [CITATION.cff](CITATION.cff)。
