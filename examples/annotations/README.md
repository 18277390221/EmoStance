# 原始格式示例数据

这个目录用于理解主线代码的数据流。结构刻意模仿真实数据目录：

```text
original_format_data/
├── gpt-5.4/
│   ├── train_emoji_annotations.json
│   ├── valid_emoji_annotations.json
│   └── test_emoji_annotations.json
├── gemini-2.5-pro/
├── claude-sonnet-4-6/
└── DeepSeek-V3.2/
```

每个 JSON 文件都是一个 dialogue 列表，每个 dialogue 内部包含：

```text
dialogue_id
split
situation
num_turns
speakers
turns
```

每个 turn 内部包含：

```text
turn_id
speaker
utterance
emoji_annotation.selected_emoji
emoji_annotation.confidence
```

这里的 `confidence` 使用真实项目的五档整数：

```text
1 = very uncertain
2 = somewhat uncertain
3 = moderate
4 = fairly confident
5 = very confident
```

可以用下面命令跑通示例 prepared 数据：

```bash
cd .

uv run python -m latent_stance_control.prepare_data \
  --annotation-root examples/original_format_data \
  --clusters examples/tiny_clusters.json \
  --out /tmp/mainline_original_format_prepared
```

