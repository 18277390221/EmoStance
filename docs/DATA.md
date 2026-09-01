# Data and reconstruction

## What is released

`data/annotation_metadata/<model>/<split>.jsonl.gz` contains one record per annotated dialogue:

```json
{
  "dialogue_id": "hit:0_conv:1",
  "split": "train",
  "turn_annotations": [
    {
      "turn_id": 0,
      "speaker": "A",
      "selected_emoji": "🥰",
      "unicode": "U+1F970",
      "confidence": 4
    }
  ]
}
```

The files do not contain utterances, situations, emotion labels, names, contact details, raw API responses, or provider account metadata. The four model directories correspond to the weak annotators described in the paper.

The repository also includes two small derived artifacts used by the default preparation command:

- `src/name_free_emoji_clustering/outputs/soft_membership/emoji_cluster_membership.csv`
- `src/name_free_emoji_clustering/outputs/tables/emoji_centroids.csv`

## Local reconstruction

1. Obtain the official EmpatheticDialogues `train.csv`, `valid.csv`, and `test.csv` files under their original license.
2. Run:

```bash
python scripts/reconstruct_emojidialogue.py \
  --metadata-root data/annotation_metadata \
  --ed-root /path/to/empatheticdialogues \
  --output-root private_data/reconstructed
```

3. Keep the reconstructed text-bearing files local. `private_data/` is ignored by Git.

The reconstruction script checks dialogue identifiers, turn bounds, speaker roles, and annotation completeness. The resulting model-specific files are accepted by `latent_stance_control.prepare_data` and the name-free clustering pipeline.

## Regenerating the release metadata

Authors with access to the private text-bearing annotation files can regenerate the released metadata deterministically:

```bash
python scripts/export_annotation_metadata.py \
  --source-root /path/to/private/annotations \
  --output-root data/annotation_metadata
```

The exporter rejects text-bearing keys and writes a manifest with per-file record counts and SHA-256 hashes over the uncompressed JSONL content.

## Intended use

EmojiDialogue is intended for research on weakly supervised affective-orientation modeling and empathetic response generation. Emoji annotations and derived regions are weak contextual signals, not gold emotion or mental-state labels. See [data/DATA_LICENSE.md](../data/DATA_LICENSE.md) for terms.
