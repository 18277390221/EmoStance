# Reproducing EmoStance

Run commands from the repository root. Paths below keep private data and large outputs outside the tracked source tree.

## 1. Reconstruct and prepare data

```bash
python scripts/reconstruct_emojidialogue.py \
  --metadata-root data/annotation_metadata \
  --ed-root /path/to/empatheticdialogues \
  --output-root private_data/reconstructed

python -m name_free_emoji_clustering \
  --root private_data/reconstructed \
  --output-dir runs/main/clustering \
  --cluster-splits train \
  --tau 50 --knn-k 8 --embedding-dim 256 \
  --lambda-ctx 0.65 --lambda-conf 0.35

python -m name_free_emoji_clustering.soft_membership \
  --artifact runs/main/clustering/cluster_visualization.html \
  --output-dir runs/main/clustering/soft_membership \
  --temperature 0.7

python -m latent_stance_control.prepare_data \
  --annotation-root private_data/reconstructed \
  --clusters runs/main/clustering/soft_membership/emoji_cluster_membership.csv \
  --emoji-vectors runs/main/clustering/tables/emoji_centroids.csv \
  --out runs/main/prepared
```

All emoji-derived quantities used by the model must be constructed from the training split. Do not include development or test utterances in the clustering command.

## 2. Train stance modules

```bash
python -m latent_stance_control.train_stance_predictor \
  --prepared runs/main/prepared \
  --out runs/main/stance_baseline \
  --model microsoft/deberta-v3-base \
  --epochs 3

python -m latent_stance_control.train_role_aware_stance_predictor \
  --prepared runs/main/prepared \
  --out runs/main/stance_role_aware \
  --model microsoft/deberta-v3-base \
  --epochs 3 --batch-size 8 --lr 1.5e-5 --max-length 320 \
  --focal-gamma 0 --class-weight-power 0.25 \
  --transition-alpha 0.05 --graph-prior-weight 0.5

python -m latent_stance_control.run_ablations \
  --prepared runs/main/prepared \
  --stance-dir runs/main/stance_role_aware \
  --out runs/main/ablations_role_aware

python -m latent_stance_control.make_predicted_control_prepared \
  --prepared runs/main/prepared \
  --stance-dir runs/main/stance_role_aware \
  --cluster-prototypes runs/main/ablations_role_aware/cluster_prototypes.json \
  --out runs/main/prepared_predicted_control
```

The role and hard-label ablations are available through `--disable-role-features` and `--hard-label-training` on `train_role_aware_stance_predictor`.

## 3. Optional c7 gate

The small-cluster gate is an optional preparation stage used by the main experimental pipeline:

```bash
python -m latent_stance_control.train_c7_gate \
  --prepared runs/main/prepared \
  --out runs/main/c7_gate

python -m latent_stance_control.apply_c7_gate_prepared \
  --prepared runs/main/prepared_predicted_control \
  --stance-dir runs/main/stance_role_aware \
  --gate-dir runs/main/c7_gate \
  --cluster-prototypes runs/main/ablations_role_aware/cluster_prototypes.json \
  --out runs/main/prepared_predicted_control_gated \
  --threshold 0.75 --active-prototype-mix 0.50
```

Use `runs/main/prepared_predicted_control` directly if the optional gate is not being reproduced.

## 4. Train the continuous prefix controller

The frozen Mistral backbone is memory intensive. The reported run uses bf16 and trains the prefix projector for one epoch.

```bash
python -m latent_stance_control.train_generator \
  --prepared runs/main/prepared \
  --out runs/main/generator \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --epochs 1 --lr 1e-4 --freeze-lm --bf16
```

## 5. Generate and rerank

Repeat stochastic evaluation with seeds 13, 21, and 42. The example below uses the gated prepared data; replace it with the ungated path if needed.

```bash
python -m latent_stance_control.generate_and_score_controls \
  --prepared runs/main/prepared \
  --predicted-prepared runs/main/prepared_predicted_control_gated \
  --generator-dir runs/main/generator \
  --stance-dir runs/main/stance_baseline \
  --out runs/main/control_seed13 \
  --splits dev,test --max-examples 512 --seed 13 \
  --max-new-tokens 64 --do-sample --bf16 --device cuda

python -m latent_stance_control.generate_and_rerank \
  --prepared runs/main/prepared \
  --predicted-prepared runs/main/prepared_predicted_control_gated \
  --generator-dir runs/main/generator \
  --stance-dir runs/main/stance_baseline \
  --out runs/main/rerank_seed13 \
  --splits dev,test --max-examples 512 --num-candidates 4 --seed 13 \
  --max-new-tokens 64 --bf16 --device cuda
```

The paper uses temperature 0.7, top-p 0.9, maximum 64 new tokens, four reranking candidates, and no length penalty. Reference-conditioned controls and reference-conditioned selection are upper-reference diagnostics, not deployable systems.

## 6. Evaluation

- `scripts/`: automatic stance, vector, and generation-control ablations.
- `system_baseline/`: full-coverage aligned system comparison.
- `human_ablation/`: focused blind pairwise ablations.
- `human_llm_emoji_audit/`: human–LLM distribution audit.
- `data_eval/`: weak-label plausibility audit.
- `rerank_efficiency/`: B=1 versus B=4 latency benchmark.

Each directory contains a short README. Raw participant exports and text-bearing evaluation material are intentionally excluded.
