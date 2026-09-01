# Reranking efficiency benchmark

`scripts/measure_rerank_efficiency.py` compares the deployable single-generation path (`B=1`) against four-candidate orientation-consistency reranking (`B=4`) on identical inputs and decoding settings. It records end-to-end latency, throughput, candidate-generation time, stance-scoring time, and final-selection time.

```bash
python rerank_efficiency/scripts/measure_rerank_efficiency.py \
  --project-root . --out-dir rerank_efficiency \
  --num-examples 512 --split test --seed 13 \
  --max-new-tokens 64 --temperature 0.7 --top-p 0.9 \
  --do-sample --bf16
```

Paper timings use a single NVIDIA RTX 4090. CPU runs are wiring checks only and must not be reported as comparable latency results.
