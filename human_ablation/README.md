# Focused human ablation

`scripts/make_human_ablation.py` builds randomized, blinded A/B pages comparing the final reranked system against no-reranking, no-role-aware, and zero-control variants. `scripts/analyze_human_ablation_results.py` decodes completed anonymous exports and reports decisive win rates, Wilson intervals, agreement, and sign tests.

Input generations and participant exports are intentionally excluded. Inspect `--help` for explicit artifact paths:

```bash
python human_ablation/scripts/make_human_ablation.py --help
python human_ablation/scripts/analyze_human_ablation_results.py --help
```

Tie/both-equally-good and neither/both-bad remain neutral and are excluded from decisive win-rate denominators.
