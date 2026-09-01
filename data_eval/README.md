# Weak-label plausibility audit

The scripts in this directory sample model–dialogue packages, build offline annotation pages, validate exports, and aggregate three-way judgments (`reasonable`, `questionable but acceptable`, and `clearly unreasonable`). The task audits contextual plausibility; it is not gold emotion labeling and does not infer a speaker's true mental state.

```bash
python data_eval/build_human_audit.py --help
python data_eval/validate_audit_outputs.py --help
python data_eval/aggregate_human_audit.py --help
```

Generated pages, translations, participant exports, and text-bearing sampled packages are intentionally excluded.
