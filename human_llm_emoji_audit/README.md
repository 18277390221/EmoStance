# Human–LLM emoji-distribution audit

This directory implements the paper's direct comparison between four weak LLM annotators and three independent human annotators. It performs deterministic stratified sampling, builds offline questionnaires, validates anonymous exports, computes emoji/region distribution agreement, and runs the row-permutation region baseline.

```bash
python human_llm_emoji_audit/scripts/build_experiment.py --help
python human_llm_emoji_audit/scripts/validate_experiment.py --help
python human_llm_emoji_audit/scripts/validate_exports.py --help
python human_llm_emoji_audit/scripts/analyze_agreement.py --help
python human_llm_emoji_audit/scripts/permutation_region_baseline.py --help
```

Questionnaire payloads, private manifests, participant exports, and reports are generated locally and ignored by Git. Do not commit real names, contact information, API logs, timestamps tied to identities, or provider account metadata.
