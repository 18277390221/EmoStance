# Aligned system-level evaluation

This harness builds one canonical EmpatheticDialogues test set, aligns every system by `example_id`, generates responses through adapters, and computes automatic metrics only for full-coverage systems. Partial or legacy artifacts stay diagnostic and are not mixed into the main table.

Configure local model, adapter, baseline, and EmpatheticDialogues paths in `configs/system_eval.yaml`, then run:

```bash
bash system_baseline/scripts/run_all.sh
```

Full 7B inference is disabled by default. Use the generation command's explicit full-run flag only after reviewing local paths and compute requirements. The original baseline repositories and checkpoints are not redistributed here.
