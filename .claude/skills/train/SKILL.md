---
name: train
description: Run training pipeline. Usage: /train [--category X] [--years YYYY-YYYY] [--force]
---

Parse arguments from the user message.
Check if `outputs/models/{CATEGORY}/iforest.joblib` already exists (incremental).
If model exists and `--force` not passed, skip training and report "already trained".

Run:
  .venv/bin/python3 -m alesc_poc.pipeline_runner [--category X] [--years YYYY-YYYY]

After completion, report:
- Which categories were trained
- How many anomalies were flagged per category
- Where results are saved
