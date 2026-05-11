---
name: report
description: Generate article draft sections. Usage: /report [--section dataset|results|discussion|all]
---

Read `outputs/results/all_metrics.json` and the anomaly CSVs.
Invoke the `report-writer` agent to draft the requested section.

Default (no --section arg): generate all three sections.
Save outputs to `outputs/reports/section_{name}.md`.

After writing, summarize: how many categories had anomalies, top flagged records, main findings.
