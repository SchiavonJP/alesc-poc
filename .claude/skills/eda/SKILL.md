---
name: eda
description: Run EDA profiling on ALESC expense data. Usage: /eda [--category X] [--year Y]
---

Parse arguments from the user message for `--category` and `--year`/`--years`.
Then invoke the `data-profiler` agent with those arguments.

If no arguments provided, run EDA on all categories for all years.
If `--category` is specified, run only that category.
If `--year YYYY` is specified, filter to that single year.
If `--years YYYY-YYYY` is specified, filter to that range.

Always save output to `outputs/reports/eda_{category}_{years}.html` and
`outputs/results/eda_{category}_summary.json`.
