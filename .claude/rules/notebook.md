# Notebook Rules

## Exploration Only
- Notebooks live in `notebooks/` and are never imported by `src/`.
- Production code must live in `src/alesc_poc/pipelines/`.
- If you find yourself copy-pasting notebook code into a pipeline, refactor into `nodes.py`.

## Output Hygiene
- Clear all outputs before committing: `jupyter nbconvert --clear-output --inplace notebooks/*.ipynb`
- Notebooks with outputs will be flagged by the post-notebook-edit hook.

## No Side Effects
- Notebooks must not write to `data/01_raw/` or `outputs/models/`.
- Use `data/05_model_output/` or local temp paths for notebook exploration artifacts.
