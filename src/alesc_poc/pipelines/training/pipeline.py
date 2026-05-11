"""
Training pipeline: runs per-category model fitting.

Because Kedro doesn't natively loop over dynamic keys, the pipeline operates
on a single `category_key` parameter. Run with:
  kedro run --pipeline training --params category_key:DIARIAS
Or use the orchestrator in pipeline_registry.py to loop over all categories.
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    build_ensemble,
    estimate_contamination,
    save_model_bundle,
    train_gmm,
    train_iforest,
    train_knn,
    train_lof,
    train_ocsvm,
    train_sod,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=estimate_contamination,
                inputs=[
                    "category_train",
                    "params:contamination_iqr_multiplier",
                    "params:contamination_min",
                    "params:contamination_max",
                ],
                outputs="contamination_rate",
                name="estimate_contamination",
            ),
            node(
                func=train_iforest,
                inputs=["category_train", "contamination_rate", "params:random_seed"],
                outputs="iforest_bundle",
                name="train_iforest",
            ),
            node(
                func=train_knn,
                inputs=[
                    "category_train", "contamination_rate",
                    "params:knn_k_candidates", "params:random_seed",
                ],
                outputs="knn_bundle",
                name="train_knn",
            ),
            node(
                func=train_lof,
                inputs=[
                    "category_train", "contamination_rate",
                    "params:lof_k_candidates",
                    "params:lof_subsample_threshold",
                    "params:lof_subsample_size",
                    "params:random_seed",
                ],
                outputs="lof_bundle",
                name="train_lof",
            ),
            node(
                func=train_gmm,
                inputs=[
                    "category_train", "contamination_rate",
                    "params:gmm_max_components", "params:random_seed",
                ],
                outputs="gmm_bundle",
                name="train_gmm",
            ),
            node(
                func=train_ocsvm,
                inputs=[
                    "category_train", "contamination_rate",
                    "params:ocsvm_subsample_threshold",
                    "params:ocsvm_subsample_size",
                    "params:ocsvm_nu",
                    "params:random_seed",
                ],
                outputs="ocsvm_bundle",
                name="train_ocsvm",
            ),
            node(
                func=train_sod,
                inputs=[
                    "category_train", "contamination_rate",
                    "params:random_seed", "params:include_sod",
                ],
                outputs="sod_bundle",
                name="train_sod",
            ),
            node(
                func=build_ensemble,
                inputs=[
                    "category_eval",   # scored on test or infer set
                    "iforest_bundle", "knn_bundle", "lof_bundle",
                    "gmm_bundle", "ocsvm_bundle", "sod_bundle",
                    "params:ensemble_strategy",
                ],
                outputs="ensemble_predictions",
                name="build_ensemble",
            ),
        ]
    )
