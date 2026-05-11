from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    correct_inflation,
    engineer_features,
    load_ipca_factors,
    split_by_category,
    split_train_test_infer,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_ipca_factors,
                inputs="params:raw_data_dir",  # resolved inside node via join
                outputs="ipca_factors",
                name="load_ipca_factors",
            ),
            node(
                func=correct_inflation,
                inputs=["intermediate_expenses", "ipca_factors"],
                outputs="adjusted_expenses",
                name="correct_inflation",
            ),
            node(
                func=split_train_test_infer,
                inputs=[
                    "adjusted_expenses",
                    "params:train_years",
                    "params:test_years",
                    "params:inference_years",
                ],
                outputs=["train_raw", "test_raw", "infer_raw", "reversals"],
                name="split_train_test_infer",
            ),
            node(
                func=engineer_features,
                inputs=[
                    "train_raw",
                    "test_raw",
                    "infer_raw",
                    "params:hash_dim_favorecido",
                    "params:mean_value_group",
                ],
                outputs=["train_features", "test_features", "infer_features"],
                name="engineer_features",
            ),
            node(
                func=split_by_category,
                inputs=[
                    "train_features",
                    "test_features",
                    "infer_features",
                    "params:min_rows_train",
                ],
                outputs="category_splits",
                name="split_by_category",
            ),
        ]
    )
