from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    compute_shap_values,
    extract_top_shap_features,
    plot_shap_summary,
    plot_waterfall_charts,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=compute_shap_values,
                inputs=["ensemble_predictions", "iforest_bundle", "params:shap_max_records"],
                outputs="shap_bundle",
                name="compute_shap_values",
            ),
            node(
                func=plot_waterfall_charts,
                inputs=[
                    "shap_bundle",
                    "params:category_key",
                    "params:top_anomalies_per_category",
                ],
                outputs="waterfall_paths",
                name="plot_waterfall_charts",
            ),
            node(
                func=plot_shap_summary,
                inputs=["shap_bundle", "params:category_key"],
                outputs="shap_summary_path",
                name="plot_shap_summary",
            ),
            node(
                func=extract_top_shap_features,
                inputs="shap_bundle",
                outputs="shap_top_features",
                name="extract_top_shap_features",
            ),
        ]
    )
