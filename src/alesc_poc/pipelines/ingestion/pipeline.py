from kedro.pipeline import Pipeline, node, pipeline

from .nodes import export_intermediate, load_raw_expenses, parse_valor, tag_reversal


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_raw_expenses,
                inputs="params:raw_data_dir",
                outputs="raw_expenses",
                name="load_raw_expenses",
            ),
            node(
                func=parse_valor,
                inputs="raw_expenses",
                outputs="parsed_expenses",
                name="parse_valor",
            ),
            node(
                func=tag_reversal,
                inputs="parsed_expenses",
                outputs="tagged_expenses",
                name="tag_reversal",
            ),
            node(
                func=export_intermediate,
                inputs="tagged_expenses",
                outputs="intermediate_expenses",
                name="export_intermediate",
            ),
        ]
    )
