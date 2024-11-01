# evaluate_component.py
from kfp.v2.dsl import component

@component(
    base_image="python:3.10",
    packages_to_install=["numpy", "mlflow~=2.4.1"],
    output_component_file='components/evaluate_component.yaml',
)
def evaluate(run_id: str, mlflow_tracking_uri: str, threshold_metrics: dict) -> bool:
    """
    Evaluate component.
    """
    from mlflow.tracking import MlflowClient
    import logging

    logging.basicConfig(level=logging.INFO)
    client = MlflowClient(tracking_uri=mlflow_tracking_uri)
    info = client.get_run(run_id)
    training_metrics = info.data.metrics

    for key, value in threshold_metrics.items():
        if key not in training_metrics or training_metrics[key] > value:
            return False
    return True
