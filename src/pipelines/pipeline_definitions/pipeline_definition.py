# pipeline_definitions.py
from kfp import dsl
from helpers import pull_data
from helpers import preprocess
from helpers import train
from helpers import evaluate
from helpers import deploy_model
from helpers import inference

@dsl.pipeline(
    name='demo-pipeline',
    description='An example pipeline for wine quality prediction.'
)
def pipeline(
    url: str,
    target: str,
    mlflow_experiment_name: str,
    mlflow_tracking_uri: str,
    mlflow_s3_endpoint_url: str,
    model_name: str,
    alpha: float,
    l1_ratio: float,
    threshold_metrics: dict,
):
    pull_task = pull_data(url=url)

    preprocess_task = preprocess(data=pull_task.outputs["data"])

    train_task = train(
        train_set=preprocess_task.outputs["train_set"],
        test_set=preprocess_task.outputs["test_set"],
        target=target,
        mlflow_experiment_name=mlflow_experiment_name,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_s3_endpoint_url=mlflow_s3_endpoint_url,
        model_name=model_name,
        alpha=alpha,
        l1_ratio=l1_ratio
    )

    evaluate_task = evaluate(
        run_id=train_task.outputs["run_id"],
        mlflow_tracking_uri=mlflow_tracking_uri,
        threshold_metrics=threshold_metrics
    )

    eval_passed = evaluate_task.output

    with dsl.Condition(eval_passed == "true"):
        deploy_model_task = deploy_model(
            model_name=model_name,
            storage_uri=train_task.outputs["storage_uri"],
        )

        inference_task = inference(
            model_name=model_name,
            scaler_in=preprocess_task.outputs["scaler_out"]
        )
        inference_task.after(deploy_model_task)
