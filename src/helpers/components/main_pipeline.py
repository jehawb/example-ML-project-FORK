# main_pipeline.py
from kfp.v2.dsl import pipeline, Input, Output, Dataset, Artifact
from pull_data_component import pull_data
from preprocess_component import preprocess
from train_component import train
from evaluate_component import evaluate
from deploy_model_component import deploy_model
from inference_component import inference
import kfp
from pipelines import arguments  # Ensure this imports your arguments

@pipeline(name="Wine Quality Prediction Pipeline")
def main_pipeline(url: str, 
                  mlflow_experiment_name: str,
                  mlflow_tracking_uri: str,
                  mlflow_s3_endpoint_url: str,
                  model_name: str,
                  alpha: float, 
                  l1_ratio: float,
                  threshold_metrics: dict):
    """
    Define the steps of the pipeline.
    """
    # Pull data component
    pull_task = pull_data(url=url)
    
    # Preprocess data component
    preprocess_task = preprocess(data=pull_task.outputs["data"])
    
    # Train model component
    train_task = train(
        train_set=preprocess_task.outputs["train_set"],
        test_set=preprocess_task.outputs["test_set"],
        mlflow_experiment_name=mlflow_experiment_name,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_s3_endpoint_url=mlflow_s3_endpoint_url,
        model_name=model_name,
        alpha=alpha,
        l1_ratio=l1_ratio
    )
    
    # Evaluate model component
    eval_task = evaluate(
        run_id=train_task.outputs["run_id"],
        mlflow_tracking_uri=mlflow_tracking_uri,
        threshold_metrics=threshold_metrics
    )
    
    # Deploy model if evaluation passes
    with eval_task.output:
        deploy_task = deploy_model(
            model_name=model_name,
            storage_uri=train_task.outputs["storage_uri"]
        )
    
    # Inference task
    inference_task = inference(
        model_name=model_name,
        scaler_in=preprocess_task.outputs["scaler_out"]
    )

if __name__ == "__main__":
    # You can pass in arguments here
    kfp.Client().create_run_from_pipeline_func(
        pipeline_func=main_pipeline,
        arguments=arguments
    )
