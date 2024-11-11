from kfp import dsl
from kfp.v2.dsl import pipeline, Input, Output, Dataset, Artifact
from pull_data_component import pull_data
from preprocess_component import preprocess
from train_component import train
from evaluate_component import evaluate
from deploy_model_component import deploy_model
from inference_component import inference

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
    
    # Check if evaluation passed and deploy model if so
    eval_output = eval_task.output  # Accessing output directly
    if eval_output:
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
    import kfp
    kfp.Client().create_run_from_pipeline_func(
        pipeline_func=main_pipeline,
        arguments={  # Example arguments
            'url': 'https://someurl.com/data.csv',
            'mlflow_experiment_name': 'wine_quality_experiment',
            'mlflow_tracking_uri': 'http://mlflow-tracking-server',
            'mlflow_s3_endpoint_url': 'https://s3.amazonaws.com',
            'model_name': 'wine_quality_model',
            'alpha': 0.1,
            'l1_ratio': 0.1,
            'threshold_metrics': {'accuracy': 0.8, 'precision': 0.7}
        }
    )
