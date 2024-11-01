# main_pipeline.py
from pull_data_component import pull_data
from preprocess_component import preprocess
from train_component import train
from evaluate_component import evaluate
from deploy_model_component import deploy_model
from inference_component import inference
from pipelines import arguments  
from submit_run_pipeline import submit_run

def main_pipeline():
    # Call the pull_data component
    pull_task = pull_data(url=arguments["url"])
    
    # Call the preprocess component
    preprocess_task = preprocess(data=pull_task.outputs["data"])
    
    # Call the train component
    train_task = train(
        train_set=preprocess_task.outputs["train_set"],
        test_set=preprocess_task.outputs["test_set"],
        mlflow_experiment_name=arguments["mlflow_experiment_name"],
        mlflow_tracking_uri=arguments["mlflow_tracking_uri"],
        mlflow_s3_endpoint_url=arguments["mlflow_s3_endpoint_url"],
        model_name=arguments["model_name"],
        alpha=arguments["alpha"],
        l1_ratio=arguments["l1_ratio"]
    )
    
    # Call the evaluate component
    eval_task = evaluate(
        run_id=train_task.outputs["run_id"],
        mlflow_tracking_uri=arguments["mlflow_tracking_uri"],
        threshold_metrics=arguments["threshold_metrics"]
    )
    
    # Check evaluation results
    if eval_task.output:
        # Call the deploy_model component
        deploy_model(model_name=arguments["model_name"], storage_uri=train_task.outputs["storage_uri"])
        
        # Call the inference component
        inference(model_name=arguments["model_name"], scaler_in=preprocess_task.outputs["scaler_out"])

if __name__ == "__main__":
    main_pipeline()