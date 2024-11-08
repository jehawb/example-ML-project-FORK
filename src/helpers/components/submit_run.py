# submit_run.py
import kfp
from main_pipeline import main_pipeline
from pipelines import arguments

def submit_pipeline():
    client = kfp.Client()  # Connect to Kubeflow Pipelines client

    # Define your experiment and run name
    experiment_name = "demo-experiment"
    run_name = "demo-run"

    # Submit the pipeline run
    client.create_run_from_pipeline_func(
        pipeline_func=main_pipeline,
        arguments=arguments,
        run_name=run_name,
        experiment_name=experiment_name,
        mode=kfp.dsl.PipelineExecutionMode.V2_COMPATIBLE,
        enable_caching=False,
        namespace="kubeflow-user-example-com"  # Make sure this is the correct namespace
    )

if __name__ == "__main__":
    submit_pipeline()
