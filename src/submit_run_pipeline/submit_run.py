# main_pipeline.py
from pipelines import pipeline
from pipelines import arguments
from pipelines import create_kfp_client
import kfp

def submit_pipeline():
    client = create_kfp_client()  # Create the Kubeflow Pipelines client
    
    run_name = "demo-run-aleksi-github"  # Customize your run name
    experiment_name = "demo-experiment"   # Customize your experiment name

    # Create a run from the pipeline function
    client.create_run_from_pipeline_func(
        pipeline_func=pipeline,
        run_name=run_name,
        experiment_name=experiment_name,
        arguments=arguments,
        mode=kfp.dsl.PipelineExecutionMode.V2_COMPATIBLE,
        enable_caching=False,
        namespace="kubeflow-user-example-com"  # Adjust the namespace as needed
    )

if __name__ == "__main__":
    submit_pipeline()