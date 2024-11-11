# inference_component.py
from kfp.v2.dsl import component, Input, Artifact

@component(
    base_image="python:3.9",
    packages_to_install=["numpy~=1.26.4", "kserve==0.11.0", "scikit-learn~=1.0.2"],
)
def inference(model_name: str, scaler_in: Input[Artifact]):
    """
    Test inference by sending a sample request to the deployed model.
    """
    from kserve import KServeClient
    import requests
    import pickle
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    namespace = 'kserve-inference'

    # Sample input data
    input_sample = [[5.6, 0.54, 0.04, 1.7, 0.049, 5, 13, 0.9942, 3.72, 0.58, 11.4],
                    [11.3, 0.34, 0.45, 2, 0.082, 6, 15, 0.9988, 2.94, 0.66, 9.2]]

    # Load scaler
    logger.info(f"Loading standard scaler from: {scaler_in.path}")
    with open(scaler_in.path, 'rb') as fp:
        scaler = pickle.load(fp)

    # Standardize input sample
    input_sample = scaler.transform(input_sample)

    # Inference service URL
    is_url = f"http://istio-ingressgateway.istio-system.svc.cluster.local:80/v1/models/{model_name}:predict"
    header = {"Host": f"{model_name}.{namespace}.example.com"}

    # Prepare inference input
    inference_input = {'instances': input_sample.tolist()}
    response = requests.post(is_url, json=inference_input, headers=header)

    if response.status_code != 200:
        raise RuntimeError(f"HTTP error '{response.status_code}': {response.json()}")
    
    logger.info(f"Prediction response: {response.json()}")
