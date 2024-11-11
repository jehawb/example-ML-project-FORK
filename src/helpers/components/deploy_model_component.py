
# deploy_model_component.py
from kfp.v2.dsl import component

@component(
    base_image="python:3.9",
    packages_to_install=["kserve==0.11.0"],
)
def deploy_model(model_name: str, storage_uri: str):
    """
    Deploy the model as an inference service with KServe.
    """
    import logging
    from kubernetes import client
    from kserve import KServeClient, V1beta1InferenceService, V1beta1InferenceServiceSpec, V1beta1PredictorSpec, V1beta1SKLearnSpec
    from kubernetes.client import V1ResourceRequirements

    logging.basicConfig(level=logging.INFO)
    model_uri = f"{storage_uri}/{model_name}"
    namespace = 'kserve-inference'
    kserve_version = 'v1beta1'
    api_version = 'serving.kserve.io/' + kserve_version

    isvc = V1beta1InferenceService(
        api_version=api_version,
        kind="InferenceService",
        metadata=client.V1ObjectMeta(name=model_name, namespace=namespace, annotations={'sidecar.istio.io/inject':'false'}),
        spec=V1beta1InferenceServiceSpec(
            predictor=V1beta1PredictorSpec(
                sklearn=V1beta1SKLearnSpec(
                    storage_uri=model_uri,
                    resources=V1ResourceRequirements(requests={"cpu": "100m", "memory": "512Mi"}, limits={"cpu": "300m", "memory": "512Mi"})
                ),
            )
        )
    )
    KServe = KServeClient()
    KServe.create(isvc)
