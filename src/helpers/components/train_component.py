# train_component.py
from kfp.v2.dsl import component, Input, Output, Dataset, Model
from typing import NamedTuple

@component(
    base_image="python:3.10",
    packages_to_install=["numpy~=1.26.4", "pandas~=1.4.2", "scikit-learn~=1.0.2", "mlflow~=2.4.1", "boto3~=1.21.0"],
    output_component_file='components/train_component.yaml',
)
def train(train_set: Input[Dataset], test_set: Input[Dataset], saved_model: Output[Model], mlflow_experiment_name: str, mlflow_tracking_uri: str, mlflow_s3_endpoint_url: str, model_name: str, alpha: float, l1_ratio: float, target: str = "quality") -> NamedTuple("Output", [('storage_uri', str), ('run_id', str),]):
    """
    Train component.
    """
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import ElasticNet
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import mlflow
    import mlflow.sklearn
    import os
    import logging
    import pickle
    from collections import namedtuple

    logging.basicConfig(level=logging.INFO)
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = mlflow_s3_endpoint_url

    train = pd.read_csv(train_set.path)
    test = pd.read_csv(test_set.path)

    train_x = train.drop([target], axis=1)
    test_x = test.drop([target], axis=1)
    train_y = train[[target]]
    test_y = test[[target]]

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name)

    with mlflow.start_run() as run:
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        model.fit(train_x, train_y)
        predicted_qualities = model.predict(test_x)

        rmse = np.sqrt(mean_squared_error(test_y, predicted_qualities))
        mae = mean_absolute_error(test_y, predicted_qualities)
        r2 = r2_score(test_y, predicted_qualities)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(model, model_name, registered_model_name="ElasticnetWineModel", serialization_format="pickle")

        with open(saved_model.path, 'wb') as fp:
            pickle.dump(model, fp, pickle.HIGHEST_PROTOCOL)

        output = namedtuple('Output', ['storage_uri', 'run_id'])
        return output(mlflow.get_artifact_uri(), run.info.run_id)
