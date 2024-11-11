# pull_data_component.py
from kfp.v2.dsl import component, Output, Dataset
import pandas as pd

@component(base_image="python:3.10", packages_to_install=["pandas", "numpy"])
def pull_data(url: str, data: Output[Dataset]):
    df = pd.read_csv(url, sep=";")
    df.to_csv(data.path, index=False)