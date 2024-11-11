# preprocess_component.py
from kfp.v2.dsl import Input, Output, Dataset, Artifact, component
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

@component(base_image="python:3.10", packages_to_install=["numpy", "pandas", "scikit-learn"])
def preprocess(
    data: Input[Dataset], 
    scaler_out: Output[Artifact], 
    train_set: Output[Dataset], 
    test_set: Output[Dataset], 
    target: str = "quality"
):
    """
    Preprocess data.
    """
    # Read the dataset
    df = pd.read_csv(data.path)

    # Scaling features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.drop(columns=[target]))

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, df[target], test_size=0.2)

    # Output the processed data
    train_set.output = X_train
    test_set.output = X_test
    scaler_out.output = scaler
