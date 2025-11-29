# Import libraries
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os
import warnings
import sys

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # Argumen MLProject
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    dataset_path = sys.argv[3] if len(sys.argv) > 3 else "DatasetAirlinePassengerSatisfaction_preprocessing/data_train.csv"
    
    # Load Dataset
    base_path = os.path.dirname(dataset_path)
    if base_path == "": base_path = "."

    data_train = pd.read_csv(os.path.join(base_path, "data_train.csv"))
    data_test = pd.read_csv(os.path.join(base_path, "data_test.csv"))

    # Split data
    X_train = data_train.drop(columns=["satisfaction"])
    y_train = data_train["satisfaction"]
    X_test = data_test.drop(columns=["satisfaction"])
    y_test = data_test["satisfaction"]

    # Contoh data input
    input_example = X_train[0:5]

    with mlflow.start_run():

        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # Train model
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        # Log Model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )
        
        # Log metrics
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)
        
        print(f"Done Training. Accuracy: {accuracy:.4f}")