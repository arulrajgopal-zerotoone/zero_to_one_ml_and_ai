import mlflow
from mlflow.models import infer_signature
from mlflow import set_tracking_uri, set_experiment

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load a regression dataset
X, y = datasets.load_diabetes(return_X_y=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define model hyperparameters
params = {
    "fit_intercept": True,
    "copy_X": True,
    "n_jobs": None,
}

# Train the model
lr = LinearRegression(**params)
lr.fit(X_train, y_train)

# Predict on the test set
y_pred = lr.predict(X_test)

# Calculate regression metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Set tracking URI and experiment
set_tracking_uri(uri="http://127.0.0.1:8080")
set_experiment("MLflow Quickstart")

# Start an MLflow run
with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)

    signature = infer_signature(X_train, lr.predict(X_train))

    model_info = mlflow.sklearn.log_model(
        sk_model=lr,
        artifact_path="diabetes_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="regression-quickstart",
    )

    mlflow.set_tag("model_type", "Linear Regression")

# Load the model and predict
loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
predictions = loaded_model.predict(X_test)

# Display results
feature_names = datasets.load_diabetes().feature_names
result = pd.DataFrame(X_test, columns=feature_names)
result["actual"] = y_test
result["predicted"] = predictions

print(result.head())
