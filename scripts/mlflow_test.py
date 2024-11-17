import mlflow

# Set the MLflow tracking URI
mlflow.set_tracking_uri("http://ec2-3-80-80-62.compute-1.amazonaws.com:5000/")
# http://ec2-3-80-80-62.compute-1.amazonaws.com:5000/

# Start an MLflow run with a name
with mlflow.start_run(run_name="test3") as run:
    # Log parameters and metrics
    mlflow.log_param("param1", 15)
    mlflow.log_metric("metric1", 0.89)

    # Log a test name as a tag
    mlflow.set_tag("test_name", "Sample Test for Metrics and Params")