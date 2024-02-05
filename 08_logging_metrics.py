import mlflow
from mlflow_utils import get_mlflow_experiment

if __name__ == "__main__":

  experiment = get_mlflow_experiment(experiment_name="testing_mlflow1")
  
  with mlflow.start_run(run_name="logging_metrics", experiment_id=experiment.experiment_id):
    mlflow.log_metric("mse", 0.01)

    metrics = {
      "mse": 0.01,
      "mae": 0.01,
      "rmse": 0.01,
      "r2": 0.01
    }

    mlflow.log_metrics(metrics)

    # print run info