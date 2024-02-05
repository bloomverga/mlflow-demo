import mlflow
from mlflow_utils import get_mlflow_experiment

if __name__ == "__main__":
  expriment = get_mlflow_experiment(experiment_name="testing_mlflow1")

  with mlflow.start_run(run_name="testing2", experiment_id=expriment.experiment_id) as run:
    mlflow.log_param("learning_rate", 0.01)

    parameters = {
      "learning_rate": 0.01,
      "epochs": 10,
      "batch_size": 100,
      "loss_function": "mse",
      "optimizer": "adam"
    }
    mlflow.log_params(parameters)

    # print info