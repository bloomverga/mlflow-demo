import mlflow
from mlflow_utils import create_mlflow_experiment

if __name__ == "__main__":

  experiment_id = create_mlflow_experiment(
    experiment_name="testing_mlflow1",
    artifact_location="testing_mlflow1_artifacts",
    tags={"env": "dev", "version": "1.0.0"}
  )

  with mlflow.start_run(run_name="testing", experiment_id = experiment_id) as run:
    # Your machine learning code goes here
    mlflow.log_param("learning_rate", 0.01)

    # Print info
    print("Run ID: {}".format(run.info.run_id))

