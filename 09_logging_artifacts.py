import mlflow
from mlflow_utils import get_mlflow_experiment

if __name__ == "__main__":

  experiment = get_mlflow_experiment(experiment_name="testing_mlflow1")

  print("Experiment Name {}".format(experiment.name))

  with mlflow.start_run(run_name="logging_artifacts", experiment_id=experiment.experiment_id):

    # Your machine learning code goes here

    # create a text file that says hello world
    with open("hello_world.txt", "w") as f:
      f.write("hello world")
      
    # Log the text file as an artifact
    mlflow.log_artifacts(local_dir="./run_artifacts", artifact_path="run_artifacts")
    mlflow.log_artifact(local_path="hello_world.txt")