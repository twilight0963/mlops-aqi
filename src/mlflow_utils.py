"""MLflow setup shared by train.py and register.py."""
import mlflow

from src.config import repo_path


def tracking_uri(params):
    """Resolve a relative sqlite path against the repo root, so scripts work from any cwd."""
    uri = params["mlflow"]["tracking_uri"]
    prefix = "sqlite:///"
    if uri.startswith(prefix) and not uri[len(prefix):].startswith("/"):
        uri = prefix + str(repo_path(uri[len(prefix):]))
    return uri


def setup(params):
    """Point MLflow at the repo-local store and return the experiment id."""
    mlflow.set_tracking_uri(tracking_uri(params))
    name = params["mlflow"]["experiment_name"]
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        return mlflow.create_experiment(name, artifact_location=repo_path("mlruns").as_uri())
    return exp.experiment_id
