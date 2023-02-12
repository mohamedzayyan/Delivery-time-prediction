import warnings

warnings.filterwarnings(action="ignore")

import hydra
import joblib
import mlflow
import pandas as pd
from helper import BaseLogger
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

logger = BaseLogger()


def load_data(path: DictConfig):
    X_test = pd.read_csv(abspath(path.X_test.path))
    y_test = pd.read_csv(abspath(path.y_test.path))
    return X_test, y_test


def load_model(model_path: str):
    return joblib.load(model_path)


def predict(model: XGBRegressor, X_test: pd.DataFrame):
    return model.predict(X_test)


def log_params(model: XGBRegressor, features: list):
    logger.log_params({"model_class": type(model).__name__})
    model_params = model.get_params()

    for arg, value in model_params.items():
        logger.log_params({arg: value})

    logger.log_params({"features": features})


def log_metrics(**metrics: dict):
    logger.log_metrics(metrics)


@hydra.main(version_base=None, config_path="../../config", config_name="main")
def evaluate(config: DictConfig):
    mlflow.set_tracking_uri(config.mlflow_tracking_ui)

    with mlflow.start_run():

        # Load data and model
        X_test, y_test = load_data(config.processed)

        model = load_model(abspath(config.model.path))

        # Get predictions
        prediction = predict(model, X_test)

        # Get metrics
        mse = mean_squared_error(y_test, prediction)
        print(f"mse of this model is {mse}.")

        r2 = r2_score(y_test, prediction)
        print(f"r2 Score of this model is {r2}.")

        # Log metrics
        log_params(model, config.process.features)
        log_metrics(mean_squared_error=mse, r2_score=r2)


if __name__ == "__main__":
    evaluate()