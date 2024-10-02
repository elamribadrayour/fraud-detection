"""fraud detection model training"""

import os
import logging
from typing import Literal

import joblib
import pandas
import xgboost
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

import helpers.paths
import helpers.logger


def get_train_data(cache_path: str) -> pandas.DataFrame:
    path = helpers.paths.get_train_data_path(cache_path=cache_path)
    logging.info(f"loading training data from {path}")
    return pandas.read_parquet(path=path)


def run_xgboost_classifier(cache_path: str) -> None:
    helpers.logger.init()
    logging.info("running xgboost classifier model training for fraud detection")

    df_train = get_train_data(cache_path=cache_path)

    model = xgboost.XGBClassifier(
        eval_metric="logloss",
        objective="binary:logistic",
    )

    model.fit(
        df_train.drop(columns=["Class"]),
        df_train["Class"],
        verbose=True,
        eval_set=[(df_train.drop(columns=["Class"]), df_train["Class"])],
    )

    dir_path = helpers.paths.get_model_path(cache_path=cache_path)
    model_path = os.path.join(dir_path, "model.ubj")
    eval_path = os.path.join(dir_path, "logloss.jpeg")

    evals_result = model.evals_result()
    epochs = len(evals_result["validation_0"]["logloss"])
    x_axis = range(0, epochs)

    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, evals_result["validation_0"]["logloss"], label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Log Loss")
    plt.title("XGBoost Log Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(eval_path, format="jpeg")
    logging.info(f"evaluation plot saved as {eval_path}")

    model.save_model(model_path)


def run_linear_regression(cache_path: str) -> None:
    helpers.logger.init()
    logging.info("running linear regression model training for fraud detection")

    df_train = get_train_data(cache_path=cache_path)

    model = LinearRegression().fit(df_train.drop(columns=["Class"]), df_train["Class"])

    dir_path = helpers.paths.get_model_path(cache_path=cache_path)
    model_path = os.path.join(dir_path, "model.pkl")
    joblib.dump(model, model_path)
    logging.info(f"model saved to {model_path}")


def run(cache_path: str, model: Literal["linear_regression", "xgboost_classifier"]) -> None:
    if model == "xgboost_classifier":
        return run_xgboost_classifier(cache_path=cache_path)
    if model == "linear_regression":
        return run_linear_regression(cache_path=cache_path)
    raise ValueError("model has an unknown value")
