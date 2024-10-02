"""Fraud detection model testing with evaluation metrics"""

import os
import numpy
import logging
from typing import Literal

import shap
import joblib
import pandas
import xgboost
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import helpers.paths
import helpers.logger


def get_test_data(cache_path: str) -> pandas.DataFrame:
    """Load test data from a specified cache path."""
    path = helpers.paths.get_test_data_path(cache_path=cache_path)
    logging.info(f"Loading testing data from {path}")
    return pandas.read_parquet(path=path)


def get_train_data(cache_path: str) -> pandas.DataFrame:
    """Load test data from a specified cache path."""
    path = helpers.paths.get_train_data_path(cache_path=cache_path)
    logging.info(f"Loading train data from {path}")
    return pandas.read_parquet(path=path)


def get_metrics(y_true: numpy.ndarray, y_pred: numpy.ndarray):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    logging.info(f"Accuracy: {accuracy:.2f}")
    logging.info(f"Precision: {precision:.2f}")
    logging.info(f"Recall: {recall:.2f}")
    logging.info(f"F1 Score: {f1:.2f}")


def run_linear_regression(cache_path: str) -> None:
    """Run linear regression testing for fraud detection with evaluation metrics."""
    helpers.logger.init()
    logging.info("Running model testing for fraud detection")

    df_test = get_test_data(cache_path=cache_path)
    df_train = get_train_data(cache_path=cache_path)
    model = joblib.load(filename=os.path.join(helpers.paths.get_model_path(cache_path=cache_path), "model.pkl"))
    y_pred_probs = model.predict(df_test.drop("Class", axis=1))
    x_sample = shap.utils.sample(df_train.drop("Class", axis=1), 100)

    get_metrics(y_true=df_test["Class"].to_numpy(), y_pred=(y_pred_probs > 0.5).astype(int))

    logging.info("computing explainer")
    explainer = shap.Explainer(model.predict, x_sample)
    shap_values = explainer(x_sample)

    dependence_dir = os.path.join(
        helpers.paths.get_model_path(cache_path=cache_path), "dependence-plots",
    )
    scatter_dir = os.path.join(
        helpers.paths.get_model_path(cache_path=cache_path), "scatter-plots",
    )
    os.makedirs(scatter_dir, exist_ok=True)
    os.makedirs(dependence_dir, exist_ok=True)
    for column in x_sample.columns:
        logging.info(f"generating dependence plot for column {column}")
        sample_ind = 20
        shap.partial_dependence_plot(
            column,
            model.predict,
            x_sample,
            model_expected_value=True,
            feature_expected_value=True,
            ice=False,
            shap_values=shap_values[sample_ind : sample_ind + 1, :],
        )

        plt.savefig(
            os.path.join(
                helpers.paths.get_model_path(cache_path=cache_path),
                os.path.join(dependence_dir, f"{column}.png"),
            ),
            format="png",
            bbox_inches="tight",
        )
        plt.close()

        shap.plots.scatter(shap_values[:, column])
        plt.savefig(
            os.path.join(
                helpers.paths.get_model_path(cache_path=cache_path),
                os.path.join(scatter_dir, f"{column}.png"),
            ),
            format="png",
            bbox_inches="tight",
        )
        plt.close()

    return


def run_xgboost_classifier(cache_path: str) -> None:
    """Run linear regression testing for fraud detection with evaluation metrics."""
    helpers.logger.init()
    logging.info("Running model testing for fraud detection")

    model = xgboost.Booster()
    model.load_model(fname=os.path.join(helpers.paths.get_model_path(cache_path=cache_path), "model.ubj"))

    df_test = get_test_data(cache_path=cache_path)

    y_pred_probs = model.predict(data=xgboost.DMatrix(df_test.drop("Class", axis=1)))
    get_metrics(y_true=df_test["Class"].to_numpy(), y_pred=(y_pred_probs > 0.5).astype(int))

    return


def run(cache_path: str, model: Literal["linear_regression", "xgboost_classifier"]) -> None:
    if model == "linear_regression":
        return run_linear_regression(cache_path=cache_path)
    if model == "xgboost_classifier":
        return run_xgboost_classifier(cache_path=cache_path)
    raise ValueError("model has an unknown value")
