import logging
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, median_absolute_error
from sklearn.model_selection import GridSearchCV, train_test_split

plt.style.use("seaborn-talk")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def load_data(
    data_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(data_path, sep=";")
    assert len(df) > 0
    df["Y"] = df["Y"].apply(lambda x: np.log(x))
    log.info("Transforming Y to log(Y)")
    log.info("Splitting data as 80/20")
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=2020)
    X_train = df_train.drop("Y", axis=1)
    y_train = df_train["Y"]
    X_test = df_test.drop("Y", axis=1)
    y_test = df_test["Y"]
    return X_train, y_train, X_test, y_test


def train_estimators(
    estimators: Dict[str, GridSearchCV],
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=True)
    for ax, (estimator_name, estimator) in zip(axes, estimators.items()):
        reg = estimator.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        score = reg.score(X_test, y_test)
        print(f"Estimator: {estimator_name}")
        print(f"Training score: {reg.score(X_train, y_train):.3f}")
        print(f"Cross-val score: {reg.best_score_:.3f}")
        print(f"Test score: {score:.3f}")
        print(f"----------")
        scores = f"$R^2$={score:.3f}, MAE={median_absolute_error(y_test, y_pred):.3f}, MSE={mean_squared_error(y_test, y_pred):.3f}"
        plot_prediction(
            y_test=y_test, y_pred=y_pred, ax=ax, scores=scores, name=estimator_name
        )


def plot_prediction(
    y_pred: np.ndarray,
    y_test: np.ndarray,
    ax=None,
    scores: str = None,
    name: str = None,
) -> None:
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.scatter(y_test, y_pred, alpha=0.3)
    ax.plot([8.5, 10.5], [8.5, 10.5], "--r")
    ax.set_ylabel("Target predicted")
    ax.set_xlabel("True Target")
    extra = plt.Rectangle(
        (0, 0), 0, 0, fc="w", fill=False, edgecolor="none", linewidth=0
    )
    if name:
        ax.set_title(name)
    if scores:
        ax.legend([extra], [scores], loc="upper left")
