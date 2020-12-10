import logging
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose._column_transformer import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, median_absolute_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

plt.style.use("seaborn-talk")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def load_data(
    data_path: str,
    target: str="Y",
    dataset = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads data from datapath. Performs train-test split and transforms response to log

    TODO: Figure out doing log transformation in Pipeline itself

    Args:
        data_path: Path to dataset

    Returns:
        Train and test data
    """
    if dataset == "vehicles":
        df = pd.read_csv(data_path, index_col=0)
        df = df[df[target] > 0]
        df = df[df["odometer"].notnull()]
    else:
        df = pd.read_csv(data_path, sep=";")
    assert len(df) > 0
    df[target] = df[target].apply(lambda x: np.log(1+x))
    log.info("Transforming Y to log(Y)")
    log.info("Splitting data as 80/20")
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=2020)
    X_train = df_train.drop(target, axis=1)
    y_train = df_train[target]
    X_test = df_test.drop(target, axis=1)
    y_test = df_test[target]
    return X_train, y_train, X_test, y_test


def train_estimators(
    estimators: Dict[str, GridSearchCV],
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> None:
    """Fit estimators and yield metrics in prediction plots

    Args:
        estimators: Key-value pairs of estimator names and estimators to be fit
        X_train: Train features
        y_train: Train response
        X_test: Test features
        y_test: Test response
    """
    fig, axes = plt.subplots(1, len(estimators.keys()), figsize=(20, 5), sharey=True)
    for ax, (estimator_name, estimator) in zip(axes, estimators.items()):
        reg = estimator.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        score = reg.score(X_test, y_test)
        print(f"Estimator: {estimator_name}")
        print(f"Training score: {reg.score(X_train, y_train):.3f}")
        print(f"Cross-val score: {reg.best_score_:.3f}")
        print(f"Test score: {score:.3f}")
        print(f"----------")
        scores = (
            f"$R^2$={score:.3f}, "
            f"MAE={median_absolute_error(y_test, y_pred):.3f}, "
            f"MSE={mean_squared_error(y_test, y_pred):.3f}"
        )
        plot_prediction(
            y_test=y_test, y_pred=y_pred, ax=ax, scores=scores, name=estimator_name
        )


def plot_prediction(
    y_pred: np.ndarray,
    y_test: np.ndarray,
    ax: plt.Axes = None,
    scores: str = None,
    name: str = None,
) -> None:
    """Plot prediction vs truth

    Args:
        y_pred: Predicted values
        y_test: Ground truth
        ax: Axes
        scores: String with scores to show as plot legend
        name: Title of plot
    """
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


def get_column_names_from_ColumnTransformer(
    column_transformer: ColumnTransformer,
) -> List[str]:
    """Get column names from column transformer.
    
    This is required because feature importance is poorly compatible with `sklearn` `Pipeline`. 

    Args:
        column_transformer: Feature transformer

    Returns:
        Column names after transformation
    """
    col_name: List[str] = []
    # the last transformer is ColumnTransformer's 'remainder'
    for transformer_in_columns in column_transformer.transformers_:
        raw_col_name = transformer_in_columns[2]
        if isinstance(transformer_in_columns[1], Pipeline):
            transformer = transformer_in_columns[1].steps[-1][1]
        else:
            transformer = transformer_in_columns[1]
        try:
            names = transformer.get_feature_names()
        except AttributeError:  # if no 'get_feature_names' function, use raw column name
            names = raw_col_name
        if isinstance(names, np.ndarray):
            col_name += names.tolist()
        elif isinstance(names, list):
            col_name += names
        elif isinstance(names, str):
            col_name.append(names)
    return col_name


def plot_permutation_importance(
    estimators: Dict[str, GridSearchCV], X_test: pd.DataFrame, y_test: pd.DataFrame
) -> None:
    """Plot permutation importance for estimators

    Args:
        estimators: Key-value pairs of estimator names and **fitted** estimators
        X_test: Test features
        y_test : Test response
    """
    fig, axes = plt.subplots(1, len(estimators), figsize=(15, 5))
    for ax, (estimator_name, estimator) in zip(axes, estimators.items()):
        est = estimator.best_estimator_
        result = permutation_importance(
            est, X_test, y_test, n_repeats=10, random_state=2020
        )
        perm_sorted_idx = result.importances_mean.argsort()
        columns = get_column_names_from_ColumnTransformer(
            est.regressor_.named_steps["preprocessor"]
        )
        log.info(columns)
        labels = [columns[i] for i in perm_sorted_idx]
        ax.boxplot(
            result.importances[perm_sorted_idx].T, vert=False, labels=labels,
        )
        ax.set_title(estimator_name)
    fig.tight_layout()
