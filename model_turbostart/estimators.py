from typing import Dict

from sklearn.compose import TransformedTargetRegressor
from sklearn.compose._column_transformer import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_estimators(preprocessing: ColumnTransformer) -> Dict[str, GridSearchCV]:
    """Create estimators for model fitting.

    ElasticNet, HistGradientBoost and RandomForest regressors are added. 
    More can be added.

    Args:
        preprocessing: Preprocessing Pipeline

    Returns:
        Key-value pairs of estimator name and instantiated estimator
    """
    # ElasticNet (alpha = 1.0 -> Lasso)
    param_grid = {
        "regressor__regressor__alpha": (0.001, 0.01, 0.1, 1.0),
        "regressor__regressor__l1_ratio": (0.05, 0.2, 0.5, 0.7, 0.9, 1.0),
    }
    en_pipe = TransformedTargetRegressor(
        regressor=Pipeline(
            [("preprocessor", preprocessing), ("regressor", ElasticNet())]
        ),
        transformer=StandardScaler(),
    )
    en_search = GridSearchCV(en_pipe, param_grid=param_grid, cv=5,)

    # RandomForest
    param_grid = {
        "regressor__regressor__n_estimators": [50, 100, 200],
        "regressor__regressor__max_depth": [5, 6, 7, 15],
    }
    rf_pipe = TransformedTargetRegressor(
        regressor=Pipeline(
            [("preprocessor", preprocessing), ("regressor", RandomForestRegressor())]
        ),
        transformer=StandardScaler(),
    )
    rf_search = GridSearchCV(rf_pipe, param_grid=param_grid, cv=5,)

    # HistGradientBoost
    param_grid = {
        "regressor__regressor__l2_regularization": [0.0, 0.1, 1.0],
        "regressor__regressor__max_depth": [6, 15],
        "regressor__regressor__max_iter": [100, 200],
    }
    hgb_pipe = TransformedTargetRegressor(
        regressor=Pipeline(
            [
                ("preprocessor", preprocessing),
                ("regressor", HistGradientBoostingRegressor()),
            ]
        ),
        transformer=StandardScaler(),
    )
    hgb_search = GridSearchCV(hgb_pipe, param_grid=param_grid, cv=5,)

    return {
        "ElasticNet": en_search,
        "RandomForest": rf_search,
        "HistGradientBoost": hgb_search,
    }
