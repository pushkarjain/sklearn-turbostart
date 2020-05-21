
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_estimators(preprocessing):
    # ElasticNet (alpha = 1.0 -> Lasso)
    param_grid = {
        "regressor__regressor__alpha": (0.001, 0.01, 0.1, 1.0),
        "regressor__regressor__l1_ratio": (0.05, 0.2, 0.5, 0.7, 0.9, 1.0),
    }
    en_pipe = TransformedTargetRegressor(
        regressor=Pipeline([("preprocessor", preprocessing), ("regressor", ElasticNet())]),
        transformer=StandardScaler(),
    )
    en_search = GridSearchCV(en_pipe, param_grid=param_grid, cv=5,)

    # RandomForest
    param_grid = {
        "regressor__regressor__n_estimators": [50, 100, 200, 300],
        "regressor__regressor__max_depth": [5, 6, 7, 8, 15, 20],
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
        "regressor__regressor__max_depth": [6, 15, 20, None],
        "regressor__regressor__max_iter": [100, 200, 300],
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
