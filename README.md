# sklearn_turbostart
Quick start your ML model development with sklearn.

Define your own preprocessing pipeline and fit a model using preddefined estimators wth `GridSearchCV` as defined in `estimators.py`. 

Currently defined regressors include

- ElasticNet
- RandomForest
- HistGradientBoost

Example

```
from sklearn.compose import ColumnTransformer
from sklearn_turbostart.estimators import create_estimators
from sklearn_turbostart.train import load_data, train_estimators
from sklearn.preprocessing import StandardScaler

# Load data
X_train, y_train, X_test, y_test = load_data("../data/data.csv")

# Preprocessing
preprocessing = ColumnTransformer(
    [
        (
            "numeric_standard_scalar",
            StandardScaler(),
            ["Feature 1", "Feature 2", "Feature 9"],
        ),
    ]
)

# Create sklearn Pipeline
estimators = create_estimators(preprocessing)

# Fit models
train_estimators(estimators, X_train, y_train, X_test, y_test)
```

After fitting, best estimator can be accessed using, 
```
estimators[<estimator_name>: Optional["HistFradientBoost", "RandomForest", "ElasticNet"]].best_estimator_.
```

Check out `Quickstarter.ipynb` in `notebooks/`