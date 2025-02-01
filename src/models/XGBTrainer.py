import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

# ------------------------- Classe XGBoost  ------------------------- #
class XGBTrainer(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, num_boost_round=100, early_stopping_rounds=15,
                 max_depth=6, learning_rate=0.1, subsample=1.0, colsample_bytree=1.0, 
                 random_state=42):
        self.n_estimators = n_estimators
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.model = None
        self.evals_result_ = None

    def fit(self, X, y, eval_set=None):
        dtrain = xgb.DMatrix(X, label=y)
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": self.max_depth,
            "eta": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "random_state": self.random_state
        }

        evals = [(dtrain, "train")]
        if eval_set is not None: # ne sera jamais appelé
            X_val, y_val = eval_set
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, "validation"))

        self.model = xgb.train(
            params, dtrain,
            num_boost_round=self.num_boost_round,
            evals=evals,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=False
        )

        return self

    def predict(self, X):
        dtest = xgb.DMatrix(X)
        return (self.model.predict(dtest) >= 0.5).astype(int)

    def predict_proba(self, X):
        dtest = xgb.DMatrix(X)
        probs = self.model.predict(dtest)
        return np.column_stack((1 - probs, probs))  # Format (classe 0, classe 1)