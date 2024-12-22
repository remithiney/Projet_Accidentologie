from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold, RFE
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 importance_threshold=0.01, 
                 variance_threshold=0.01, 
                 correlation_threshold=0.8, 
                 k_best=80, 
                 random_state=42):
        self.importance_threshold = importance_threshold
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.k_best = k_best
        self.random_state = random_state
        self.selected_features = None
        self.feature_importances_ = None

    def _remove_low_variance(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        selector = VarianceThreshold(threshold=self.variance_threshold)
        selector.fit(X)
        return X.loc[:, selector.get_support()]

    def _select_by_importance(self, X, y):
        model = RandomForestClassifier(random_state=self.random_state)
        model.fit(X, y)

        # Trier les caractéristiques par importance
        self.feature_importances_ = model.feature_importances_
        sorted_indices = np.argsort(self.feature_importances_)[::-1]

        # Sélectionner les k meilleures caractéristiques si k_best est défini
        if self.k_best is not None:
            top_k_indices = sorted_indices[:self.k_best]
            important_features = X.columns[top_k_indices]
        else:
            important_features = X.columns[np.argsort(self.feature_importances_)[::-1]]

        return X[important_features]

    def _remove_high_correlation(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > self.correlation_threshold)]
        return X.drop(columns=to_drop, errors="ignore")

    def fit(self, X, y):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        self.feature_names_in_ = X.columns

        X = self._remove_low_variance(X)
        X = self._select_by_importance(X, y)
        X = self._remove_high_correlation(X)
        #X = self._apply_rfe(X, y) 

        self.selected_features = X.columns
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names_in_)

        if self.selected_features is not None:
            return X[self.selected_features]
        else:
            raise ValueError("No features were selected. Ensure `fit` was called before `transform`.")
