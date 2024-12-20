from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, TransformerMixin

class DebugStep(BaseEstimator, TransformerMixin):
    def __init__(self, message):
        self.message = message

    def fit(self, X, y=None):
        print(f"[DEBUG] Fit {self.message}, X shape: {X.shape}")
        return self

    def transform(self, X):
        print(f"[DEBUG] Transform {self.message}, X shape: {X.shape}")
        return X