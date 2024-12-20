from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTENC
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from models.feature_selector import FeatureSelector
import traceback
import time
import mlflow


class PipelineBuilder:
    def __init__(self, logger, smotenc_params=None, feature_selector_params=None):
        self.logger = logger
        self.smotenc_params = smotenc_params or {'random_state': 42, 'sampling_strategy': 'minority'}
        self.feature_selector_params = feature_selector_params or {
            'variance_threshold': 0.01,
            'importance_threshold': 0.01,
            'correlation_threshold': 0.7
        }

    def get_column_types(self, X):
        try:
            start_time = time.time()
            with mlflow.start_run(nested=True, run_name="Identify_Columns"):
                self.logger.info("Identifying column types...")
                numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
                categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
                mlflow.log_metric("num_columns", len(numerical_columns))
                mlflow.log_metric("cat_columns", len(categorical_columns))
                mlflow.log_metric("execution_time", time.time() - start_time)
                return numerical_columns, categorical_columns
        except Exception as e:
            self.logger.error("Error while identifying column types.")
            self.logger.error(traceback.format_exc())
            raise

    def build_preprocessor(self, numerical_cols, categorical_cols):
        try:
            start_time = time.time()
            with mlflow.start_run(nested=True, run_name="Build_Preprocessor"):
                self.logger.info("Building preprocessor...")
                numerical_pipeline = Pipeline([
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', MinMaxScaler())
                ])
                categorical_pipeline = Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))
                ])
                preprocessor = ColumnTransformer([
                    ('num', numerical_pipeline, numerical_cols),
                    ('cat', categorical_pipeline, categorical_cols)
                ])
                mlflow.log_metric("execution_time", time.time() - start_time)
                self.logger.info("Preprocessor built successfully.")
                return preprocessor
        except Exception as e:
            self.logger.error("Error while building preprocessor.")
            self.logger.error(traceback.format_exc())
            raise

    def build_pipeline(self, model, X, feature_selector_params=None):
        try:
            if feature_selector_params is None:
                feature_selector_params = self.feature_selector_params

            with mlflow.start_run(nested=True, run_name="Build_Full_Pipeline"):
                self.logger.info(f"Building pipeline for {model}...")
                numerical_cols, categorical_cols = self.get_column_types(X)

                # Étape 1: Prétraitement
                preprocessor = self.build_preprocessor(numerical_cols, categorical_cols)

                # Étape 2: Gestion du déséquilibre des classes avec SMOTENC
                categorical_indices = [X.columns.get_loc(col) for col in categorical_cols]
                smotenc = SMOTENC(categorical_features=categorical_indices, **self.smotenc_params)
                mlflow.log_params(self.smotenc_params)

                # Étape 3: Sélection des features
                start_time = time.time()
                feature_selector = FeatureSelector(**self.feature_selector_params)
                mlflow.log_params(self.feature_selector_params)
                mlflow.log_metric("feature_selection_time", time.time() - start_time)

                # Pipeline final
                pipeline = ImbPipeline([
                    ('preprocessor', preprocessor),
                    ('smotenc', smotenc),
                    ('feature_selection', feature_selector),
                    ('classifier', model)
                ])
                mlflow.log_param("model_name", model.__class__.__name__)
                self.logger.info("Pipeline built successfully.")
                return pipeline

        except Exception as e:
            self.logger.error("Error while building pipeline.")
            self.logger.error(traceback.format_exc())
            raise
