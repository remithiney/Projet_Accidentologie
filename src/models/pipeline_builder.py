from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from models.feature_selector import FeatureSelector
import traceback
import time
import mlflow
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class PipelineBuilder:
    def __init__(self, logger, undersampling_params=None, feature_selector_params=None):
        self.logger = logger
        self.undersampling_params = undersampling_params or {'random_state': 42, 'sampling_strategy': 'majority'}
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

                # Debug logs
                self.logger.debug(f"Numerical columns detected: {numerical_columns}")
                self.logger.debug(f"Categorical columns detected: {categorical_columns}")

                X[categorical_columns] = X[categorical_columns].astype(str)
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

                # Debug logs
                self.logger.debug(f"Numerical columns: {numerical_cols}")
                self.logger.debug(f"Categorical columns: {categorical_cols}")

                numerical_pipeline = Pipeline([
                    ('imputer', SimpleImputer(missing_values=-1, strategy='mean')),
                    ('scaler', MinMaxScaler())
                ])

                categorical_pipeline = Pipeline([
                    ('imputer', SimpleImputer(missing_values="-1", strategy='most_frequent')),
                    ('encoder', OneHotEncoder(handle_unknown='ignore'))
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

                # Debug logs
                self.logger.debug(f"Feature selector parameters: {feature_selector_params}")

                numerical_cols, categorical_cols = self.get_column_types(X)

                # Étape 1: Prétraitement
                preprocessor = self.build_preprocessor(numerical_cols, categorical_cols)

                # Étape 2: Gestion du déséquilibre des classes avec RandomUnderSampler
                undersampler = RandomUnderSampler(**self.undersampling_params)
                mlflow.log_params(self.undersampling_params)

                # Étape 3: Sélection des features
                start_time = time.time()
                feature_selector = FeatureSelector(**self.feature_selector_params)
                mlflow.log_params(self.feature_selector_params)
                mlflow.log_metric("feature_selection_time", time.time() - start_time)

                # Pipeline final
                pipeline = ImbPipeline([
                    ('preprocessor', preprocessor),
                    ('undersampler', undersampler),
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
