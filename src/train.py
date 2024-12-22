import logging
import json
import pandas as pd
import mlflow
import mlflow.sklearn
from itertools import product
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTENC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from models.data_loader import DataLoader
from models.evaluator import Evaluator
from models.pipeline_builder import PipelineBuilder
from models.trainer import Trainer
from loghandler import LogHandlerManager

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Mapping des modèles
MODEL_MAPPING = {
    "RandomForest": RandomForestClassifier,
    "XGBoost": XGBClassifier,
    "LightGBM": LGBMClassifier,
    "LogisticRegression": LogisticRegression
}

# Charger le fichier de configuration
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

model_configs = config["model_parameters"]["models"]
feature_selection_params = config["model_parameters"]["feature_selection"]
decision_thresholds = config["model_parameters"]["decision_thresholds"]
scoring_metrics = config["model_parameters"]["scoring_metrics"]

# Journalisation
log_handler_manager = LogHandlerManager(config["logs"])
log_handler_manager.start_listener()
logger = logging.getLogger(__name__)
logger.addHandler(log_handler_manager.get_queue_handler())
logger.setLevel(logging.DEBUG)

mlflow.set_experiment("Accidents")

# Chargement des données
data_loader = DataLoader(logger)
X_train, X_test, y_train, y_test = data_loader.load_data(
    config["path_to_processed_csv"] + "/merged_data_2019_2022.csv",
    target_column="grav")

# Initialisation des classes nécessaires
trainer = Trainer(logger, scoring="f1")
evaluator = Evaluator(logger)
results = []

# Test de toutes les combinaisons de paramètres globaux
with mlflow.start_run(run_name="Accidents_Workflow"):
    for variance, importance, correlation in product(
        feature_selection_params["variance_threshold"],
        feature_selection_params["importance_threshold"],
        feature_selection_params["correlation_threshold"]
    ):
        # Paramètres de sélection des features
        feature_params = {
            "variance_threshold": variance,
            "importance_threshold": importance,
            "correlation_threshold": correlation
        }
        pipeline_builder = PipelineBuilder(logger, feature_selector_params=feature_params)

        for model_name, model_config in model_configs.items():
            try:
                model_class = MODEL_MAPPING[model_name]
                param_grid = model_config["params"]
                model_instance = model_class()

                with mlflow.start_run(run_name=f"{model_name}_params_{feature_params}", nested=True):
                    # Étape 1: Construire le pipeline
                    pipeline = pipeline_builder.build_pipeline(model_instance, X_train)
                    mlflow.log_params(feature_params)

                    for decision_threshold, scoring in product(decision_thresholds, scoring_metrics):
                        with mlflow.start_run(run_name=f"{model_name}_threshold_{decision_threshold}_scoring_{scoring}", nested=True):
                            # Étape 2: Entraînement avec RandomizedSearchCV
                            best_model, best_params = trainer.train_model(
                                pipeline, param_grid, X_train, y_train, model_name=model_name
                            )
                            mlflow.log_params(best_params)

                            # Étape 3: Évaluation
                            metrics = evaluator.evaluate_model(
                                best_model, X_test, y_test, model_name=model_name, threshold=decision_threshold
                            )
                            mlflow.log_metrics(metrics)

                            # Log du modèle
                            mlflow.sklearn.log_model(best_model, f"models/{model_name}_threshold_{decision_threshold}_scoring_{scoring}")

                            # Stockage des résultats
                            results.append({
                                "model": model_name,
                                "decision_threshold": decision_threshold,
                                "scoring": scoring,
                                "metrics": metrics,
                                "best_params": best_params,
                                "feature_params": feature_params
                            })

                            logger.info(f"Model {model_name} evaluated with params {feature_params}, threshold {decision_threshold}, scoring {scoring}.")

            except Exception as e:
                logger.error(f"Error processing model {model_name}: {e}")
                mlflow.log_param(f"error_{model_name}", str(e))

# Affichage des résultats
results_df = pd.DataFrame(results)
output_file_path = config['model_result_path']
with open(output_file_path, 'w') as file:
    file.write("=== Model Performance Results ===\n")
    file.write(results_df.to_string(index=False))

logger.info(results_df)
log_handler_manager.stop_listener()

print(f"Les résultats ont été sauvegardés dans un fichier texte : {output_file_path}")
