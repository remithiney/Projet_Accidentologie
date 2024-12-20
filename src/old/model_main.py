import mlflow
import mlflow.sklearn
import logging
import json
import pandas as pd
from loghandler import LogHandlerManager
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from models.data_loader import DataLoader
from models.evaluator import Evaluator
from models.piepline_builder_old import PipelineBuilder
from models.trainer_old import Trainer

# Charger le fichier de configuration
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Journalisation
log_handler_manager = LogHandlerManager(config["logs"])
log_handler_manager.start_listener()
logger = logging.getLogger(__name__)
logger.addHandler(log_handler_manager.get_queue_handler())
logger.setLevel(logging.DEBUG)

# Chargement des données
data_loader = DataLoader(logger)
X_train, X_test, y_train, y_test = data_loader.load_data("../data/processed/merged_data_2019_2022.csv", target_column="grav", sample_ratio=0.10)

# Modèles et leurs grilles d'hyperparamètres
models = {
    'RandomForest': (
        RandomForestClassifier(random_state=42, verbose=2),
        {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [5, 10, 20, None],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4, 8],
            'classifier__max_features': ['sqrt', 'log2', None],
            'classifier__bootstrap': [True, False]
        }
    ),
    'XGBoost': (
        XGBClassifier(random_state=42, eval_metric='logloss'),
        {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [3, 5, 7],
            'classifier__learning_rate': [0.01, 0.05, 0.1],
            'classifier__subsample': [0.6, 0.8, 1.0],
            'classifier__colsample_bytree': [0.6, 0.8, 1.0],
            'classifier__gamma': [0, 0.1, 0.5, 1],
            'classifier__reg_alpha': [0, 0.1, 1],
            'classifier__reg_lambda': [1, 2, 5, 10]
        }
    ),
    'LightGBM': (
        LGBMClassifier(random_state=42),
        {
            'classifier__num_leaves': [31, 50, 100],
            'classifier__max_depth': [-1, 10, 20],
            'classifier__learning_rate': [0.01, 0.05, 0.1],
            'classifier__n_estimators': [100, 200, 500],
            'classifier__min_child_samples': [20, 50, 100]
        }
    ),
    'LogisticRegression': (
        LogisticRegression(max_iter=1000, random_state=42),
        {
            'classifier__penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'classifier__C': [0.01, 0.1, 1, 10, 100],
            'classifier__solver': ['liblinear', 'saga']
        }
    ),
    'KNN': (
        KNeighborsClassifier(),
        {
            'classifier__n_neighbors': [3, 5, 10, 15],
            'classifier__weights': ['uniform', 'distance'],
            'classifier__metric': ['euclidean', 'manhattan', 'minkowski']
        }
    ),
}


# Initialisation des classes nécessaires
pipeline_builder = PipelineBuilder(logger)
trainer = Trainer(logger)
evaluator = Evaluator(logger)

# Stockage des résultats
results = []

for model_name, (model, param_grid) in models.items():
    logger.info(f"Training and evaluating model: {model_name}")
    
    # Démarrer un run MLflow
    with mlflow.start_run(run_name=model_name):
        try:
            # Construire le pipeline
            pipeline = pipeline_builder.build_pipeline(model, X_train)

            # Entraîner le modèle avec RandomizedSearchCV
            best_model, best_params = trainer.train_model(pipeline, param_grid, X_train, y_train)

            # Log des hyperparamètres dans MLflow
            mlflow.log_params(best_params)

            # Ajuster le seuil de décision et évaluation
            y_proba = best_model.predict_proba(X_test)[:, 1]
            optimal_threshold = 0.5
            y_pred = (y_proba >= optimal_threshold).astype(int)

            # Calcul des métriques
            recall, roc_auc = evaluator.evaluate_model(best_model, X_test, y_test, model_name)

            # Log des métriques dans MLflow
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("roc_auc", roc_auc)

            # Enregistrer le modèle dans MLflow
            mlflow.sklearn.log_model(best_model, artifact_path="model")

            # Sauvegarder les résultats
            results.append({
                "model": model_name,
                "recall": recall,
                "roc_auc": roc_auc,
                "best_params": best_params
            })
        except Exception as e:
            logger.error(f"An error occurred while processing model {model_name}: {e}")


# Affichage des résultats
results_df = pd.DataFrame(results)
output_file_path = config['model_result_path']
with open(output_file_path, 'w') as file:
    file.write("=== Model Performance Results ===\n")
    file.write(results_df.to_string(index=False))
logger.info(results_df)

log_handler_manager.stop_listener()




print(f"Les résultats ont été sauvegardés dans un fichier texte : {output_file_path}")

