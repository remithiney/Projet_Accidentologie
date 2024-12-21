import json
import numpy as np
import logging
import time
from loghandler import LogHandlerManager
from random import choice
from sklearn.metrics import matthews_corrcoef, recall_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from models.data_loader import DataLoader
from old.piepline_builder_old import PipelineBuilder
from models.evaluator import Evaluator
import mlflow

mlflow.set_experiment("Accidents")

# Charger la configuration depuis config.json
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Journalisation
log_handler_manager = LogHandlerManager(config["logs"])
log_handler_manager.start_listener()
logger = logging.getLogger(__name__)
logger.addHandler(log_handler_manager.get_queue_handler())
logger.setLevel(logging.DEBUG)

# Définir les hyperparamètres
models = ['RandomForest', 'XGBoost', 'LightGBM', 'LogisticRegression', 'KNN']
variance_thresholds = np.linspace(0.00, 0.10, 11)
importance_thresholds = np.linspace(0.00, 0.10, 11)
correlation_thresholds = np.linspace(1.0, 0.5, 6)
decision_thresholds = np.linspace(0.0, 0.9, 10)

# Hyperparamètres RL
alpha = 0.1  # Taux d'apprentissage
gamma = 0.9  # Facteur d'actualisation

def update_q_table(state, reward, next_state):
    """Mise à jour cumulative de la Q-Table."""
    state_key = tuple(state.items())
    next_state_key = tuple(next_state.items())
    if state_key not in q_table:
        q_table[state_key] = 0  # Initialiser à 0 si l'état n'existe pas
    if next_state_key not in q_table:
        q_table[next_state_key] = 0  # Même chose pour l'état suivant

    # Mise à jour de la Q-Table
    q_table[state_key] = q_table[state_key] + alpha * (
        reward + gamma * max(q_table[next_state_key], 0) - q_table[state_key]
    )

# Mapping des modèles
model_mapping = {
    'RandomForest': RandomForestClassifier,
    'XGBoost': XGBClassifier,
    'LightGBM': LGBMClassifier,
    'LogisticRegression': LogisticRegression,
    'KNN': KNeighborsClassifier
}

# Initialisation des classes nécessaires
data_loader = DataLoader(logger=logger)  # Ajouter un logger si nécessaire
pipeline_builder = PipelineBuilder(logger=logger)  # Ajouter un logger si nécessaire
evaluator = Evaluator(logger=logger)  # Ajouter un logger si nécessaire

# Charger les données
X_train, X_test, y_train, y_test = data_loader.load_data(
    config["path_to_processed_csv"] + "/merged_data_2019_2022.csv",
    target_column="grav",
    sample_ratio=0.10
)

# Q-Table pour stocker les récompenses
q_table = {}

# Fonction pour calculer la récompense (basée sur le recall)
def calculate_reward(y_true, y_pred):
    recall = recall_score(y_true, y_pred, average='weighted')
    return recall

# Initialisation d'un état aléatoire
def initialize_state():
    return {
        'model': choice(models),
        'variance_threshold': choice(variance_thresholds),
        'importance_threshold': choice(importance_thresholds),
        'correlation_threshold': choice(correlation_thresholds),
        'decision_threshold': choice(decision_thresholds)
    }

# Modifier l'état en appliquant une action aléatoire
def take_action(state):
    param_to_modify = choice(list(state.keys()))
    if param_to_modify == 'model':
        state['model'] = choice(models)
    elif param_to_modify == 'variance_threshold':
        state['variance_threshold'] = choice(variance_thresholds)
    elif param_to_modify == 'importance_threshold':
        state['importance_threshold'] = choice(importance_thresholds)
    elif param_to_modify == 'correlation_threshold':
        state['correlation_threshold'] = choice(correlation_thresholds)
    elif param_to_modify == 'decision_threshold':
        state['decision_threshold'] = choice(decision_thresholds)
    return state

# Entraînement RL
with mlflow.start_run(run_name="RLOptimizer"):
    for episode in range(config.get('num_episodes', 100)):
        with mlflow.start_run(run_name=f"Episode_{episode+1}", nested=True):
            state = initialize_state()
            for step in range(config.get('steps_per_episode', 20)):
                # Construire le modèle et le pipeline
                model_class = model_mapping[state['model']]
                model = model_class()

                pipeline = pipeline_builder.build_pipeline(
                    model,
                    X_train,
                    feature_selector_params={
                        'variance_threshold': state['variance_threshold'],
                        'importance_threshold': state['importance_threshold'],
                        'correlation_threshold': state['correlation_threshold']
                    }
                )

                # Entraîner et évaluer
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)

                # Calculer la récompense
                reward = calculate_reward(y_test, y_pred)

                # Loguer le modèle en cours avec la récompense via logger et MLflow
                logger.info(f"Current model: {state['model']}, Reward: {reward}")
                logger.info(f"Selected parameters: {state}")

                # Utiliser un suffixe pour différencier les logs dans MLflow
                mlflow.log_param(f"current_model_step_{step}", state['model'])
                mlflow.log_metric(f"current_reward_step_{step}", reward)

                # Obtenir un nouvel état après action
                next_state = take_action(state)

                # Mettre à jour la Q-Table
                update_q_table(state, reward, next_state)

                # Passer à l'état suivant
                state = next_state

    # Identifier la meilleure configuration
    best_state = max(q_table, key=q_table.get)
    best_reward = q_table[best_state]

    # Loguer la meilleure configuration
    mlflow.log_params({f"best_{key}": value for key, value in dict(best_state).items()})
    mlflow.log_metric("best_reward", best_reward)

    # Sauvegarder le modèle correspondant à la meilleure configuration
    best_model_class = model_mapping[dict(best_state)['model']]
    best_model = best_model_class()
    best_pipeline = pipeline_builder.build_pipeline(
        best_model,
        X_train,
        feature_selector_params={
            'variance_threshold': dict(best_state)['variance_threshold'],
            'importance_threshold': dict(best_state)['importance_threshold'],
            'correlation_threshold': dict(best_state)['correlation_threshold']
        }
    )
    best_pipeline.fit(X_train, y_train)
    mlflow.sklearn.log_model(best_pipeline, artifact_path="best_model")

print(f"Best configuration: {best_state}, Reward: {best_reward}")
