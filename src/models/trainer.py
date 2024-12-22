from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, matthews_corrcoef
import traceback
import time
import joblib
import mlflow
import mlflow.sklearn
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class Trainer:
    def __init__(self, logger, scoring='recall', save_model_path=None):
        self.logger = logger
        self.scoring = scoring
        self.save_model_path = save_model_path

    def _get_search(self, pipeline, param_grid, n_iter, cv):
        return RandomizedSearchCV(
            pipeline,
            param_distributions=param_grid,
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
            scoring=self.scoring,
            n_iter=n_iter,
            n_jobs=-1,
            random_state=42,
            verbose=3
        )

    def _save_model(self, model, model_name):
        if self.save_model_path:
            model_path = os.path.join(self.save_model_path, f"{model_name}.joblib")
            self.logger.info(f"Saving best model to {model_path}")
            joblib.dump(model, model_path)
            mlflow.log_artifact(model_path)

    def train_model(self, pipeline, param_grid, X_train, y_train, model_name="Model", n_iter=5, cv=3):
        try:
            # Démarrer un sous-run MLflow
            with mlflow.start_run(nested=True, run_name=f"Training_{model_name}"):
                self.logger.info(f"Starting model training for {model_name}...")
                start_time = time.time()

                # Configuration et recherche des hyperparamètres
                search = self._get_search(pipeline, param_grid, n_iter, cv)
                search.fit(X_train, y_train)
                elapsed_time = time.time() - start_time

                # Extraction des résultats
                best_model = search.best_estimator_
                best_params = search.best_params_
                self.logger.info(f"Training completed in {elapsed_time:.2f} seconds")
                self.logger.info(f"Best parameters: {best_params}")

                # Log des paramètres et métriques dans MLflow
                mlflow.log_param("scoring", self.scoring)
                mlflow.log_params(best_params)
                mlflow.log_metric("training_time", elapsed_time)

                # Log du modèle
                mlflow.sklearn.log_model(best_model, artifact_path=f"models/{model_name}_model")

                # Sauvegarde locale
                self._save_model(best_model, model_name)

                return best_model, best_params

        except Exception as e:
            self.logger.error("Error during model training.")
            self.logger.error(traceback.format_exc())

            # Log de l'erreur dans MLflow
            error_path = "training_error.log"
            with open(error_path, "w") as f:
                f.write(traceback.format_exc())
            mlflow.log_artifact(error_path)
            raise