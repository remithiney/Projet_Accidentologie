import traceback
import time
from models.metrics_calculator import MetricsCalculator
from models.artifact_logger import ArtifactLogger
import mlflow

class Evaluator:
    def __init__(self, logger, save_curves=True):
        self.logger = logger
        self.save_curves = save_curves
        self.metrics_calculator = MetricsCalculator(logger)
        self.artifact_logger = ArtifactLogger(logger, save_curves)

    def evaluate_model(self, model, X_test, y_test, model_name, threshold=0.5):
        try:
            self.logger.info(f"Evaluating model: {model_name}")
            start_time = time.time()

            # Prédictions probabilistes ou classiques
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
                y_pred = (y_proba >= threshold).astype(int)  # Appliquer le seuil
            else:
                y_pred = model.predict(X_test)
                y_proba = None

            # Calcul des métriques
            self.logger.info("Calculating metrics...")
            metrics = self.metrics_calculator.calculate_metrics(y_test, y_pred, y_proba)

            # Log des métriques dans MLflow
            self.log_metrics_to_mlflow(metrics)

            # Log des artefacts
            self.logger.info("Logging artifacts...")
            self.artifact_logger.log_artifacts(model_name, X_test, y_test, y_pred, y_proba, model)

            # Temps d'évaluation
            elapsed_time = time.time() - start_time
            self.logger.info(f"Model evaluation completed in {elapsed_time:.2f} seconds")
            mlflow.log_metric("evaluation_time", elapsed_time)

            return metrics
        except Exception as e:
            self.logger.error("Error during model evaluation.")
            self.logger.error(traceback.format_exc())
            mlflow.log_param("evaluation_error", str(e))
            raise

    def log_metrics_to_mlflow(self, metrics):
        self.logger.info("Logging metrics to MLflow...")
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
            self.logger.info(f"Logged {metric_name}: {metric_value}")
