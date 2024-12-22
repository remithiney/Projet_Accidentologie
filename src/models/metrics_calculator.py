from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    f1_score,
    cohen_kappa_score,
    matthews_corrcoef,
    log_loss,
    brier_score_loss,
    recall_score,
    balanced_accuracy_score
)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class MetricsCalculator:
    def __init__(self, logger):
        self.logger = logger

    def calculate_metrics(self, y_test, y_pred, y_proba=None):
        metrics = {}

        # Métriques de classification
        metrics['f1_score'] = f1_score(y_test, y_pred, average='weighted')
        metrics['cohen_kappa'] = cohen_kappa_score(y_test, y_pred)
        metrics['mcc'] = matthews_corrcoef(y_test, y_pred)
        metrics['recall'] = recall_score(y_test, y_pred, average='weighted')
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_test, y_pred) 

        self.logger.info(f"F1-score: {metrics['f1_score']:.4f}")
        self.logger.info(f"Cohen Kappa: {metrics['cohen_kappa']:.4f}")
        self.logger.info(f"MCC: {metrics['mcc']:.4f}")
        self.logger.info(f"Recall: {metrics['recall']:.4f}")
        self.logger.info(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")

        if y_proba is not None:
            # Métriques probabilistes
            metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
            metrics['log_loss'] = log_loss(y_test, y_proba)
            metrics['brier_score'] = brier_score_loss(y_test, y_proba)

            self.logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
            self.logger.info(f"Log Loss: {metrics['log_loss']:.4f}")
            self.logger.info(f"Brier Score: {metrics['brier_score']:.4f}")

        return metrics
