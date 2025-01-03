import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    classification_report
)
import mlflow
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class ArtifactLogger:
    def __init__(self, logger, save_curves=True):
        self.logger = logger
        self.save_curves = save_curves
        self.report_dir = "reports"
        os.makedirs(self.report_dir, exist_ok=True)

    def log_artifacts(self, model_name, X_test, y_test, y_pred, y_proba, model):
        # Rapport de classification
        self.log_classification_report(model_name, y_test, y_pred)
        
        # Matrice de confusion
        self.log_confusion_matrix(model_name, y_test, y_pred)

        # Courbes ROC et Precision-Recall
        if y_proba is not None:
            self.log_curves(model_name, y_test, y_proba)

        # Résidus
        self.log_residuals(model_name, y_test, y_pred, y_proba)

        # Importance des caractéristiques
        self.log_feature_importance(model_name, model, X_test)

    def log_classification_report(self, model_name, y_test, y_pred):
        report = classification_report(y_test, y_pred, output_dict=True)
        report_path = os.path.join(self.report_dir, f"{model_name}_classification_report.csv")
        pd.DataFrame(report).transpose().to_csv(report_path)
        mlflow.log_artifact(report_path)

    def log_confusion_matrix(self, model_name, y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        cm_path = os.path.join(self.report_dir, f"{model_name}_confusion_matrix.png")
        plt.savefig(cm_path, bbox_inches="tight")
        mlflow.log_artifact(cm_path)

    def log_curves(self, model_name, y_test, y_proba):

        # Calcul de la courbe ROC et de l'AUC
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_value = roc_auc_score(y_test, y_proba)
        
        # Affichage de la courbe ROC avec AUC
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_value:.2f})")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {model_name}")
        plt.legend()
        
        # Sauvegarde de la figure
        roc_path = os.path.join(self.report_dir, f"{model_name}_roc_curve.png")
        plt.savefig(roc_path)
        plt.close()  # Ferme la figure pour libérer la mémoire
        
        # Enregistrement de l'AUC et de la figure dans MLflow
        mlflow.log_artifact(roc_path)
        mlflow.log_metric(f"{model_name}_auc", auc_value)


    def log_residuals(self, model_name, y_test, y_pred, y_proba):
        residuals = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        if y_proba is not None:
            residuals['Proba'] = y_proba
        residuals_path = os.path.join(self.report_dir, f"{model_name}_residuals.csv")
        residuals.to_csv(residuals_path, index=False)
        mlflow.log_artifact(residuals_path)

    def log_feature_importance(self, model_name, model, X_test):
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feature_names = X_test.columns
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values(by="Importance", ascending=False)
            importance_path = os.path.join(self.report_dir, f"{model_name}_feature_importance.csv")
            importance_df.to_csv(importance_path, index=False)
            mlflow.log_artifact(importance_path)
