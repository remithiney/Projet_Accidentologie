from sklearn.metrics import classification_report, roc_curve, auc, f1_score, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import traceback
import os


class Evaluator:
    def __init__(self, logger, save_curves=True):
        self.logger = logger
        self.save_curves = save_curves

    def evaluate_model(self, model, X_test, y_test, model_name, report_dir="reports"):
        try:
            self.logger.info(f"Evaluating model: {model_name}")
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

            # Calcul des métriques principales
            f1 = f1_score(y_test, y_pred, average='weighted')
            report = classification_report(y_test, y_pred, output_dict=True)
            self.logger.info(f"F1-score: {f1:.4f}")
            self.logger.info(f"W - Precision: {report['weighted avg']['precision']:.4f}")
            self.logger.info(f"W - Recall: {report['weighted avg']['recall']:.4f}")

            # Sauvegarde du rapport de classification
            os.makedirs(report_dir, exist_ok=True)
            report_path = os.path.join(report_dir, f"{model_name}_classification_report.csv")
            pd.DataFrame(report).transpose().to_csv(report_path)
            self.logger.info(f"Classification report saved to {report_path}")

            metrics = {'f1_score': f1}

            # Cm
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap=plt.cm.Blues)
            plt.title(f"Confusion Matrix - {model_name}")
            cm_path = report_dir + "cm_" + model_name
            plt.savefig(cm_path, bbox_inches="tight")

            # Courbe ROC
            if y_proba is not None:
                fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label=1)
                roc_auc = auc(fpr, tpr)
                self.logger.info(f"ROC AUC: {roc_auc:.4f}")
                metrics['roc_auc'] = roc_auc

                plt.figure()
                plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
                plt.legend()
                plt.title(f"ROC Curve - {model_name}")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                if self.save_curves:
                    roc_path = os.path.join(report_dir, f"{model_name}_roc_curve.png")
                    plt.savefig(roc_path)
                    self.logger.info(f"ROC curve saved to {roc_path}")
                plt.show()

                # Courbe PR
                precision, recall, _ = precision_recall_curve(y_test, y_proba)
                plt.figure()
                plt.plot(recall, precision, label="Precision-Recall Curve")
                plt.legend()
                plt.title(f"Precision-Recall Curve - {model_name}")
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                if self.save_curves:
                    pr_path = os.path.join(report_dir, f"{model_name}_pr_curve.png")
                    plt.savefig(pr_path)
                    self.logger.info(f"Precision-Recall curve saved to {pr_path}")
                plt.show()

            # Sauvegarde des résidus
            residuals = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
            if y_proba is not None:
                residuals['Proba'] = y_proba
            residuals_path = os.path.join(report_dir, f"{model_name}_residuals.csv")
            residuals.to_csv(residuals_path, index=False)
            self.logger.info(f"Residuals saved to {residuals_path}")

            return metrics
        except Exception as e:
            self.logger.error("Error during model evaluation.")
            self.logger.error(traceback.format_exc())
            raise
