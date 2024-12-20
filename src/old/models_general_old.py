# %%
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.utils import resample
from sklearn.metrics import balanced_accuracy_score, classification_report, ConfusionMatrixDisplay, roc_auc_score, roc_curve, auc, precision_recall_curve
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from models.feature_selector import FeatureSelector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os


# %%
def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop(columns=['grav'])
    y = data['grav'] - 1  # Réduction des classes (étiquettes 0 à 2)
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# %%
def balance_dataset(X, y, sample_size=None, random_state=42):
    # Combiner les données pour faciliter l'échantillonnage
    data = X.copy()
    data['target'] = y

    # Déterminer la taille de l'échantillon par classe
    if sample_size is None:
        sample_size = data['target'].value_counts().min()

    # Créer un échantillon équilibré
    balanced_data = pd.concat([
        resample(data[data['target'] == cls],
                 replace=False,  # Pas de sur-échantillonnage
                 n_samples=sample_size,
                 random_state=random_state)
        for cls in data['target'].unique()
    ])
    
    # Séparer X et y après équilibrage
    X_balanced = balanced_data.drop(columns=['target'])
    y_balanced = balanced_data['target']
    
    return X_balanced, y_balanced


# %%
def build_pipeline(model, numerical_cols, categorical_cols, method= "model"):
    # Preprocessor
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler())
    ])
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])

    # Complete pipeline
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selector', FeatureSelector(method=method)),
        ('smote', SMOTE(sampling_strategy='not majority', random_state=42)),
        ('classifier', model)
    ])
    return pipeline

# %%
def evaluate_model(model, X_test, y_test, model_name, save_reports=True, report_dir="reports"):
    # Prédire les étiquettes pour les données de test
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Calculer les métriques
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(f"\n=== Evaluation for {model_name} ===")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))  # Affichage console

    # Matrice de confusion
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues')
    plt.title(f"Matrice de confusion - {model_name}")
    plt.show()

    # Courbe ROC et AUC
    if y_proba is not None:
        fpr, tpr, thresholds = roc_curve(y_test, y_proba, pos_label=1)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
        plt.title(f"Courbe ROC - {model_name}")
        plt.xlabel("Taux de Faux Positifs (FPR)")
        plt.ylabel("Taux de Vrais Positifs (TPR)")
        plt.legend()
        plt.show()
    else:
        roc_auc = None

    # Sauvegarde des résultats
    if save_reports:
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, f"{model_name}_classification_report.csv")
        pd.DataFrame(report).transpose().to_csv(report_path)
        print(f"Classification report saved to {report_path}")

    # Retourner toutes les métriques clés
    return {
        "balanced_accuracy": bal_acc,
        "classification_report": report,
        "roc_auc": roc_auc
    }


# %%
def save_training_sample(X, y, sample_size=1000, file_path="training_sample.csv"):
    # Combiner X et y pour créer un DataFrame complet
    data = X.copy()
    data['target'] = y

    # Sélectionner un échantillon aléatoire
    sample = data.sample(n=sample_size, random_state=42)

    # Sauvegarder l'échantillon dans un fichier CSV
    sample.to_csv(file_path, index=False)
    print(f"Échantillon aléatoire sauvegardé dans {file_path}")


# %%
def test_models(models, X_train, y_train, X_test, y_test, numerical_cols, categorical_cols, report_dir="reports"):
    results = []
    detailed_reports = {}
    methods = ["not", "correlation", "model"]
    
    for model_name, (model, param_grid) in models.items():
        for method in methods:
            print(f"\n=== Testing model: {model_name} with feature selection method: {method} ===")

            # Construire le pipeline
            pipeline = build_pipeline(model, numerical_cols, categorical_cols, method=method)

            start_time = time.time()
            # Optimisation des hyperparamètres
            random_search = RandomizedSearchCV(
                pipeline, 
                param_distributions=param_grid, 
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='balanced_accuracy', 
                n_iter=10, 
                n_jobs=-1, 
                random_state=42
            )
            random_search.fit(X_train, y_train)
            elapsed_time = time.time() - start_time
            print(f'{elapsed_time} secondes.')
            best_model = random_search.best_estimator_
            best_params = random_search.best_params_

            # Évaluer le modèle
            model_name_with_method = f"{model_name}_{method}"
            evaluation_results = evaluate_model(
                best_model, X_test, y_test, model_name=model_name_with_method, report_dir=report_dir
            )

            # Sauvegarder les rapports détaillés
            detailed_reports[model_name_with_method] = evaluation_results["classification_report"]

            # Ajouter les résultats principaux dans la liste
            results.append({
                'model': model_name_with_method,
                'balanced_accuracy': evaluation_results["balanced_accuracy"],
                'roc_auc': evaluation_results["roc_auc"],  # Peut être None si le modèle ne supporte pas predict_proba
                'best_params': best_params,
                'training_time_seconds': elapsed_time
            })
    
    # Résultats globaux sous forme de DataFrame
    return pd.DataFrame(results), detailed_reports


# %%
# Main Execution
X_train, X_test, y_train, y_test = load_data('../data/processed/merged_data_2019_2022.csv')

numerical_columns = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_columns = X_train.select_dtypes(include=['object']).columns

X_train_balanced, y_train_balanced = balance_dataset(X_train, y_train)

save_training_sample(X_train_balanced, y_train_balanced, sample_size=1000, file_path="balanced_training_sample.csv")
# Models and their hyperparameters
models = {
    'RandomForest': (
        RandomForestClassifier(random_state=42),
        {
            'classifier__n_estimators': [50, 100, 150, 200, 300],
            'classifier__max_depth': [5, 10, 15, None],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4]
        }
    ),
    'XGBoost': (
        XGBClassifier(random_state=42, eval_metric='logloss'),
        {
            'classifier__n_estimators': [50, 100, 150, 200],
            'classifier__max_depth': [3, 5, 7, 10],
            'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
            'classifier__subsample': [0.6, 0.8, 1.0],
            'classifier__colsample_bytree': [0.6, 0.8, 1.0]
        }
    ),
    'LogisticRegression': (
        LogisticRegression(random_state=42, max_iter=1000),
        {
            'classifier__C': [0.001, 0.01, 0.1, 1, 10],
            'classifier__penalty': ['l2', 'none'],
            'classifier__solver': ['lbfgs', 'saga']
        }
    ),
    'KNN': (
        KNeighborsClassifier(),
        {
            'classifier__n_neighbors': [3, 5, 7, 9, 11],
            'classifier__weights': ['uniform', 'distance'],
            'classifier__p': [1, 2]
        }
    ),
    'AdaBoost': (
        AdaBoostClassifier(random_state=42),
        {
            'classifier__n_estimators': [50, 100, 150, 200],
            'classifier__learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0]
        }
    ),
    'GradientBoosting': (
        GradientBoostingClassifier(random_state=42),
        {
            'classifier__n_estimators': [50, 100, 150, 200],
            'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
            'classifier__max_depth': [3, 5, 7]
        }
    )
}

# Test and evaluate models
results_df, reports = test_models(models, X_train_balanced, y_train_balanced, X_test, y_test, numerical_columns, categorical_columns)

# Save results to CSV
results_df.to_csv('model_comparison_results.csv', index=False)

results_df.plot(x='model', y='balanced_accuracy', kind='bar', title='Comparaison des modèles')
plt.ylabel('Balanced Accuracy')
plt.show()