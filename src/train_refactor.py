
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import xgboost as xgb 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import mlflow
import mlflow.xgboost
import os



import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------- Paramètres globaux ------------------------- #
_FILE_PATH = '../data/processed/merged_data_2019_2022.csv'
#_MODEL_PATH = '../models/best_xgb_pipeline.pkl'
_RANDOM_STATE = 42
_STRATIFIED_SPLITS = 5
_VARIANCE_THRESHOLD = 0.01
_NUMERICAL_TYPES = ['int64', 'float64']
_CATEGORICAL_TYPES = ['object', 'category']
_REPLACE_VALUE_NUM = -1
_REPLACE_VALUE_CAT = "-1"
_REPLACE_NUM_STRATEGY = "mean"
_REPLACE_CAT_STRATEGY = "most_frequent"
_RANDOMIZED_SEARCH_ITER = 30
_RANDOMIZED_SEARCH_SCORING = "accuracy"
#_DO_LAST_CV = False
_TRESHOLD_PROBA = 0.6
_CHOSEN_MODEL_NAME = "random_forest"
_MODEL_FILENAME = f"../models/{_CHOSEN_MODEL_NAME}_{_TRESHOLD_PROBA}.joblib"
_COL_TO_DROP = ['an','tpc',"an_nais"]



    

# ------------------------- Paramètres model ------------------------- #

model_configs = {
    "xgboost": {
        "estimator": xgb.XGBClassifier(eval_metric="logloss", random_state=_RANDOM_STATE),
        "param_distributions": {
            "clf__n_estimators": np.arange(100, 600, 100),
            #"clf__num_boost_round": np.arange(100, 600, 100),
            "clf__max_depth": np.arange(2, 9, 2),
            "clf__learning_rate": np.logspace(-2, -0.7, 5),
            "clf__subsample": np.linspace(0.6, 1.0, 4),
            "clf__colsample_bytree": np.linspace(0.7, 1.0, 4),
            #"clf__early_stopping_rounds": np.arange(10, 51, 10),
        },
        "log_model_function": mlflow.xgboost.log_model
    },
    "random_forest": {
        "estimator": RandomForestClassifier(random_state=_RANDOM_STATE),
        "param_distributions": {
            "clf__n_estimators": np.arange(50, 201, 50),
            "clf__max_depth": np.arange(2, 12, 2),
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4],
            "clf__bootstrap": [True, False]
        },
        "log_model_function": mlflow.sklearn.log_model
    },
    "logistic_regression": {
        "estimator": LogisticRegression(random_state=_RANDOM_STATE, solver='saga', max_iter=1000),
        "param_distributions": {
            "clf__penalty": ["l1", "l2", "elasticnet"],
            "clf__C": np.logspace(-2, 1, 5),
            "clf__l1_ratio": np.linspace(0, 1, 5)
        },
        "log_model_function": mlflow.sklearn.log_model
    }
}


# ------------------------- Chargement des données ------------------------- #
data = pd.read_csv(_FILE_PATH)
X = data.drop(columns=['grav'])
y = data['grav']

X.drop(columns=_COL_TO_DROP, inplace=True)

numerical_columns = X.select_dtypes(include=_NUMERICAL_TYPES).columns.tolist()
categorical_columns = X.select_dtypes(include=_CATEGORICAL_TYPES).columns.tolist()

X[categorical_columns] = X[categorical_columns].astype(str)

X[numerical_columns] = X[numerical_columns].replace(_REPLACE_VALUE_NUM, np.nan)
X[categorical_columns] = X[categorical_columns].replace(_REPLACE_VALUE_CAT, np.nan)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=_RANDOM_STATE)


# ------------------------- Construction du pipeline ------------------------- #
def build_pipeline(numerical_columns, categorical_columns, base_estimator): 
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy=_REPLACE_NUM_STRATEGY)),
        ("scaler", MinMaxScaler())
    ])
    
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy=_REPLACE_CAT_STRATEGY)),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ])
    
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numerical_columns),
        ("cat", categorical_transformer, categorical_columns)
    ])
    
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("variance", VarianceThreshold(threshold=_VARIANCE_THRESHOLD)),
        ("undersampler", RandomUnderSampler(random_state=_RANDOM_STATE)),
        ("pca", PCA(n_components=0.9)),
        #("LDA", LDA()),
        ("clf", base_estimator)
    ])
    
    return pipeline




# ------------------------- Recherche des meilleurs hyperparamètres avec RandomizedSearchCV ------------------------- #


chosen_config = model_configs[_CHOSEN_MODEL_NAME]
base_estimator = chosen_config["estimator"]
param_distributions = chosen_config["param_distributions"]
log_model_fn = chosen_config["log_model_function"]

pipeline = build_pipeline(numerical_columns, categorical_columns, base_estimator)

cv = StratifiedKFold(n_splits=_STRATIFIED_SPLITS, shuffle=True, random_state=_RANDOM_STATE)

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=_RANDOMIZED_SEARCH_ITER,
    scoring=_RANDOMIZED_SEARCH_SCORING,
    cv=cv,
    n_jobs=-1,
    verbose=3,
    random_state=_RANDOM_STATE
)

random_search.fit(X_train, y_train)
best_pipe = random_search.best_estimator_

joblib.dump(best_pipe, _MODEL_FILENAME)
print(f'Pipeline saved as: {_MODEL_FILENAME}.')

# --- MLflow logging --- #
mlflow.set_experiment("My_Accidents_Experiment")

with mlflow.start_run():
    mlflow.log_param("model_name", _CHOSEN_MODEL_NAME)
    
    y_pred_proba = best_pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= _TRESHOLD_PROBA).astype(int)
    
    mlflow.log_param("threshold_proba", _TRESHOLD_PROBA)
    
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    precision = conf_matrix[1, 1] / (conf_matrix[0, 1] + conf_matrix[1, 1])
    recall = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
    f1 = 2 * (precision * recall) / (precision + recall)

    mlflow.log_metrics({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Classe 0", "Classe 1"],
                yticklabels=["Classe 0", "Classe 1"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    confusion_matrix_path = "confusion_matrix.png"
    plt.savefig(confusion_matrix_path)
    mlflow.log_artifact(confusion_matrix_path)
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    roc_curve_path = "roc_curve.png"
    plt.savefig(roc_curve_path)
    mlflow.log_artifact(roc_curve_path)
    plt.show()

    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(recall_vals, precision_vals, color='green', lw=2, label="Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    pr_curve_path = "precision_recall_curve.png"
    plt.savefig(pr_curve_path)
    mlflow.log_artifact(pr_curve_path)
    plt.show()


    final_model = best_pipe.named_steps["clf"]
    log_model_fn(final_model, "model") 

    print("Run complete.")
