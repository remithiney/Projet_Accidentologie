# Import des bibliothèques nécessaires
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Fonction de preprocessing
def create_preprocessor(numerical_columns, categorical_columns):
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler())
    ])
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_columns),
        ('cat', categorical_pipeline, categorical_columns)
    ])
    return preprocessor

def find_optimal_variance_threshold(X, y, preprocessor, sample_frac=0.1, thresholds=np.arange(0.01, 0.25, 0.05)):
    # Échantillon des données
    df_sample = X.sample(frac=sample_frac, random_state=42)
    y_sample = y.loc[df_sample.index]
    
    # Repasser les données par le pipeline
    X_p_sample = preprocessor.fit_transform(df_sample)
    
    model_scores = []
    for t in thresholds:
        selector = VarianceThreshold(threshold=t)
        X_reduced = selector.fit_transform(X_p_sample)
        
        # Modèle d'évaluation
        model = RandomForestClassifier(random_state=42)
        score = cross_val_score(model, X_reduced, y_sample, cv=3, scoring='accuracy').mean()
        model_scores.append(score)
    
    # Trouver le seuil optimal
    optimal_threshold = thresholds[np.argmax(model_scores)]
    print(f"Meilleur seuil de variance : {optimal_threshold}")
    
    # Visualisation des résultats
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, model_scores, marker='o', color='b')
    plt.axvline(optimal_threshold, color='r', linestyle='--', label=f'Seuil optimal = {optimal_threshold}')
    plt.xlabel("Seuil de variance")
    plt.ylabel("Performance (Accuracy)")
    plt.title("Impact du seuil de variance sur les performances")
    plt.legend()
    plt.grid()
    plt.show()
    
    return optimal_threshold

# Fonction pour trouver les k meilleures caractéristiques selon un modèle
def find_optimal_features_using_model(X, y, threshold=0.95):
    
    # Entraîner un modèle basé sur les arbres
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X, y)
    
    # Extraire les importances des caractéristiques
    feature_importances = model.feature_importances_
    
    # Calculer l'importance cumulée
    sorted_indices = np.argsort(feature_importances)[::-1]
    cumulative_importance = np.cumsum(feature_importances[sorted_indices])
    
    # Trouver le nombre minimal de caractéristiques pour atteindre le seuil
    optimal_k = np.argmax(cumulative_importance >= threshold) + 1
    print(f"Nombre optimal de caractéristiques pour {threshold * 100}% de l'importance cumulée : {optimal_k}")
    
    # Visualisation
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_importance) + 1), cumulative_importance, marker='o', label='Importance cumulée')
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'{threshold * 100}% d\'importance')
    plt.axvline(x=optimal_k, color='g', linestyle='--', label=f'k optimal = {optimal_k}')
    plt.xlabel("Nombre de caractéristiques sélectionnées")
    plt.ylabel("Importance cumulée")
    plt.title("Importance cumulée des caractéristiques")
    plt.legend()
    plt.show()
    
    return optimal_k, sorted_indices[:optimal_k]

# Chargement des données
df = pd.read_csv('../data/processed/merged_data_2019_2022.csv')
X = df.drop(columns=['grav'])
y = df['grav']

# Définir les colonnes numériques et catégoriques
numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Créer le preprocessor
preprocessor = create_preprocessor(numerical_columns, categorical_columns)

# Appliquer le pipeline sur les données complètes
X_p = preprocessor.fit_transform(X)

# Trouver le seuil optimal de variance
optimal_threshold = find_optimal_variance_threshold(X, y, preprocessor)

# Appliquer VarianceThreshold avec le seuil optimal
var_selector = VarianceThreshold(threshold=optimal_threshold)
X_var_selected = var_selector.fit_transform(X_p)
print(f"Nombre de colonnes après VarianceThreshold : {X_var_selected.shape[1]}")

# Utiliser find_optimal_features_using_model après VarianceThreshold
optimal_k, top_features_indices = find_optimal_features_using_model(X_var_selected, y, threshold=0.95)

# Récupérer les noms des colonnes après transformation
categorical_columns_transformed = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_columns)
all_columns_after_preprocessing = numerical_columns + list(categorical_columns_transformed)

# Obtenir les indices des colonnes sélectionnées par VarianceThreshold
selected_indices_after_variance = var_selector.get_support(indices=True)

# Mapper les indices des colonnes importantes (après le modèle) aux colonnes originales
selected_columns = [
    all_columns_after_preprocessing[selected_indices_after_variance[i]] for i in top_features_indices
]

print(f"Nombre de colonnes sélectionnées après RandomForest : {len(selected_columns)}")
print(f"Colonnes sélectionnées : {selected_columns}")
