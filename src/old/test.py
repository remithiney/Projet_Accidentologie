import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTENC

# Charger les données
file_path = "../data/processed/merged_data_2019_2022.csv"  # Remplacez par le chemin correct
data = pd.read_csv(file_path)
target_column = "grav"

# Échantillonner 10% des données pour réduire le temps de calcul
data_sampled = data.sample(frac=0.1, random_state=42)

# Identification des colonnes
numerical_cols = data_sampled.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = data_sampled.select_dtypes(include=['object', 'category']).columns.tolist()

X = data_sampled.drop(columns=[target_column])
y = data_sampled[target_column]

# Afficher la distribution avant SMOTENC
print(f"Distribution avant SMOTENC: {Counter(y)}")

# Appliquer SMOTENC
categorical_indices = [X.columns.get_loc(col) for col in categorical_cols]
smotenc = SMOTENC(categorical_features=categorical_indices, random_state=42)
X_resampled, y_resampled = smotenc.fit_resample(X, y)

# Afficher la distribution après SMOTENC
print(f"Distribution après SMOTENC: {Counter(y_resampled)}")
