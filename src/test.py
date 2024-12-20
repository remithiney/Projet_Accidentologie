import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
import pandas as pd
from imblearn.over_sampling import SMOTENC

# Charger les données
# Assurez-vous que ce chemin correspond à votre fichier de données
file_path = '../data/processed/merged_data_2019_2022.csv'  # Remplacez par le chemin correct

data = pd.read_csv(file_path)
target_column = 'grav'  # Colonne cible

# Identifier les colonnes catégoriques et numériques
categorical_cols = ['col_cat1', 'col_cat2']  # Remplacez par vos colonnes catégoriques
numerical_cols = list(set(data.columns) - set(categorical_cols) - {target_column})

# Séparer les caractéristiques et la cible
X = data.drop(columns=[target_column])
y = data[target_column]

# Indices des colonnes catégoriques
categorical_indices = [X.columns.get_loc(col) for col in categorical_cols]

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Distribution des classes avant SMOTENC
before_smote = Counter(y_train)

# Appliquer SMOTENC
smotenc = SMOTENC(categorical_features=categorical_indices, random_state=42, sampling_strategy='minority')
X_resampled, y_resampled = smotenc.fit_resample(X_train, y_train)

# Distribution des classes après SMOTENC
after_smote = Counter(y_resampled)

# Tracer les distributions
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Distribution avant SMOTENC
ax[0].bar(before_smote.keys(), before_smote.values(), color='blue', alpha=0.7)
ax[0].set_title("Distribution des classes avant SMOTENC")
ax[0].set_xlabel("Classes")
ax[0].set_ylabel("Nombre d'échantillons")

# Distribution après SMOTENC
ax[1].bar(after_smote.keys(), after_smote.values(), color='green', alpha=0.7)
ax[1].set_title("Distribution des classes après SMOTENC")
ax[1].set_xlabel("Classes")
ax[1].set_ylabel("Nombre d'échantillons")

plt.tight_layout()
plt.show()
