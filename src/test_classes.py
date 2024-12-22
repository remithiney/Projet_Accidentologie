import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Charger le jeu de données
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Séparer les données en deux dataframes basés sur la colonne cible 'grav'
def split_by_target(data, target_column):
    indemne = data[data[target_column] == 1]  # Classe 'indemne'
    grave = data[data[target_column] == 2]    # Classe 'grave'
    return indemne, grave

# Générer une matrice de corrélation et l'afficher
def plot_correlation_matrix(df, title):
    correlation_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Afficher les statistiques descriptives côte à côte
def compare_describes(df1, df2, df1_name, df2_name):
    describe_df1 = df1.describe().T
    describe_df2 = df2.describe().T

    compare_df = pd.concat([describe_df1, describe_df2], axis=1, keys=[df1_name, df2_name])
    print(compare_df)
    return compare_df

# Comparer les variables qualitatives
def compare_categorical(df1, df2, df1_name, df2_name):
    categorical_columns = df1.select_dtypes(include=['object', 'category']).columns
    comparison_results = {}

    for col in categorical_columns:
        df1_counts = df1[col].value_counts(normalize=True)
        df2_counts = df2[col].value_counts(normalize=True)

        comparison = pd.concat([df1_counts, df2_counts], axis=1, keys=[df1_name, df2_name]).fillna(0)
        comparison_results[col] = comparison

        print(f"Comparaison pour la variable '{col}':")
        print(comparison)
        print("\n")

    return comparison_results

# Main
if __name__ == "__main__":
    # Chemin vers le fichier CSV
    file_path = "../data/processed/merged_data_2019_2022.csv"
    target_column = "grav"

    # Charger les données
    data = load_data(file_path)

    # Créer les dataframes 'indemne' et 'grave'
    indemne, grave = split_by_target(data, target_column)

    print(f"Nombre de lignes pour 'indemne': {len(indemne)}")
    print(f"Nombre de lignes pour 'grave': {len(grave)}")

    # Comparer les statistiques descriptives
    compare_describes(indemne, grave, "Indemne", "Grave")

    # Comparer les variables qualitatives
    compare_categorical(indemne, grave, "Indemne", "Grave")
