# %%
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from IPython.display import display, HTML
import csv
import os
import logging
import math
import re
import json
import seaborn as sns

# %%
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# %%
PATH_TO_CSVS = 'data/raw'
PATH_TO_JSON = 'notebooks/descvar.json'

# %%
# https://stackoverflow.com/questions/46135839/auto-detect-the-delimiter-in-a-csv-file-using-pd-read-csv comme base
# detecte automatiquement le sep d'un fichier csv


def get_delimiter(file_path, bytes=4096):
    try:
        with open(file_path, 'r') as file:
            data = file.read(bytes)
        sniffer = csv.Sniffer()
        delimiter = sniffer.sniff(data).delimiter
        return delimiter
    except Exception as e:
        logging.error(f"Erreur lors de la détection du délimiteur: {e}")
        return None

# %%
# lecture d'un fichier csv en essayant différents encodage.


def read_csv_file(file_path):
    if not os.path.exists(file_path):
        return None, False, f"Fichier non trouvé: {file_path}"
    
    delimiter = get_delimiter(file_path)
    if not delimiter:
        return None, False, f"Impossible de détecter le délimiteur pour le fichier: {file_path}"
    
    encodings = ['utf-8', 'latin1', 'ISO-8859-1']
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, low_memory=False, encoding=encoding, delimiter=delimiter)
            return df, True, None
        except (UnicodeDecodeError, pd.errors.ParserError) as e:
            logging.warning(f"Erreur avec l'encodage {encoding} pour le fichier {file_path}: {e}")
    
    return None, False, f"Impossible de lire le fichier {file_path} avec les encodages: {encodings}."

# %%
# chargement des datasets dans des listes


def load_datasets(prefixes, years, base_path= PATH_TO_CSVS):
    dataframes = []
    
    for prefix in prefixes:
        datasets = []
        for year in years:
            connector = '_' if year <= 2016 else '-'
            file_name = os.path.join(base_path, f'{prefix}{connector}{year}.csv')
            df, success, error = read_csv_file(file_name)
            if success:
                datasets.append({file_name: df})
            else:
                logging.error(error)
        dataframes.append(datasets)
    
    return dataframes

# %%
years = list(range(2005, 2023))
prefixes= ['caracteristiques', 'lieux', 'usagers', 'vehicules']

dataframes = load_datasets(prefixes, years)

# log
for prefix, df_list in zip(prefixes, dataframes):
        logging.info(f'{prefix}: {len(df_list)} datasets chargés.')

logging.info(f'Total datasets chargés: {sum(len(dfs) for dfs in dataframes)}.')

# %%
def extract_year(file_name):
    match = re.search(r'(\d{4})\.csv$', file_name)
    if match:
        return match.group(1)
    else:
        return None

# %%
#génère le menu
def generate_navigation_menu(columns, dataset_name):
    links = [f'<a href="#{dataset_name}_{col}">{col}</a>' for col in columns]
    return f'<div id="menu_{dataset_name}"><h2>Menu {dataset_name}</h2><ul>{"".join(f"<li>{link}</li>" for link in links)}</ul></div>'

# %%
# Fonction pour générer des résumés des datasets
def summarize_dataset(dataset_dict, dataset_name):
    summary = {}
    
    for year, df in dataset_dict.items():
        dataset_key = f"{dataset_name}_{year}"
        summary[dataset_key] = {
            'Dataset Name': dataset_key,
            'Total Rows': df.shape[0],
            'Total Columns': df.shape[1],
            'Missing Values (%)': df.isna().mean().round(4) * 100,
            'Most Frequent Values': df.mode().iloc[0]
        }
        
        # Affichage des résumés
        print(f"Résumé pour {dataset_key}:")
        print(f"Total Rows: {summary[dataset_key]['Total Rows']}, Total Columns: {summary[dataset_key]['Total Columns']}")
        print(f"Pourcentage de valeurs manquantes par colonne:\n{summary[dataset_key]['Missing Values (%)']}")
        print(f"Valeurs les plus fréquentes:\n{summary[dataset_key]['Most Frequent Values']}")
        print("="*50)

        # Affichage d'un barplot des valeurs manquantes
        plt.figure(figsize=(12, 6))
        missing_values = summary[dataset_key]['Missing Values (%)']
        missing_values[missing_values > 0].sort_values(ascending=False).plot(kind='bar', color='orange')
        plt.title(f'Pourcentage de valeurs manquantes par colonne - {dataset_key}')
        plt.xlabel('Colonnes')
        plt.ylabel('Pourcentage de valeurs manquantes (%)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
    return summary

# %%
#création d'une grille de boxplot
def create_boxplot_grid(column, datasets):

    if not any(pd.api.types.is_numeric_dtype(df[column]) for dataset in datasets for file_name, df in dataset.items() if column in df.columns):
        logging.info(f"La colonne {column} n'est pas numérique.")
        return

    n_datasets = len(datasets)
    n_cols = 5
    n_rows = math.ceil(n_datasets / n_cols)
    
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(16, n_rows * 2), squeeze=False)
    fig.suptitle(f'Boxplots pour: {column}', fontsize=16)
    
    for ax, dataset in zip(axes.flatten(), datasets):
        for file_name, df in dataset.items():
            if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
                ax.boxplot(df[column].dropna(), vert=True)
                ax.set_title(extract_year(file_name))
                ax.set_xlabel(column)
                ax.set_ylabel('Valeurs')
    
    # Masquer les axes non utilisés
    for ax in axes.flatten()[n_datasets:]:
        ax.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# %%
#création du bar plot pour visualiser la distribution des modalités.
def create_total_stacked_barplot(column, datasets, max_modalities=50):
    modality_counts = {}

    for dataset in datasets:
        for file_name, df in dataset.items():
            if column in df.columns:
                modality_count = df[column].value_counts()
                modality_counts[file_name] = modality_count
    
    all_modalities = set()
    for counts in modality_counts.values():
        all_modalities.update(counts.index)
    
    if len(all_modalities) > max_modalities:
        logging.warning(f"Le nombre de modalités uniques dans la colonne {column} excède le seuil de {max_modalities}. Auncun bar plot généré.")
        return
    
    modality_data = {modality: [] for modality in all_modalities}
    years = [extract_year(file_name) for file_name in modality_counts.keys()]
    
    for modality in all_modalities:
        for file_name in modality_counts.keys():
            count = modality_counts[file_name].get(modality, 0)
            modality_data[modality].append(count)
    
    df_modalities = pd.DataFrame(modality_data, index=years).transpose()
    
    df_modalities.plot(kind='bar', stacked=True, figsize=(15, 7), colormap='viridis')
    plt.title(f'{column}')
    plt.xlabel('Modalités')
    plt.ylabel('Count')
    plt.legend(title='Years', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# %%
# evolution de la répartition 
def create_lineplot_evolution(column, datasets, max_modalities=12):
    modality_counts = {}

    for dataset in datasets:
        for file_name, df in dataset.items():
            if column in df.columns:
                year = extract_year(file_name)
                modality_count = df[column].value_counts(normalize=True) * 100
                if year not in modality_counts:
                    modality_counts[year] = modality_count
                else:
                    modality_counts[year] = modality_counts[year].add(modality_count, fill_value=0)
    
    all_modalities = set()
    for counts in modality_counts.values():
        all_modalities.update(counts.index)
    
    if len(all_modalities) > max_modalities:
        logging.warning(f"Le nombre de modalités uniques dans la colonne {column} excède le seuil de {max_modalities}. Aucun graphique en ligne généré.")
        return
    
    modality_data = {modality: [] for modality in all_modalities}
    years = sorted(modality_counts.keys())
    
    for modality in all_modalities:
        for year in years:
            count = modality_counts[year].get(modality, 0)
            modality_data[modality].append(count)
    
    df_modalities = pd.DataFrame(modality_data, index=years)
    
    df_modalities.plot(kind='line', figsize=(15, 7), marker='o')
    plt.title(f'Evolution de la distribution {column}')
    plt.xlabel('Années')
    plt.ylabel('Proportion (%)')
    plt.legend(title='Modalités', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# %%
# permet d'afficher les info depuis le json
def display_variable_info(col_name, dataset_name, descvar):
    dataset_name_upper = dataset_name.upper()
    if dataset_name_upper in descvar:
        section = descvar[dataset_name_upper]
        if col_name in section:
            details = section[col_name]
            if isinstance(details, dict):    
                description = details.get('description', 'Pas de description.')
                modalities = details.get('modalities', {})
                
                print(f"Description: {description}")
                if modalities:
                    print("\nModalités:")
                    for key, value in modalities.items():
                        print(f"- {key}: {value}")
            else:
                print(f"Description: {details}")
            return
    logging.info(f"Pas de description pour `{col_name}` dans la section `{dataset_name_upper}`.")

# %%
# analyse une colonne unique
def analyze_column(column, datasets, total_rows, dataset_name, descvar= None):
    column_results = []
    
    for dataset in datasets:
        for file_name, df in dataset.items():
            if column in df.columns:
                col_type = df[column].dtype
                col_mode = df[column].mode()[0] if not df[column].mode().empty else "N/A"
                null_proportion_file = df[column].isnull().mean() *100
                null_proportion_total = df[column].isnull().sum() / total_rows
                column_results.append([
                    extract_year(file_name), 
                    col_type, 
                    col_mode, 
                    null_proportion_file, 
                    null_proportion_total
                ])
    
    if column_results:
        display(HTML(f'<div id="{dataset_name}_{column}"><h2>Colonne: {column}</h2>'))
        print(f"lignes: {total_rows}\n")
        display_variable_info(column, dataset_name, descvar)
        print(tabulate(column_results, headers=[
            "Année", "Type", "Mode", 
            "Proportion valeurs nulles (fichier)", 
            "Proportion valeurs nulles (total)"
        ]))
        
        #if any(pd.api.types.is_numeric_dtype(df[column]) for dataset in datasets for file_name, df in dataset.items() if column in df.columns):
        create_boxplot_grid(column, datasets) 
        create_total_stacked_barplot(column, datasets)
        create_lineplot_evolution(column, datasets)

        display(HTML(f'<p><a href="#menu_{dataset_name}">Retour au menu</a></p></div>'))


# %%
# on analyse toutes les colonnes à la suite
def analyze_all_columns(datasets, dataset_name, descvar= None):
    
    logging.info(f'Chargement de {dataset_name}.')
    
    all_columns = set()

    for dataset in datasets: 
        for file_name, df in dataset.items():
            all_columns.update(df.columns)
    
    total_rows = sum(df.shape[0] for dataset in datasets for file_name, df in dataset.items())
    
    navigation_menu = generate_navigation_menu(all_columns, dataset_name) 
    display(HTML(navigation_menu))

    #traitement pour chaque colonne
    for column in all_columns:
        analyze_column(column, datasets, total_rows, dataset_name, descvar= descvar)
            

# %%
with open(PATH_TO_JSON, encoding= 'utf-8') as f:
    descvar = json.load(f)

for dataset_name, datasets in zip(prefixes, dataframes):
    # Résumé de chaque fichier
    for dataset in datasets: 
        summarize_dataset(dataset, dataset_name)
        
    # Analyse de chaque colonne
    analyze_all_columns(datasets, dataset_name, descvar= descvar)


