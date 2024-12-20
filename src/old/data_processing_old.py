# %%
import pandas as pd
import numpy as np
import os
import json
import logging
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor
from data_loading import DataLoader
from loghandler import LogHandlerManager

# %%
# Load configuration file
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Log 
log_handler_manager = LogHandlerManager(config["logs"])
log_handler_manager.start_listener()

# %%
class DataProcessor:
    def __init__(self, config: Dict[str, Any], log_handler_manager: LogHandlerManager):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(log_handler_manager.get_queue_handler())
        self.logger.setLevel(logging.DEBUG)
        
        # Pass logger to DataLoader
        self.data_loader = DataLoader(config["path_to_csvs"], self.logger)
        self.dataframes = self.data_loader.load_datasets_parallel(config["prefixes"], config["years_to_process"])
        self.reference_structures = {}

    def preprocess_datasets(self) -> None:
        for prefix, df_list in self.dataframes.items():    
            # Supprimer les colonnes spécifiées dans 'features_to_drop' pour chaque préfixe
            features_to_drop = self.config.get('features_to_drop', {}).get(prefix, [])
            for i, df in enumerate(df_list):
                columns_to_drop = [col for col in features_to_drop if col in df.columns]
                df.drop(columns=columns_to_drop, inplace=True)
                self.logger.info(f"Colonnes {columns_to_drop} supprimées pour {prefix}-{self.config['years_to_process'][i]}.")

                # Supprimer les colonnes avec 80% ou plus de valeurs manquantes
                missing_threshold = 0.8 * len(df)
                columns_with_many_nans = [col for col in df.columns if df[col].isna().sum() >= missing_threshold]
                df.drop(columns=columns_with_many_nans, inplace=True)
                self.logger.info(f"Colonnes {columns_with_many_nans} supprimées pour {prefix}-{self.config['years_to_process'][i]} en raison de valeurs manquantes.")

                # Supprimer les doublons
                before_dropping_duplicates = len(df)
                df.drop_duplicates(inplace=True)
                after_dropping_duplicates = len(df)
                if before_dropping_duplicates != after_dropping_duplicates:
                    self.logger.info(f"{before_dropping_duplicates - after_dropping_duplicates} doublons supprimés pour {prefix}-{self.config['years_to_process'][i]}.")

            # Prétraitement manuel en fonction du préfixe
            if prefix == "caracteristiques":
                self.preprocess_caracteristiques(df_list, prefix)
            elif prefix == "lieux":
                self.preprocess_lieux(df_list, prefix)
            elif prefix == "usagers":
                self.preprocess_usagers(df_list, prefix)
            elif prefix == "vehicules":
                self.preprocess_vehicules(df_list, prefix)

    def preprocess_caracteristiques(self, df_list: List[pd.DataFrame], prefix: str) -> None:
        for i, df in enumerate(df_list):
            if "Accident_Id" in df.columns:
                df.rename(columns={"Accident_Id": "Num_Acc"}, inplace=True)
                self.logger.info(f"Colonne 'Accident_Id' renommée en 'Num_Acc' pour {prefix}-{self.config['years_to_process'][i]}.")
            if "hrmn" in df.columns:
                df['hr'] = df['hrmn'].str.split(':').str[0]
                df.drop(columns=['hrmn'], inplace=True)
                self.logger.info(f"Colonne 'hrmn' transformée en 'hr' pour {prefix}-{self.config['years_to_process'][i]}.")

    def preprocess_lieux(self, df_list: List[pd.DataFrame], prefix: str) -> None:
        pass

    def preprocess_usagers(self, df_list: List[pd.DataFrame], prefix: str) -> None:
        for i, df in enumerate(df_list):
            if "id_usager" in df.columns:
                missing_ids = df['id_usager'].isna()
                if missing_ids.any():
                    max_id = df['id_usager'].max(skipna=True)
                    next_id = max_id + 1 if pd.notna(max_id) else 1
                    df.loc[missing_ids, 'id_usager'] = range(next_id, next_id + missing_ids.sum())
                    self.logger.info(f"Valeurs manquantes de 'id_usager' remplies avec des valeurs uniques pour {prefix}-{self.config['years_to_process'][i]}.")

    def preprocess_vehicules(self, df_list: List[pd.DataFrame], prefix: str) -> None:
        pass

    def extract_reference_structure(self) -> None:
        for prefix, df_list in self.dataframes.items():
            if df_list:
                # Utiliser le dataframe le plus récent pour définir la structure de référence
                reference_df = df_list[-1]
                reference_structure = {
                    "columns": list(reference_df.columns),
                    "dtypes": reference_df.dtypes.to_dict()
                }
                self.reference_structures[prefix] = reference_structure
                self.logger.info(f"Structure de référence extraite pour {prefix} : {reference_structure}")

    def apply_reference_structure(self, df: pd.DataFrame, reference_structure: Dict[str, Any], dataset_name: str) -> pd.DataFrame:

        reference_columns = reference_structure["columns"]
        reference_dtypes = reference_structure["dtypes"]

        # Supprimer les colonnes qui n'existent pas dans la structure de référence
        df = df[[col for col in df.columns if col in reference_columns]]
        
        # Ajouter les colonnes manquantes et repositionner les colonnes
        for col in reference_columns:
            if col not in df.columns:
                df[col] = pd.NA
        df = df[reference_columns]

        # Convertir les types de colonnes pour correspondre à la structure de référence
        for col, dtype in reference_dtypes.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                except Exception as e:
                    self.logger.warning(f"Impossible de convertir la colonne {col} dans {dataset_name} au type {dtype}: {e}")
        
        return df

    def concatenate_and_save(self) -> None:
        for prefix, df_list in self.dataframes.items():
            if df_list:
                concatenated_df = pd.concat(df_list, ignore_index=True)
                output_file = f"{prefix}_{self.config['years_to_process'][0]}_{self.config['years_to_process'][-1]}_process.csv"
                output_path = os.path.join(self.config["path_to_processed_csv"], output_file)
                concatenated_df.to_csv(output_path, index=False)
                self.logger.info(f"Dataframe concaténé pour {prefix} sauvegardé dans {output_path}")

    def merge_all_dataframes(self) -> pd.DataFrame:
        # Charger les dataframes concaténés pour chaque préfixe
        path_to_processed = self.config["path_to_processed_csv"]
        df_caracteristiques = pd.read_csv(os.path.join(path_to_processed, f"caracteristiques_{self.config['years_to_process'][0]}_{self.config['years_to_process'][-1]}_process.csv"))
        df_lieux = pd.read_csv(os.path.join(path_to_processed, f"lieux_{self.config['years_to_process'][0]}_{self.config['years_to_process'][-1]}_process.csv"))
        df_usagers = pd.read_csv(os.path.join(path_to_processed, f"usagers_{self.config['years_to_process'][0]}_{self.config['years_to_process'][-1]}_process.csv"))
        df_vehicules = pd.read_csv(os.path.join(path_to_processed, f"vehicules_{self.config['years_to_process'][0]}_{self.config['years_to_process'][-1]}_process.csv"))

        # Fusionner les caractéristiques et les lieux
        merged_df = pd.merge(df_caracteristiques, df_lieux, on='Num_Acc', how='left')
        self.logger.info("Caractéristiques et lieux fusionnés avec succès.")

        # Fusionner avec les usagers
        merged_df = pd.merge(merged_df, df_usagers, on='Num_Acc', how='left')
        self.logger.info("Caractéristiques/lieux et usagers fusionnés avec succès.")

        # Fusionner avec les véhicules
        merged_df = pd.merge(merged_df, df_vehicules, on=['Num_Acc', 'id_vehicule', 'num_veh'], how='left')
        self.logger.info("Caractéristiques/lieux/usagers et véhicules fusionnés avec succès.")

        return merged_df

    def process_all_datasets(self) -> None:
        try:
            self.preprocess_datasets()
            self.extract_reference_structure()
            for prefix, df_list in self.dataframes.items():
                reference_structure = self.reference_structures.get(prefix)
                if reference_structure:
                    for i, df in enumerate(df_list):
                        dataset_name = f"{prefix}-{self.config['years_to_process'][i]}"
                        df = self.apply_reference_structure(df, reference_structure, dataset_name)
                        self.logger.info(f"Structure de référence appliquée à {dataset_name}.")
            self.concatenate_and_save()
            merged_df = self.merge_all_dataframes()
            output_file = f"merged_data_{self.config['years_to_process'][0]}_{self.config['years_to_process'][-1]}.csv"
            output_path = os.path.join(self.config["path_to_processed_csv"], output_file)
            merged_df.to_csv(output_path, index=False)
            self.logger.info(f"Dataframe final fusionné sauvegardé dans {output_path}")
        finally:
            log_handler_manager.stop_listener()

# %%
# ON instancie DataProcessor
data_processor = DataProcessor(config, log_handler_manager)

data_processor.process_all_datasets()
