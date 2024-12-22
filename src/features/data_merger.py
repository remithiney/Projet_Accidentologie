import pandas as pd
import os
import logging
from typing import Dict, Any

class DataMerger:
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger

    def merge_all_dataframes(self) -> pd.DataFrame:
        # Charger les dataframes concaténés pour chaque préfixe
        path_to_processed = self.config["path_to_processed_csv"]
        df_caracteristiques = pd.read_csv(os.path.join(path_to_processed, f"caracteristiques_{self.config['years_to_process'][0]}_{self.config['years_to_process'][-1]}_process.csv"))
        df_lieux = pd.read_csv(os.path.join(path_to_processed, f"lieux_{self.config['years_to_process'][0]}_{self.config['years_to_process'][-1]}_process.csv"))
        df_usagers = pd.read_csv(os.path.join(path_to_processed, f"usagers_{self.config['years_to_process'][0]}_{self.config['years_to_process'][-1]}_process.csv"))
        df_vehicules = pd.read_csv(os.path.join(path_to_processed, f"vehicules_{self.config['years_to_process'][0]}_{self.config['years_to_process'][-1]}_process.csv"))

        # Fusionner les usagers et les vehicule
        df_merged = pd.merge(df_usagers, df_vehicules, on=['id_vehicule', "Num_Acc"], how='left')
        self.logger.info("Fusion usagers et véhicules.")

        # Fusionner avec les caractéristiques
        df_merged = pd.merge(df_merged, df_caracteristiques, on='Num_Acc', how='left')
        self.logger.info("Fusion avec caractéristiques.")

        # Fusionner avec les lieux
        df_merged = pd.merge(df_merged, df_lieux, on='Num_Acc', how='left')
        self.logger.info("Fusion avec lieux.")

        self.logger.info(f'merged shape: {df_merged.shape}')

        return df_merged
