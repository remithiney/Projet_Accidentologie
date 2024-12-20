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
