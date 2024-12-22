import pandas as pd
import logging
from typing import Dict, List, Any

class Preprocessor:
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger

    def preprocess(self, dataframes: Dict[str, List[pd.DataFrame]]) -> Dict[str, List[pd.DataFrame]]:
        # Logique de prétraitement pour chaque préfixe
        for prefix, df_list in dataframes.items():
            self.logger.info(f"Prétraitement pour {prefix}")
            self._drop_features(df_list, prefix)
            self._apply_prefix_specific_processing(df_list, prefix)
        return dataframes

    def _drop_features(self, df_list: List[pd.DataFrame], prefix: str):
        features_to_drop = self.config.get('features_to_drop', {}).get(prefix, [])
        for i, df in enumerate(df_list):
            columns_to_drop = [col for col in features_to_drop if col in df.columns]
            df.drop(columns=columns_to_drop, inplace=True)
            self.logger.info(f"Colonnes {columns_to_drop} supprimées pour {prefix}.")

    def _apply_prefix_specific_processing(self, df_list: List[pd.DataFrame], prefix: str):
        # Logique de prétraitement spécifique en fonction du préfixe
        if prefix == "caracteristiques":
            self._preprocess_caracteristiques(df_list)
        elif prefix == "usagers":
            self._preprocess_usagers(df_list)

    def _preprocess_caracteristiques(self, df_list: List[pd.DataFrame]):
        for i, df in enumerate(df_list):
            if "Accident_Id" in df.columns:
                df.rename(columns={"Accident_Id": "Num_Acc"}, inplace=True)
                self.logger.info(f"Colonne 'Accident_Id' renommée en 'Num_Acc' pour caracteristiques.")
            

    def _preprocess_usagers(self, df_list: List[pd.DataFrame]):
        pass

    def solve_duplicates(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        duplicate_count = dataframe.duplicated().sum()
        clean_df = dataframe.drop_duplicates()
        #self.logger.info(f"Duplicated: {dataframe[dataframe.duplicated()]}")
        self.logger.info(f'{duplicate_count} doublons supprimé.')
        return clean_df
