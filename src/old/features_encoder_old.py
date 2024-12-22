import pandas as pd
import logging
from typing import Dict, Any, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
import numpy as np


class FeaturesEncoder:
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.scalers = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler()
        }
        self.one_hot_encoder = OneHotEncoder(sparse_output=False, drop='first')
        self.fitted = False  # Pour suivre si les objets ont été ajustés

    def fit(self, dataframe: pd.DataFrame) -> None:
        # Ajuster le MinMaxScaler
        if "features_normalize" in self.config:
            for column in self.config["features_normalize"]:
                if column in dataframe.columns:
                    self.scalers["minmax"].fit(dataframe[[column]])
                    self.logger.info(f"MinMaxScaler ajusté pour la colonne '{column}'.")

        # Ajuster le StandardScaler
        if "features_standardize" in self.config:
            for column in self.config["features_standardize"]:
                if column in dataframe.columns:
                    self.scalers["standard"].fit(dataframe[[column]])
                    self.logger.info(f"StandardScaler ajusté pour la colonne '{column}'.")

        # Ajuster le OneHotEncoder
        if "features_onehot" in self.config:
            self.one_hot_encoder.fit(dataframe[self.config["features_onehot"]])
            self.logger.info(f"OneHotEncoder ajusté pour les colonnes : {self.config['features_onehot']}.")

        self.fitted = True
        self.logger.info("FeaturesEncoder ajusté avec succès.")

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise ValueError("Les encodeurs et scalers doivent être ajustés avant de transformer. Appelez `fit()` d'abord.")

        # Normaliser les colonnes spécifiées
        if "features_normalize" in self.config:
            for column in self.config["features_normalize"]:
                if column in dataframe.columns:
                    dataframe[[column]] = self.scalers["minmax"].transform(dataframe[[column]])
                    self.logger.info(f"Colonne '{column}' normalisée entre 0 et 1.")

        # Standardiser les colonnes spécifiées
        if "features_standardize" in self.config:
            for column in self.config["features_standardize"]:
                if column in dataframe.columns:
                    dataframe[[column]] = self.scalers["standard"].transform(dataframe[[column]])
                    self.logger.info(f"Colonne '{column}' standardisée avec une moyenne de 0 et un écart-type de 1.")

        # Encodage One-Hot des colonnes spécifiées
        if "features_onehot" in self.config:
            encoded_cols = self.one_hot_encoder.transform(dataframe[self.config["features_onehot"]])
            encoded_df = pd.DataFrame(encoded_cols, columns=self.one_hot_encoder.get_feature_names_out(self.config["features_onehot"]))
            dataframe = pd.concat([dataframe.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
            dataframe.drop(columns=self.config["features_onehot"], inplace=True)
            self.logger.info(f"Encodage one-hot appliqué aux colonnes : {self.config['features_onehot']}.")

        return dataframe
