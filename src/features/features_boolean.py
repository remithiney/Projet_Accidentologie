import pandas as pd
import logging
from typing import Dict, Any

class FeaturesBoolean:
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger

    def create_secu_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        # Création des colonnes booléennes basées sur les critères donnés
        dataframe['has_ceinture'] = dataframe[['secu1', 'secu2', 'secu3']].isin([1]).any(axis=1).astype(int)
        dataframe['has_gants'] = dataframe[['secu1', 'secu2', 'secu3']].isin([6, 7]).any(axis=1).astype(int)
        dataframe['has_casque'] = dataframe[['secu1', 'secu2', 'secu3']].isin([2]).any(axis=1).astype(int)
        dataframe['has_airbag'] = dataframe[['secu1', 'secu2', 'secu3']].isin([5, 7]).any(axis=1).astype(int)
        dataframe['has_gilet'] = dataframe[['secu1', 'secu2', 'secu3']].isin([4]).any(axis=1).astype(int)
        dataframe['has_de'] = dataframe[['secu1', 'secu2', 'secu3']].isin([3]).any(axis=1).astype(int)

        # Logger les informations sur les colonnes créées
        self.logger.info("Colonnes ~booléennes 'has_ceinture', 'has_gants', 'has_casque', 'has_airbag', 'has_gilet', 'has_de' créées avec succès.")
        
        # Suppression des colonnes 'secu1', 'secu2', 'secu3'
        dataframe.drop(columns=['secu1', 'secu2', 'secu3'], inplace=True)
        self.logger.info("Colonnes 'secu1', 'secu2', 'secu3' supprimées après création des colonnes booléennes.")
        
        return dataframe

    def create_choc_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        # Création des colonnes booléennes basées sur la valeur de la variable 'choc'
        dataframe['choc_avant'] = dataframe['choc'].isin([1, 2, 3, 9]).astype(int)
        dataframe['choc_arriere'] = dataframe['choc'].isin([4, 5, 6, 9]).astype(int)
        dataframe['choc_gauche'] = dataframe['choc'].isin([3, 6, 8, 9]).astype(int)
        dataframe['choc_droit'] = dataframe['choc'].isin([2, 5, 7, 9]).astype(int)

        # Logger les informations sur les colonnes créées
        self.logger.info("Colonnes ~booléennes 'choc_avant', 'choc_arriere', 'choc_gauche', 'choc_droit' créées avec succès.")
        
        # Suppression de la colonne 'choc'
        dataframe.drop(columns=['choc'], inplace=True)
        self.logger.info("Colonne 'choc' supprimée après création des colonnes booléennes.")
        
        return dataframe

    def create_boolean_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe = self.create_secu_features(dataframe)
        dataframe = self.create_choc_features(dataframe)
        self.logger.info("Toutes les colonnes booléennes de sécurité et de choc ont été créées avec succès.")
        
        return dataframe