import pandas as pd
import logging
import json
from typing import Dict, Any, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

class FeaturesProcessor:
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger

    def process_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        # Charger le fichier JSON contenant les informations sur les modalités
        with open(self.config['features_fusion_path'], 'r') as f:
            fusion_data = json.load(f)

        # Convertir les colonnes spécifiées en type string pour appliquer les remplacements
        for column in fusion_data.keys():
            if column in dataframe.columns:
                dataframe[column] = dataframe[column].astype(str)
                self.logger.info(f"Colonne '{column}' convertie en type 'string' pour le traitement des modalités.")

        # Appliquer les remplacements des modalités
        for column, details in fusion_data.items():
            if column in dataframe.columns:
                if 'modalities' in details:
                    for new_modality, old_values in details['modalities'].items():
                        old_values_str = [str(val) for val in old_values]  # Convertir les valeurs en chaînes de caractères
                        dataframe[column] = dataframe[column].replace(old_values_str, new_modality)
                        self.logger.info(f"Modalités de la colonne '{column}' remplacées par '{new_modality}' pour les valeurs {old_values_str}.")
                else:
                    self.logger.warning(f"Clé 'modalities' manquante pour la colonne '{column}'. Aucun remplacement effectué.")

        # Remplacer les modalités spécifiées par le mode de la colonne
        for column, details in fusion_data.items():
            if column in dataframe.columns and 'to_mode' in details:
                mode_value = dataframe[column].mode()[0]  # Calculer le mode de la colonne
                to_replace = [str(val) for val in details['to_mode']]  # Convertir les valeurs en chaînes de caractères
                dataframe[column] = dataframe[column].replace(to_replace, mode_value)
                self.logger.info(f"Valeurs {to_replace} de la colonne '{column}' remplacées par le mode '{mode_value}'.")

        # Appliquer le type final des colonnes
        for column, details in fusion_data.items():
            if column in dataframe.columns:
                if 'dtype' in details:
                    dtype = details['dtype']
                    dataframe[column] = dataframe[column].astype(dtype)
                    self.logger.info(f"Colonne '{column}' convertie en type '{dtype}'.")
                else:
                    self.logger.warning(f"Clé 'dtype' manquante pour la colonne '{column}'. Aucun changement de type effectué.")

        # Modifier 'id_vehicule' en int64 et ajouter la colonne 'nb_v'
        if 'id_vehicule' in dataframe.columns:
            dataframe['id_vehicule'] = dataframe['id_vehicule'].astype(str).str.replace(r'\s+', '', regex=True)  # Supprimer les espaces
            dataframe['id_vehicule'] = pd.to_numeric(dataframe['id_vehicule'], errors='coerce')  # Convertir en numérique, NaN pour les erreurs
            dataframe['id_vehicule'] = dataframe['id_vehicule'].fillna(-1).astype('int64')  # Remplacer les NaN par -1 et convertir en int64
            self.logger.info("Colonne 'id_vehicule' convertie en type 'int64'.")
        
        if 'Num_Acc' in dataframe.columns and 'id_vehicule' in dataframe.columns:
            dataframe['nb_v'] = dataframe.groupby('Num_Acc')['id_vehicule'].transform('nunique').astype('int64')
            self.logger.info("Colonne 'nb_v' ajoutée, indiquant le nombre de véhicules impliqués dans chaque accident.")

        # Ajouter la colonne 'nb_u' indiquant le nombre d'usagers impliqués dans chaque accident
        if 'id_usager' in dataframe.columns:
            dataframe['id_usager'] = dataframe['id_usager'].astype(str).str.replace(r'\s+', '', regex=True)  # Supprimer les espaces
            dataframe['id_usager'] = pd.to_numeric(dataframe['id_usager'], errors='coerce')  # Convertir en numérique, NaN pour les erreurs
            dataframe['id_usager'] = dataframe['id_usager'].fillna(-1).astype('int64')  # Remplacer les NaN par -1 et convertir en int64
            self.logger.info("Colonne 'id_usager' convertie en type 'int64'.")
        
        if 'Num_Acc' in dataframe.columns and 'id_usager' in dataframe.columns:
            dataframe['nb_u'] = dataframe.groupby('Num_Acc')['id_usager'].transform('nunique').astype('int64')
            self.logger.info("Colonne 'nb_u' ajoutée, indiquant le nombre d'usagers impliqués dans chaque accident.")

        # Supprimer les colonnes 'num_veh', 'id_usager', 'id_vehicule', 'Num_Acc', 'dep', 'nbv' et 'an'
        columns_to_drop = ['num_veh', 'id_usager', 'id_vehicule', 'Num_Acc', 'dep', 'nbv', 'an']
        columns_existing = [col for col in columns_to_drop if col in dataframe.columns]
        if columns_existing:
            dataframe.drop(columns=columns_existing, inplace=True)
            self.logger.info(f"Colonnes {columns_existing} supprimées.")

        # Gestion des valeurs manquantes et modification du type pour 'an_nais'
        if 'an_nais' in dataframe.columns:
            mode_an_nais = dataframe['an_nais'].mode()[0]  # Calculer le mode de la colonne
            dataframe['an_nais'] = dataframe['an_nais'].fillna(mode_an_nais).astype('int64')  # Remplacer les NaN par le mode et convertir en int64
            self.logger.info("Colonne 'an_nais' convertie en type 'int64' et les valeurs manquantes remplacées par le mode.")

        # Gestion des valeurs de 'vma'
        if 'vma' in dataframe.columns:
            mode_vma = dataframe['vma'][dataframe['vma'] != -1].mode()[0]
            dataframe['vma'] = dataframe['vma'].replace(-1, mode_vma)
            dataframe['vma'] = dataframe['vma'].clip(upper=130)
            self.logger.info("Valeurs de la colonne 'vma' traitées : -1 remplacés par le mode, valeurs supérieures à 130 limitées à 130.")

        # Gestion de la variable cible 'grav'
        if 'grav' in dataframe.columns:
            mode_grav = dataframe['grav'][dataframe['grav'] != -1].mode()[0]
            dataframe['grav'] = dataframe['grav'].replace(-1, mode_grav)
            dataframe['grav'] = dataframe['grav'].replace(4, 3) # On fusionne les bléssés graves et légers.
            dataframe['grav'] = dataframe['grav'].astype('int64')
            self.logger.info("Prétraitrement de la variable cible 'grav'.")

        return dataframe

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
    
class FeaturesEncoder:
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.scalers = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler()
        }
        self.one_hot_encoder = OneHotEncoder(sparse_output=False, drop='first')

    def normalize_features(self, dataframe: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        for column in columns:
            if column in dataframe.columns:
                dataframe[[column]] = self.scalers["minmax"].fit_transform(dataframe[[column]])
                self.logger.info(f"Colonne '{column}' normalisée entre 0 et 1.")
            else:
                self.logger.warning(f"Colonne '{column}' non trouvée dans le DataFrame. Normalisation ignorée.")
        return dataframe

    def standardize_features(self, dataframe: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        for column in columns:
            if column in dataframe.columns:
                dataframe[[column]] = self.scalers["standard"].fit_transform(dataframe[[column]])
                self.logger.info(f"Colonne '{column}' standardisée avec une moyenne de 0 et un écart-type de 1.")
            else:
                self.logger.warning(f"Colonne '{column}' non trouvée dans le DataFrame. Standardisation ignorée.")
        return dataframe

    def one_hot_encode_features(self, dataframe: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        for column in columns:
            if column in dataframe.columns:
                encoded_cols = self.one_hot_encoder.fit_transform(dataframe[[column]])
                encoded_df = pd.DataFrame(encoded_cols, columns=self.one_hot_encoder.get_feature_names_out([column]))
                dataframe = pd.concat([dataframe.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
                dataframe.drop(columns=[column], inplace=True)
                self.logger.info(f"Encodage one-hot appliqué à la colonne '{column}'.")
            else:
                self.logger.warning(f"Colonne '{column}' non trouvée dans le DataFrame. Encodage one-hot ignoré.")
        return dataframe

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if "features_normalize" in self.config:
            dataframe = self.normalize_features(dataframe, self.config["features_normalize"])
        if "features_standardize" in self.config:
            dataframe = self.standardize_features(dataframe, self.config["features_standardize"])
        if "features_onehot" in self.config:
            dataframe = self.one_hot_encode_features(dataframe, self.config["features_onehot"])
        return dataframe
