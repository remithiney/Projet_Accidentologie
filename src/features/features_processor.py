import pandas as pd
import logging
import json
from features.dep_to_reg import DepToReg
from typing import Dict, Any

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
                        old_values_str = [str(val) for val in old_values]
                        dataframe[column] = dataframe[column].replace(old_values_str, new_modality)
                        self.logger.info(f"Modalités de la colonne '{column}' remplacées par '{new_modality}' pour les valeurs {old_values_str}.")
                else:
                    self.logger.warning(f"Clé 'modalities' manquante pour la colonne '{column}'. Aucun remplacement effectué.")

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
            dataframe['id_vehicule'] = dataframe['id_vehicule'].astype(str).str.replace(r'\s+', '', regex=True)
            dataframe['id_vehicule'] = pd.to_numeric(dataframe['id_vehicule'], errors='coerce')
            dataframe['id_vehicule'] = dataframe['id_vehicule'].astype('int64', errors='ignore')
            self.logger.info("Colonne 'id_vehicule' convertie en type 'int64'.")
        
        if 'Num_Acc' in dataframe.columns and 'id_vehicule' in dataframe.columns:
            dataframe['nb_v'] = dataframe.groupby('Num_Acc')['id_vehicule'].transform('nunique').astype('int64')
            self.logger.info("Colonne 'nb_v' ajoutée, indiquant le nombre de véhicules impliqués dans chaque accident.")

        if 'Num_Acc' in dataframe.columns and 'id_usager' in dataframe.columns:
            dataframe['nb_u'] = dataframe.groupby('Num_Acc')['id_usager'].transform('nunique').astype('int64')
            self.logger.info("Colonne 'nb_u' ajoutée, indiquant le nombre d'usagers impliqués dans chaque accident.")

        # Supprimer les colonnes 'num_veh', 'id_usager', 'id_vehicule', 'Num_Acc'.
        columns_to_drop = ['num_veh', 'id_usager', 'id_vehicule', 'Num_Acc']
        columns_existing = [col for col in columns_to_drop if col in dataframe.columns]
        if columns_existing:
            dataframe.drop(columns=columns_existing, inplace=True)
            self.logger.info(f"Colonnes {columns_existing} supprimées.")

        # Traitement des colonnes spécifiques
        if 'vma' in dataframe.columns:
            dataframe['vma'] = dataframe['vma'].clip(upper=130)
            self.logger.info("Valeurs de la colonne 'vma' traitées : valeurs supérieures à 130 limitées à 130.")

        if "hrmn" in dataframe.columns:
            dataframe['hrmn'] = dataframe['hrmn'].astype(str)
            dataframe['hr'] = dataframe['hrmn'].str.split(':').str[0]
            dataframe['hr'] = dataframe['hr'].astype('int64')
            dataframe.drop(columns=['hrmn'], inplace=True)
            self.logger.info(f"Colonne 'hrmn' transformée en 'hr' pour caracteristiques.")

        if "lartpc" in dataframe.columns:
            dataframe['lartpc'] = pd.to_numeric(dataframe['lartpc'], errors='coerce')
            dataframe['lartpc'] = dataframe['lartpc'].abs()
            dataframe['lartpc'] = (dataframe['lartpc'] != 0).astype(int)
            dataframe['lartpc'] = dataframe['lartpc'].map({0: 'False', 1: 'True'})
            dataframe['tpc'] = dataframe['lartpc'].astype('category')
            dataframe.drop(columns=['lartpc'], inplace=True)

        if 'dep' in dataframe.columns:
            dep_to_reg = DepToReg(self.config['path_to_reg'], self.logger)
            dataframe['reg'] = dep_to_reg.transform(dataframe, 'dep')
            dataframe['reg'] = dataframe['reg'].astype('category')
            dataframe.drop(columns=['dep'], inplace=True)

        if 'grav' in dataframe.columns:
            dataframe['grav'] = dataframe['grav'].replace(4, 1)  # Blessé léger = indemne
            dataframe['grav'] = dataframe['grav'].replace(3, 2)  # Blessé grave = tué
            dataframe['grav'] = dataframe['grav'].astype('int64')
            self.logger.info("Prétraitement de la variable cible 'grav'.")

        print(dataframe.dtypes)

        return dataframe
