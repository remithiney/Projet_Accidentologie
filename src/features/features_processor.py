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
        columns_to_drop = ['num_veh', 'id_usager', 'id_vehicule', 'Num_Acc', 'nbv', 'an']
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

        # On découpe les heures en 
        if "hrmn" in dataframe.columns:
            dataframe['hrmn'] = dataframe['hrmn'].astype(str)
            dataframe['hr'] = dataframe['hrmn'].str.split(':').str[0]
            dataframe['hr'] = dataframe['hr'].astype('int64')
            dataframe.drop(columns=['hrmn'], inplace=True)
            self.logger.info(f"Colonne 'hrmn' transformée en 'hr' pour caracteristiques.")

        # On binarise lartpc pour tpc (présence terre plein central)
        if "lartpc" in dataframe.columns:
            dataframe['lartpc'] = pd.to_numeric(dataframe['lartpc'], errors='coerce')
            dataframe['lartpc'] = dataframe['lartpc'].abs()
            dataframe['lartpc'] = dataframe['lartpc'].fillna(0) #les valeurs manquantes 
            dataframe['lartpc'] = (dataframe['lartpc'] != 0).astype(int)
            dataframe['lartpc'] = dataframe['lartpc'].map({0: 'False', 1: 'True'})
            dataframe['tpc'] = dataframe['lartpc'].astype('category')
            dataframe.drop(columns=['lartpc'], inplace=True)

        # On a globalement 2 options pour la position géographique:
        # On change les départements vers des régions pour diminuer les modalités (plus pratique pour un tier également).
        # Ou on clusterise les couples latitude et longitude.

        # On choisit de changer en région
        if 'dep' in dataframe.columns:
            dep_to_reg = DepToReg(self.config['path_to_reg'], self.logger)
            dataframe['reg'] = dep_to_reg.transform(dataframe, 'dep')
            dataframe['reg'] = dataframe['reg'].astype('category')
            dataframe.drop(columns=['dep'], inplace= True)


        # Gestion de la variable cible 'grav'
        # Difficultés avec les modèles pour le moment, on va binariser. On redécoupera par la suite en 3 ou 4 classes, en entrainant des modèles à reperer chaque classe
        if 'grav' in dataframe.columns:
            mode_grav = dataframe['grav'][dataframe['grav'] != -1].mode()[0]
            dataframe['grav'] = dataframe['grav'].replace(-1, mode_grav)
            dataframe['grav'] = dataframe['grav'].replace(4, 1) # Blessé léger = indemne
            dataframe['grav'] = dataframe['grav'].replace(3, 2) # Blessé grave = tué
            dataframe['grav'] = dataframe['grav'].astype('int64')
            self.logger.info("Prétraitrement de la variable cible 'grav'.")

        print(dataframe.dtypes)

        return dataframe
