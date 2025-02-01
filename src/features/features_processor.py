import pandas as pd
import numpy as np
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

        # Modifier les valeurs manquantes en -1
        for column, details in fusion_data.items():
            if column in dataframe.columns:
                # Remplacement des valeurs spécifiées dans 'to_nan' par NaN
                if 'to_nan' in details:
                    to_nan_values = [str(val) for val in details['to_nan']]
                    dataframe[column] = dataframe[column].replace(to_nan_values, -1)
                    self.logger.info(f"Valeurs {to_nan_values} remplacées par NaN dans la colonne '{column}'.")

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

        # Colonnes implies
        for col in ['imply_cycle_edp', 'imply_2rm', 'imply_vl', 'imply_pl']:
            dataframe[col] = 0

        # Pour chaque catégorie, créer une colonne binaire et compter
        dataframe['is_cycle_edp'] = (dataframe['catv'] == 'cycle_edp').astype(int)
        dataframe['is_2rm']      = (dataframe['catv'] == '2rm').astype(int)
        dataframe['is_vl']       = (dataframe['catv'] == 'vl').astype(int)
        dataframe['is_pl']       = (dataframe['catv'] == 'pl').astype(int)

        # Obtenir le nombre total par accident
        dataframe['count_cycle_edp'] = dataframe.groupby('Num_Acc')['is_cycle_edp'].transform('sum')
        dataframe['count_2rm']       = dataframe.groupby('Num_Acc')['is_2rm'].transform('sum')
        dataframe['count_vl']        = dataframe.groupby('Num_Acc')['is_vl'].transform('sum')
        dataframe['count_pl']        = dataframe.groupby('Num_Acc')['is_pl'].transform('sum')

        # Déterminer l’implication
        dataframe['imply_cycle_edp'] = (
            # Si je suis moi-même cycle_edp, il faut que count_cycle_edp > 1
            ((dataframe['catv'] == 'cycle_edp') & (dataframe['count_cycle_edp'] > 1)) |
            # Sinon, si je ne suis pas cycle_edp, il faut que count_cycle_edp > 0
            ((dataframe['catv'] != 'cycle_edp') & (dataframe['count_cycle_edp'] > 0))
        ).astype(int)

        # Faire la même chose pour 2rm, vl, pl
        dataframe['imply_2rm'] = (
            ((dataframe['catv'] == '2rm') & (dataframe['count_2rm'] > 1)) |
            ((dataframe['catv'] != '2rm') & (dataframe['count_2rm'] > 0))
        ).astype(int)

        dataframe['imply_vl'] = (
            ((dataframe['catv'] == 'vl') & (dataframe['count_vl'] > 1)) |
            ((dataframe['catv'] != 'vl') & (dataframe['count_vl'] > 0))
        ).astype(int)

        dataframe['imply_pl'] = (
            ((dataframe['catv'] == 'pl') & (dataframe['count_pl'] > 1)) |
            ((dataframe['catv'] != 'pl') & (dataframe['count_pl'] > 0))
        ).astype(int)

        # Nettoyage : on peut supprimer les colonnes intermédiaires si on veut
        dataframe.drop([
            'is_cycle_edp', 'is_2rm', 'is_vl', 'is_pl',
            'count_cycle_edp', 'count_2rm', 'count_vl', 'count_pl'
        ], axis=1, inplace=True)

        
        self.logger.info("Colonnes 'imply_cycle_edp', 'imply_2rm', 'imply_vl', 'imply_pl' ajoutées.")


        # Supprimer les colonnes 'num_veh', 'id_usager', 'id_vehicule', 'Num_Acc'.
        columns_to_drop = ['num_veh', "num_veh_x", "num_veh_y", 'id_usager', 'id_vehicule', 'Num_Acc']
        columns_existing = [col for col in columns_to_drop if col in dataframe.columns]
        if columns_existing:
            dataframe.drop(columns=columns_existing, inplace=True)
            self.logger.info(f"Colonnes {columns_existing} supprimées.")

        # Traitement des colonnes spécifiques
        if 'an_nais' in dataframe.columns:
            dataframe['an_nais'] = dataframe['an_nais'].clip(lower=self.config['oldest_year'], upper=self.config['years_to_process'][-1])
            dataframe["an_nais"] = dataframe["an_nais"].replace(np.nan, dataframe["an_nais"].median())
            self.logger.info(f"an_nais: np.nan to {dataframe['an_nais'].median()}")
            dataframe['age'] = dataframe['an'] - dataframe["an_nais"].astype("int64")
            dataframe['age'] = dataframe['age'].astype("int64")
            #dataframe.drop(columns=['an_nais'], inplace= True)
            self.logger.info(f"Variable age créee.")
            
        if all(col in dataframe.columns for col in ['an', 'mois', 'jour']):
            # Convertir directement les colonnes 'an', 'mois', 'jour' en une date
            dataframe['an'] = dataframe['an'].clip(lower=self.config['years_to_process'][0], upper= self.config['years_to_process'][-1])
            dataframe["an"] = dataframe["an"].replace(np.nan, dataframe["an"].median())
            self.logger.info(f"an: np.nan to {dataframe['an_nais'].median()}")
            dataframe['mois'] = dataframe['mois'].clip(lower= 1, upper= 12)
            dataframe["mois"] = dataframe["mois"].replace(np.nan, dataframe["mois"].median())
            self.logger.info(f"mois: np.nan to {dataframe['mois'].median()}")
            dataframe['jour'] = dataframe['jour'].clip(lower= 1, upper= 31)
            dataframe["jour"] = dataframe["jour"].replace(np.nan, dataframe["mois"].median())
            self.logger.info(f"jour: np.nan to {dataframe['jour'].median()}")
            dataframe['date'] = pd.to_datetime(dataframe[['an', 'mois', 'jour']].rename(columns={'an': 'year', 'mois': 'month', 'jour': 'day'}), errors='coerce')
            self.logger.warning(dataframe[dataframe['date'].isna()][['an', 'mois', 'jour']])
            
            # Ajouter le jour de la semaine
            dataframe['jour_semaine'] = dataframe['date'].dt.day_name().astype("category")
            dataframe.drop(columns=['date'], inplace= True)
        else:
            raise ValueError("Les colonnes nécessaires ('an', 'mois', 'jour') sont manquantes.")


        if 'vma' in dataframe.columns:
            dataframe['vma'] = dataframe['vma'].clip(upper=self.config['vma_cap'])
            self.logger.info(f"Valeurs de la colonne 'vma' traitées : valeurs limitées à {self.config['vma_cap']}.")

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
            dataframe['tpc'] = dataframe['lartpc'].astype('int64')
            dataframe.drop(columns=['lartpc'], inplace=True)

        if 'dep' in dataframe.columns:
            dep_to_reg = DepToReg(self.config['path_to_reg'], self.logger)
            dataframe['reg'] = dep_to_reg.transform(dataframe, 'dep')
            dataframe['reg'] = dataframe['reg'].astype('category')
            dataframe.drop(columns=['dep'], inplace=True)

        if 'grav' in dataframe.columns:
            #dataframe['grav'] = dataframe['grav'].replace(4, 0)
            dataframe['grav'] = dataframe['grav'].replace(4, 1)  # on regroupe Blessé léger + indemne
            dataframe['grav'] = dataframe['grav'].replace(3, 2)  # idem Blessé grave + tué
            dataframe['grav'] = dataframe['grav'].astype('int64')
            dataframe = dataframe[dataframe['grav'] != -1] # On retire les fuyards
            dataframe['grav'] = dataframe['grav'] - 1 # Ramener les modalités à 0 et 1
            self.logger.info("Prétraitement de la variable cible 'grav'.")

        print(dataframe.dtypes)

        return dataframe
