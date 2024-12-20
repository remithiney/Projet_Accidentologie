import pandas as pd

class DepToReg:
    def __init__(self, csv_path, logger):
        self.csv_path = csv_path
        self.logger = logger
        self.dep_to_reg_map = self._load_mapping()

    def _load_mapping(self):
        try:
            mapping_df = pd.read_csv(self.csv_path)
            if 'departement' not in mapping_df.columns or 'region' not in mapping_df.columns:
                self.logger.error("Le fichier CSV doit contenir les colonnes 'departement' et 'region'.")
                raise ValueError("Colonnes manquantes dans le fichier CSV.")

            mapping_df['departement'] = mapping_df['departement'].apply(
                lambda x: str(x).zfill(2) if str(x).isdigit() else str(x)
            )
            mapping_dict = dict(zip(mapping_df['departement'], mapping_df['region']))
            self.logger.info("Mapping départements-régions chargé avec succès.")
            return mapping_dict
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement du fichier CSV : {e}")
            raise

    def transform(self, dataframe, dep_column):
        if dep_column not in dataframe.columns:
            self.logger.error(f"La colonne '{dep_column}' est absente du DataFrame.")
            raise KeyError(f"Colonne '{dep_column}' manquante dans le DataFrame.")

        try:
            # Première passe sans transformation
            result_series = dataframe[dep_column].map(self.dep_to_reg_map)
            missing_values = dataframe[dep_column][result_series.isna()]

            # Si des valeurs manquent, faire une seconde passe avec zfill
            if not missing_values.empty:
                dataframe[dep_column] = dataframe[dep_column].apply(
                    lambda x: str(x).zfill(2) if pd.notnull(x) and str(x).isdigit() else x
                )
                result_series = dataframe[dep_column].map(self.dep_to_reg_map)
                missing_values = dataframe[dep_column][result_series.isna()]

            if not missing_values.empty:
                unique_missing = missing_values.value_counts()
                truncated_list = unique_missing.index.tolist()[:10]  # Limiter à 10 valeurs uniques
                self.logger.warning(f"{len(missing_values)} départements n'ont pas été trouvés dans le mapping. Exemples : {truncated_list}")
            else:
                self.logger.info("Transformation des départements en régions réussie.")

            return result_series
        except Exception as e:
            self.logger.error(f"Erreur lors de la transformation des départements en régions : {e}")
            raise
