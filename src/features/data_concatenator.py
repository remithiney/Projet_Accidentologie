import pandas as pd
import os
import logging
from typing import Dict, List, Any

class DataConcatenator:
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger

    def concatenate_and_save(self, dataframes: Dict[str, List[pd.DataFrame]]) -> None:
        for prefix, df_list in dataframes.items():
            if df_list:
                concatenated_df = pd.concat(df_list, ignore_index=True)

                # Logique spécifique après la concaténation des dataframes usagers
                if prefix == "usagers" and "id_usager" in concatenated_df.columns:
                    # On veut règler le problème de id_usager : il manque des valeurs
                    concatenated_df['id_usager'] = concatenated_df['id_usager'].astype(str)
                    concatenated_df['id_usager'] = pd.to_numeric(concatenated_df['id_usager'], errors='coerce')
                    missing_ids = concatenated_df['id_usager'].isna()

                    if missing_ids.any():
                        max_id = concatenated_df['id_usager'].max(skipna=True)
                        if pd.isna(max_id):
                            max_id = 0 
                        next_id = int(max_id) + 1
                        concatenated_df.loc[missing_ids, 'id_usager'] = range(next_id, next_id + missing_ids.sum())
                        self.logger.info(f"Valeurs manquantes de 'id_usager' remplies avec des valeurs uniques après concaténation pour {prefix}.")
                    
                    concatenated_df['id_usager'] = concatenated_df['id_usager'].astype('int64')
                    self.logger.info(f"Colonne 'id_usager' convertie en type int64 pour {prefix}.")

                output_file = f"{prefix}_{self.config['years_to_process'][0]}_{self.config['years_to_process'][-1]}_process.csv"
                output_path = os.path.join(self.config["path_to_processed_csv"], output_file)
                concatenated_df.to_csv(output_path, index=False)
                self.logger.info(f"Dataframe concaténé pour {prefix} sauvegardé dans {output_path}")
