import pandas as pd
import logging
from typing import Dict, List, Any

class ReferenceStructureExtractor:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.reference_structures = {}

    def extract_structure(self, dataframes: Dict[str, List[pd.DataFrame]]):
        for prefix, df_list in dataframes.items():
            if df_list:
                reference_df = df_list[-1]
                reference_structure = {
                    "columns": list(reference_df.columns),
                    "dtypes": reference_df.dtypes.to_dict()
                }
                self.reference_structures[prefix] = reference_structure
                self.logger.info(f"Structure de référence extraite pour {prefix}")
    
    def apply_structure(self, df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        reference_structure = self.reference_structures.get(prefix)
        if not reference_structure:
            return df

        reference_columns = reference_structure["columns"]
        reference_dtypes = reference_structure["dtypes"]

        df = df[[col for col in df.columns if col in reference_columns]]
        
        for col in reference_columns:
            if col not in df.columns:
                df[col] = pd.NA
        df = df[reference_columns]

        for col, dtype in reference_dtypes.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                except Exception as e:
                    self.logger.warning(f"Impossible de convertir la colonne {col} au type {dtype}: {e}")
        
        return df
