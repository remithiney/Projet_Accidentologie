# %%
import pandas as pd
import numpy as np
import os
import json
import logging
import time
from typing import Dict, Any
from features.data_loading import DataLoader
from loghandler import LogHandlerManager
from features.preprocessor import Preprocessor
from features.reference_structure import ReferenceStructureExtractor
from features.data_concatenator import DataConcatenator
from features.data_merger import DataMerger
from features.features_processor import FeaturesProcessor
from features.features_boolean import FeaturesBoolean

# %%
# Charger le fichier de configuration
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Journalisation
log_handler_manager = LogHandlerManager(config["logs"])
log_handler_manager.start_listener()
logger = logging.getLogger(__name__)
logger.addHandler(log_handler_manager.get_queue_handler())
logger.setLevel(logging.DEBUG)

# %%
# Flux de travail principal pour le traitement des données
class DataProcessor:
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.data_loader = DataLoader(config, logger)
        self.dataframes = self.data_loader.load_datasets_parallel(config["prefixes"], config["years_to_process"])

    def process_all_datasets(self) -> None:
        try:
            start_time= time.time()
            
            # Prétraitement
            preprocessor = Preprocessor(self.config, self.logger)
            self.dataframes = preprocessor.preprocess(self.dataframes)

            # Extraire et appliquer la structure de référence
            reference_extractor = ReferenceStructureExtractor(self.logger)
            reference_extractor.extract_structure(self.dataframes)
            for prefix, df_list in self.dataframes.items():
                for i, df in enumerate(df_list):
                    dataset_name = f"{prefix}-{self.config['years_to_process'][i]}"
                    df = reference_extractor.apply_structure(df, prefix)
                    self.logger.info(f"Structure de référence appliquée à {dataset_name}.")

            # Concaténer et sauvegarder
            data_concatenator = DataConcatenator(self.config, self.logger)
            data_concatenator.concatenate_and_save(self.dataframes)

            # Fusionner tous les DataFrames
            data_merger = DataMerger(self.config, self.logger)
            
            df_merged = data_merger.merge_all_dataframes()

            # A essayer dans le pipeline directement pour plus de modularité sur les modèles
            # Appliquer le traitement des caractéristiques
            features_processor = FeaturesProcessor(self.config, self.logger)
            df_merged = features_processor.process_features(df_merged)
            self.logger.info("Traitement des caractéristiques appliqué au DataFrame fusionné.")

            features_boolean = FeaturesBoolean(self.config, self.logger)
            df_merged = features_boolean.create_boolean_features(df_merged)

            df_merged= preprocessor.solve_duplicates(df_merged)
            
            # Sauvegarder le DataFrame sans encodage
            output_file = f"merged_data_{self.config['years_to_process'][0]}_{self.config['years_to_process'][-1]}.csv"
            output_path = os.path.join(self.config["path_to_processed_csv"], output_file)
            df_merged.to_csv(output_path, index=False)
            self.logger.info(f"DataFrame fusionné sauvegardé dans {output_path}.")

            end_time = time.time()
            execution_time = end_time - start_time
            self.logger.info(f"Traitement total: {execution_time:.2f} secondes")

            

        finally:
            log_handler_manager.stop_listener()

# %%
# Instancier et exécuter le DataProcessor
data_processor = DataProcessor(config, logger)
data_processor.process_all_datasets()
