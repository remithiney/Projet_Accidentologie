import os
import pandas as pd
import sweetviz as sv
import json
import logging
from typing import List
from data_loading import DataLoader
from loghandler import LogHandlerManager

# Load configuration file
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

class SweetvizAnalyzer:
    def __init__(self, output_folder: str, log_handler_manager: LogHandlerManager):
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)  # Crée le dossier si inexistant
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(log_handler_manager.get_queue_handler())
        self.logger.setLevel(logging.DEBUG)

        # Pass logger to DataLoader
        self.data_loader = DataLoader(config["path_to_csvs"], self.logger)
        self.dataframes = self.data_loader.load_datasets_parallel(config["prefixes"], config["years"])

    def analyze_datasets(self):
        for prefix, df_list in self.dataframes.items():
            for i, df in enumerate(df_list):
                try:
                    dataset_name = f"{prefix}-{config['years'][i]}"
                    self.logger.info(f"Analyse du dataset : {dataset_name}")
                    report = sv.analyze(df)
                    output_path = os.path.join(self.output_folder, f"{dataset_name}_report.html")
                    report.show_html(output_path)
                    self.logger.info(f"Rapport généré : {output_path}")
                except Exception as e:
                    self.logger.error(f"Erreur lors de l'analyse du dataset {dataset_name}: {str(e)}")

    def compare_oldest_newest(self):
        for prefix, df_list in self.dataframes.items():
            if len(df_list) >= 2:
                try:
                    # Identifier les datasets les plus anciens et les plus récents
                    oldest_df = df_list[0]
                    newest_df = df_list[-1]
                    oldest_year = config["years"][0]
                    newest_year = config["years"][-1]
                    
                    # Comparaison avec Sweetviz
                    self.logger.info(f"Comparaison entre {prefix}-{oldest_year} et {prefix}-{newest_year}")
                    report = sv.compare([oldest_df, f"{prefix}-{oldest_year}"], [newest_df, f"{prefix}-{newest_year}"])
                    output_path = os.path.join(self.output_folder, f"{prefix}_comparison_{oldest_year}_vs_{newest_year}.html")
                    report.show_html(output_path)
                    self.logger.info(f"Rapport de comparaison généré : {output_path}")
                except Exception as e:
                    self.logger.error(f"Erreur lors de la comparaison des datasets pour {prefix}: {str(e)}")

# Utilisation de la classe
log_handler_manager = LogHandlerManager(config["logs"])
log_handler_manager.start_listener()

analyzer = SweetvizAnalyzer(config['sweetviz_output_folder'], log_handler_manager)
analyzer.compare_oldest_newest()

log_handler_manager.stop_listener()
