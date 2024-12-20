import os
import json
import csv
import logging
import concurrent.futures
import time
from typing import List, Dict, Tuple, Union, Any
import pandas as pd

# Classe de chargement
class DataLoader:
    def __init__(self, config: Dict[str, Any], logger: logging.Logger = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

    def get_delimiter(self, file_path: str, bytes: int = 4096) -> Union[str, None]:
        try:
            with open(file_path, 'r') as file:
                data = file.read(bytes)
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(data).delimiter
            return delimiter
        except Exception as e:
            self.logger.error(f"Erreur lors de la détection du délimiteur: {e}")
            return None

    def read_csv_file(self, file_path: str) -> Tuple[Union[pd.DataFrame, None], bool, Union[str, None]]:
        if not os.path.exists(file_path):
            return None, False, f"Fichier non trouvé: {file_path}"
        
        delimiter = self.get_delimiter(file_path)
        if not delimiter:
            return None, False, f"Impossible de détecter le délimiteur pour le fichier: {file_path}"
        
        for encoding in self.config['encodings']:
            try:
                df = pd.read_csv(file_path, low_memory=False, encoding=encoding, delimiter=delimiter)
                return df, True, None
            except (UnicodeDecodeError, pd.errors.ParserError) as e:
                self.logger.warning(f"Erreur avec l'encodage {encoding} pour le fichier {file_path}: {e}")
        
        return None, False, f"Impossible de lire le fichier {file_path} avec les encodages: {self.config['encodings']}."

    def load_dataset(self, prefix: str, year: int) -> Tuple[str, Union[pd.DataFrame, None]]:
        connector = '_' if year <= 2016 else '-'
        file_name = os.path.join(self.config['path_to_csvs'], f'{prefix}{connector}{year}.csv')
        self.logger.info(f'file name = {file_name}')
        df, success, error = self.read_csv_file(file_name)
        if success:
            self.logger.info(f"Chargement réussi du fichier: {file_name}")
            return file_name, df
        else:
            self.logger.error(f"Erreur lors du chargement du fichier {file_name}: {error}")
            return file_name, None

    def load_datasets_parallel(self, prefixes: List[str], years: List[int]) -> Dict[str, List[pd.DataFrame]]:
        datasets = {prefix: [] for prefix in prefixes}
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_dataset = {
                executor.submit(self.load_dataset, prefix, year): (prefix, year)
                for prefix in prefixes for year in years
            }
            
            for future in concurrent.futures.as_completed(future_to_dataset):
                prefix, year = future_to_dataset[future]
                try:
                    file_name, df = future.result()
                    if df is not None:
                        datasets[prefix].append(df)
                except Exception as e:
                    self.logger.error(f"Erreur lors du chargement du dataset pour {prefix} {year}: {e}")
        
        end_time = time.time()
        total_time = end_time - start_time
        self.logger.info(f"Temps total de chargement des fichiers: {total_time:.2f} secondes")
        
        return datasets
