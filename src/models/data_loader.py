import pandas as pd
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class DataLoader:
    def __init__(self, logger):
        self.logger = logger

    def load_data(self, file_path, target_column, sample_ratio=None, test_size=0.2, random_state=42):
        try:
            self.logger.info(f"Loading data from {file_path}")
            data = pd.read_csv(file_path)
            data = data[data[target_column] > -1]
            X = data.drop(columns=[target_column])
            y = data[target_column]

            if sample_ratio is not None:
                _, X, _, y = train_test_split(X, y, test_size=sample_ratio, random_state=random_state, stratify=y)  

            self.logger.info("Data loaded successfully")
            return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        except Exception as e:
            self.logger.error("Error while loading data.")
            self.logger.error(f"Exception: {e}")
            raise
