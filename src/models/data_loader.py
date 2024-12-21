import pandas as pd
from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(self, logger):
        self.logger = logger

    def load_data(self, file_path, target_column, sample_ratio=0.33, test_size=0.2, random_state=42):
        try:
            self.logger.info(f"Loading data from {file_path}")
            data = pd.read_csv(file_path)
            X = data.drop(columns=[target_column])
            y = data[target_column]-1 # Commencer à 0

            # Pour accélérer le temps de calcul on va faire un premier échantillonage
            _, X_sampled, _, y_sampled = train_test_split(X, y, test_size=sample_ratio, random_state=random_state, stratify=y)  

            self.logger.info("Data loaded successfully")
            return train_test_split(X_sampled, y_sampled, test_size=test_size, random_state=random_state, stratify=y_sampled)
        except Exception as e:
            self.logger.error("Error while loading data.")
            self.logger.error(f"Exception: {e}")
            raise
