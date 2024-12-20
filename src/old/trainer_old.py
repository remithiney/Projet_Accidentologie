from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import traceback
import time
import joblib

class Trainer:
    def __init__(self, logger, scoring='recall', save_model_path=None):
        self.logger = logger
        self.scoring = scoring
        self.save_model_path = save_model_path

    def train_model(self, pipeline, param_grid, X_train, y_train, n_iter=5, cv=3):
        try:
            self.logger.info("Starting model training...")
            start_time = time.time()

            search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_grid,
                cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
                scoring=self.scoring,
                n_iter=n_iter,
                n_jobs=-1,
                random_state=42,
                verbose=3
            )

            search.fit(X_train, y_train)
            elapsed_time = time.time() - start_time

            best_model = search.best_estimator_
            best_params = search.best_params_
            self.logger.info(f"Training completed in {elapsed_time:.2f} seconds")
            self.logger.info(f"Best parameters: {best_params}")

            # Optionally save the model
            if self.save_model_path:
                self.logger.info(f"Saving best model to {self.save_model_path}")
                joblib.dump(best_model, self.save_model_path)

            return best_model, best_params

        except Exception as e:
            self.logger.error("Error during model training.")
            self.logger.error(traceback.format_exc())
            raise
