import os
import sys
from dataclasses import dataclass

from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # 🚀 Using only XGBoost with the exact parameters from your previous model
            models = {
                "XGBRegressor": XGBRegressor(
                    random_state=42, 
                    tree_method='gpu_hist', 
                    n_estimators=300,        # Matches your standalone model
                    learning_rate=0.05,      # Matches your standalone model
                    max_depth=8,             # Matches your standalone model
                    subsample=0.8,           # Matches your standalone model
                    colsample_bytree=0.8,    # Matches your standalone model
                    reg_alpha=0.3,           # Matches your standalone model
                    reg_lambda=0.8           # Matches your standalone model
                )
            }

            params = {
                "XGBRegressor": {}  # No need for hyperparameter tuning, already set
            }

            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                models=models, param=params
            )

            ## Get best model score
            best_model_score = max(sorted(model_report.values()))

            ## Get best model name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best model found: {best_model_name} with R² score: {best_model_score}")

            # 🚀 Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # 🚀 Predict and calculate R² score
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
