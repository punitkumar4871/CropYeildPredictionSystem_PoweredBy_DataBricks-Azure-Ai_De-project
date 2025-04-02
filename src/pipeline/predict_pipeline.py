import sys
import pandas as pd
import numpy as np
import os
import xgboost as xgb
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            
            print("Loading model and preprocessor...")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            print("Transforming data...")
            data_scaled = preprocessor.transform(features).astype(np.float64)  # Convert float32 to float64

            print("Making prediction...")
            model.set_params(device="cpu")  # Ensure model runs on CPU
            preds = model.predict(data_scaled)            

            # ✅ Debugging - Print prediction results
            print("Raw Prediction:", preds, "Type:", type(preds))

            # Ensure preds is iterable
            if isinstance(preds, (np.float32, float)):  
                preds = [float(preds)]  # Convert to a list

            elif isinstance(preds, np.ndarray):  
                preds = preds.tolist()  # Convert to a Python list  

            print("Processed Prediction:", preds, "Type:", type(preds))

            return preds
            
        except Exception as e:
            raise CustomException(e, sys)
class CustomData:
    def __init__(self,
                 State_Name: str,
                 District_Name: str,
                 Season: str,
                 Crop: str,
                 Crop_Year: int,
                 Area: float,
                 annual_rainfall: float):
        
        self.State_Name = State_Name
        self.District_Name = District_Name
        self.Season = Season
        self.Crop = Crop
        self.Crop_Year = Crop_Year
        self.Area = Area
        self.annual_rainfall = annual_rainfall
        

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "State_Name": [self.State_Name],
                "District_Name": [self.District_Name],
                "Season": [self.Season],
                "Crop": [self.Crop],
                "Crop_Year": [self.Crop_Year],
                "Area": [self.Area],
                "annual_rainfall": [self.annual_rainfall],
                
            }

            return pd.DataFrame(custom_data_input_dict)
            
        except Exception as e:
            raise CustomException(e, sys)
