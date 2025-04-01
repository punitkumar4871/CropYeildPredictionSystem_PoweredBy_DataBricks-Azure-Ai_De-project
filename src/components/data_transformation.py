import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function is responsible for data transformation
        """
        try:
            numerical_columns = ['Crop_Year', 'Area', 'annual_rainfall', 'Soil pH']
            categorical_columns = ['State_Name', 'District_Name', 'Season', 'Crop', 'Soil Type']

            #  Numerical Pipeline
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            #  Categorical Pipeline
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                ("scaler", StandardScaler(with_mean=False))
            ])

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            #  Combine Pipelines
            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            #  Load datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data successfully")

            # Get preprocessing object
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "yeild"
            extra_column = "Production"  # Column to drop
            numerical_columns = ['Crop_Year', 'Area', 'annual_rainfall', 'Soil pH']

            # Split features and target
            input_feature_train_df = train_df.drop(columns=[target_column_name, extra_column], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name, extra_column], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object to training and testing data")

            #  Transform data
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            #  Ensure data is dense (avoid sparse matrices)
            if isinstance(input_feature_train_arr, np.ndarray) == False:
                input_feature_train_arr = input_feature_train_arr.toarray()
            if isinstance(input_feature_test_arr, np.ndarray) == False:
                input_feature_test_arr = input_feature_test_arr.toarray()

            #  Ensure target variable has correct shape
            target_feature_train_df = target_feature_train_df.values.reshape(-1, 1)
            target_feature_test_df = target_feature_test_df.values.reshape(-1, 1)

            
            # Concatenate features & target variable
            train_arr = np.hstack((input_feature_train_arr, target_feature_train_df))
            test_arr = np.hstack((input_feature_test_arr, target_feature_test_df))

            logging.info("Preprocessing object saved successfully.")

            #  Save preprocessor object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
