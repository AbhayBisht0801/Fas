import os
import tensorflow as tf
import time
from urllib.parse import urlparse
from zipfile import ZipFile
import mlflow
import mlflow.keras
from src.Fashion_Recommednation_System.config.configuration import EvaluationConfig
from pathlib import Path

class Evaluation:
    def __init__(self,config:EvaluationConfig):
        self.config=config
        def load_model(path: Path) -> tf.keras.Model:
            return tf.keras.models.load_model(path)
        self.model=load_model(self.config.path_of_model)
    def log_into_mlflow(self):  
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
             mlflow.log_artifact("artifacts/training/extracted_feature.pkl", artifact_path="extracted_features")
             mlflow.log_artifact("artifacts/data_ingestion/cuisine_updated.csv", artifact_path="cusine_data")
            
            # Model registry does not work with file store
             if tracking_url_type_store != "file":
                 mlflow.keras.log_model(self.model, "model", registered_model_name="Feature_extractor")
             else:
                 mlflow.keras.log_model(self.model, "model")
    