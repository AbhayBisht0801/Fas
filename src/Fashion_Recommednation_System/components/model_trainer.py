import tensorflow as tf
from src.Fashion_Recommednation_System import logger
from pathlib import Path
from src.Fashion_Recommednation_System.entity.config_entity import TrainingConfig
from src.Fashion_Recommednation_System.utils.common import feature_extraction,save_object
import os
class Training:
    def __init__(self,config:TrainingConfig):
        self.config=config
        self.img_name=[]
    def complete_model(self):
        self.model=tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
    def extract_path(self):
        path=self.config.training_data
        for image_path in os.listdir(path):
            self.img_name.append(os.path.join(path,image_path))
    

    def extract_feature(self):
        feature_list=[]
        for image in self.img_name:
            features=feature_extraction(image,self.model)
            feature_list.append(features)
        save_object(file_path=self.config.saved_features,obj=feature_list)
        
        
        

    
