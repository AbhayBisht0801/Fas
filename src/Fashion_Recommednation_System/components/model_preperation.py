import tensorflow as tf
from src.Fashion_Recommednation_System import logger
from pathlib import Path
from src.Fashion_Recommednation_System.entity.config_entity import PrepareBaseModelConfig
class PrepareBaseModel:
    def __init__ (self,config:PrepareBaseModelConfig):
        self.config=config
    def base_model(self):
        self.model=tf.keras.applications.ResNet50(
            input_shape=self.config.params_image_size,
            include_top=self.config.params_include_top,
            weights=self.config.params_weights
        )
        self.save_model(path=self.config.base_model_path,model=self.model)
        
    def prepare_full_model(self):
        base_model=self.model
        base_model.trainable=False
        feature_extractor=tf.keras.models.Sequential()
        feature_extractor.add(base_model)
        feature_extractor.add(tf.keras.layers.GlobalAveragePooling2D())
        feature_extractor.summary()
        self.save_model(path=self.config.updated_base_model_path, model=feature_extractor)

        return feature_extractor
    
    
    @staticmethod
    def save_model(path:Path,model:tf.keras.Model):
        model.save(path)
    
    


        
