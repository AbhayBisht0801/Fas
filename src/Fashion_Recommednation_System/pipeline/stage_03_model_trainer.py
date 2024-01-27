from src.Fashion_Recommednation_System.config.configuration import ConfigurationManager
from src.Fashion_Recommednation_System.components.model_trainer import Training
from src.Fashion_Recommednation_System import logger

STAGE_NAME="Traning"
class ModeltrainingPipeline:
    def __init__(self):
        pass
    def main(self):

        
        config=ConfigurationManager()
        training_config=config.get_training_config()
        training=Training(config=training_config)
        training.complete_model()
        training.extract_path()
        training.extract_feature()

if __name__=="__main__":
    try:
        logger.info(f'************')
        logger.info(f'>>>>>>>>stage{STAGE_NAME} started<<<<<<<<')
        obj=ModeltrainingPipeline()
        obj.main()
        logger.info(f'>>>>stage {STAGE_NAME} compelted <<<<<<<<<<')
    except Exception as e:
        raise e
        
