from src.Fashion_Recommednation_System.config.configuration import ConfigurationManager
from src.Fashion_Recommednation_System.components.model_preperation import PrepareBaseModel
from src.Fashion_Recommednation_System  import logger

STAGE_NAME='prepare Model'

class PrepareModelTrainingPipeline:
    def __init__(self):
        pass
    def main(self):
    
        config=ConfigurationManager()
        prepare_model_config=config.get_prepare_base_model_config()
        prepare_model=PrepareBaseModel(config=prepare_model_config)
        prepare_model.base_model()
        prepare_model.prepare_full_model()
   



if __name__ =='__main__':
    try:
        logger.info(f'***************')
        logger.info(f'>>>>> stage {STAGE_NAME} started <<<<<<')
        obj=PrepareModelTrainingPipeline()
        obj.main()
        logger.info(f'>>>>> stage {STAGE_NAME} completed <<<<<<<')
    except Exception as e:
        logger.exception(e)