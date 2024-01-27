from src.Fashion_Recommednation_System import logger
from src.Fashion_Recommednation_System.pipeline.stage__01_data_ingestion import DataIngestionTrainingPipeline

STAGE_NAME='data ingestion Stage'

try:
    logger.info(f'>>>>>stage{STAGE_NAME} started<<<<<')
    obj=DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f'>>>>>>stage {STAGE_NAME} completedd<<<<<<')
except Exception as e:
    logger.exception(e)
    raise e