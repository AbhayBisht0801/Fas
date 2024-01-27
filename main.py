from src.Fashion_Recommednation_System import logger
from src.Fashion_Recommednation_System.pipeline.stage__01_data_ingestion import DataIngestionTrainingPipeline
from src.Fashion_Recommednation_System.pipeline.stage_02_prepare_model import PrepareModelTrainingPipeline
from src.Fashion_Recommednation_System.pipeline.stage_03_model_trainer import ModeltrainingPipeline
from src.Fashion_Recommednation_System.pipeline.stage_04_mlflow_Integration import Integration_pipline


STAGE_NAME='Mlflow_integration'
try:
    logger.info(f'***********')
    logger.info(f'>>>>>>>stage {STAGE_NAME} started<<<<<<<<')
    obj=Integration_pipline()
    obj.main()
    logger.info(f'>>>>>>>satge {STAGE_NAME} completed<<<<<<<')
except Exception as e:
    logger.exception(e)
