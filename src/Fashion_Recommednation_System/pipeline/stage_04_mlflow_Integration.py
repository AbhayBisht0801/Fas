from src.Fashion_Recommednation_System.config.configuration import ConfigurationManager
from src.Fashion_Recommednation_System.mlfow_integration import Evaluation
from src.Fashion_Recommednation_System import logger
import os



STAGE_NAME ='Mlflow integration'
class Integration_pipline:
    def __init__(self):
        pass
    def main(self):
        try:
            config=ConfigurationManager()
            eval_config=config.get_components()
            evaluation=Evaluation(eval_config)
            evaluation.log_into_mlflow()
        except Exception as e:
            raise e

if __name__=='__main__':
    try:
        logger.info(f'***********')
        logger.info(f'>>>>>>>stage {STAGE_NAME} started<<<<<<<<')
        obj=Integration_pipline()
        obj.main()
        logger.info(f'>>>>>>>stage {STAGE_NAME} completed<<<<<<<')
    except Exception as e:
        logger.exception(e)
        raise e
