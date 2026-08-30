from src.receipt_intelligence.config.configuration import ConfigurationManager
from src.receipt_intelligence.components.data_validation import DataValidation
from src.receipt_intelligence import logger


STAGE_NAME = "DATA VALIDATION STAGE"

class DataValidationTrainingPipeline:

    def __init__(self):
        pass

    def main(self):

        config = ConfigurationManager()

        validation_config = (
            config.get_data_validation_config()
        )

        validation = DataValidation(
            config=validation_config
        )

        validation_status = (
            validation.run()
        )

if __name__ == "__main__":
    try:

        logger.info(
            f">>>>>> stage {STAGE_NAME} started <<<<<<"
        )

        obj = (
            DataValidationTrainingPipeline()
        )

        obj.main()

        logger.info(
            f">>>>>> stage {STAGE_NAME} completed <<<<<<"
        )

    except Exception as e:

        logger.exception(e)

        raise e