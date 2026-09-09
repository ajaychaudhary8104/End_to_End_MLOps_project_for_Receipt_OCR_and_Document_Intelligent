from src.receipt_intelligence.config.configuration import ConfigurationManager
from src.receipt_intelligence.components.image_preprocessing import ImagePreprocessor
from src.receipt_intelligence import logger


STAGE_NAME = "IMAGE PREPROCESSING STAGE"

class ImagePreprocessingTrainingPipeline:

    def __init__(self):
        pass

    def main(self):

        config = ConfigurationManager()

        image_preprocessing_config = (
            config.get_image_preprocessing_config()
        )

        image_preprocesser = ImagePreprocessor(
            config=image_preprocessing_config
        )

        summary = (
            image_preprocesser.run()
        )

if __name__ == "__main__":
    try:

        logger.info(
            f">>>>>> stage {STAGE_NAME} started <<<<<<"
        )

        obj = (
            ImagePreprocessingTrainingPipeline()
        )

        obj.main()

        logger.info(
            f">>>>>> stage {STAGE_NAME} completed <<<<<<"
        )

    except Exception as e:

        logger.exception(e)

        raise e