from src.receipt_intelligence.config.configuration import ConfigurationManager
from src.receipt_intelligence.components.inference import ReceiptInferencePipeline
from src.receipt_intelligence import logger


STAGE_NAME = "INFERENCE STAGE"

class InferencePipeline:

    def __init__(self):
        pass

    def main(self):

        config = ConfigurationManager()

        inference_config = (
            config.get_inference_config()
        )

        inference = ReceiptInferencePipeline(
            config=config, inference_config=inference_config
        )

        result = (
            inference.predict_directory()
        )

if __name__ == "__main__":
    try:

        logger.info(
            f">>>>>> stage {STAGE_NAME} started <<<<<<"
        )

        obj = (
               InferencePipeline()
        )

        obj.main()

        logger.info(
            f">>>>>> stage {STAGE_NAME} completed <<<<<<"
        )

    except Exception as e:

        logger.exception(e)

        raise e