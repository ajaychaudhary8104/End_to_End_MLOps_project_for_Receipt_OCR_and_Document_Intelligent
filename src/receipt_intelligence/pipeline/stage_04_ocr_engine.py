from src.receipt_intelligence.config.configuration import ConfigurationManager
from src.receipt_intelligence.components.ocr_engine import OCREngine
from src.receipt_intelligence import logger


STAGE_NAME = "OCR ENGINE STAGE"

class OCREngineTrainingPipeline:

    def __init__(self):
        pass

    def main(self):

        config = ConfigurationManager()

        ocr_config = (
            config.get_ocr_engine_config()
        )

        ocr_engine = OCREngine(
            config=ocr_config
        )

        summary = (
            ocr_engine.run()
        )

if __name__ == "__main__":
    try:

        logger.info(
            f">>>>>> stage {STAGE_NAME} started <<<<<<"
        )

        obj = (
            OCREngineTrainingPipeline()
        )

        obj.main()

        logger.info(
            f">>>>>> stage {STAGE_NAME} completed <<<<<<"
        )

    except Exception as e:

        logger.exception(e)

        raise e