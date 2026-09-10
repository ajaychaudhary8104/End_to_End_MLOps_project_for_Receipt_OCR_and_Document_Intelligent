from src.receipt_intelligence.config.configuration import ConfigurationManager
from src.receipt_intelligence.components.ocr_text_cleaning import OCRTextCleaner
from src.receipt_intelligence import logger


STAGE_NAME = "OCR TEXT CLEANING STAGE"

class OCRTextCleaningPipeline:

    def __init__(self):
        pass

    def main(self):

        config = ConfigurationManager()

        ocr_text_cleaning_config = (
            config.get_ocr_text_cleaning_config()
        )

        ocr_text_cleaner = OCRTextCleaner(
            config=ocr_text_cleaning_config
        )

        summary = (
            ocr_text_cleaner.run()
        )

if __name__ == "__main__":
    try:

        logger.info(
            f">>>>>> stage {STAGE_NAME} started <<<<<<"
        )

        obj = (
            OCRTextCleaningPipeline()
        )

        obj.main()

        logger.info(
            f">>>>>> stage {STAGE_NAME} completed <<<<<<"
        )

    except Exception as e:

        logger.exception(e)

        raise e