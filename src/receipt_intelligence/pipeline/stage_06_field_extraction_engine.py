from src.receipt_intelligence.config.configuration import ConfigurationManager
from src.receipt_intelligence.components.field_extraction_engine import FieldExtractor
from src.receipt_intelligence import logger


STAGE_NAME = "FIELD EXTRACTION STAGE"

class FieldExtractionPipeline:

    def __init__(self):
        pass

    def main(self):

        config = ConfigurationManager()

        field_extraction_config = (
            config.get_field_extraction_engine_config()
        )

        field_extractor = FieldExtractor(
            config=field_extraction_config
        )

        summary = (
            field_extractor.run()
        )

if __name__ == "__main__":
    try:

        logger.info(
            f">>>>>> stage {STAGE_NAME} started <<<<<<"
        )

        obj = (
            FieldExtractionPipeline()
        )

        obj.main()

        logger.info(
            f">>>>>> stage {STAGE_NAME} completed <<<<<<"
        )

    except Exception as e:

        logger.exception(e)

        raise e