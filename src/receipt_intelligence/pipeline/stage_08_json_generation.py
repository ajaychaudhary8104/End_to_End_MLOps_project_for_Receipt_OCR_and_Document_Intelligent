from src.receipt_intelligence.config.configuration import ConfigurationManager
from src.receipt_intelligence.components.json_generation import JSONGenerator
from src.receipt_intelligence import logger


STAGE_NAME = "JSON GENERATION STAGE"

class JSONGenerationPipeline:

    def __init__(self):
        pass

    def main(self):

        config = ConfigurationManager()

        json_generator_config = (
            config.get_json_generator_config()
        )

        json_generator = JSONGenerator(
            config=json_generator_config
        )

        summary = (
            json_generator.run()
        )

if __name__ == "__main__":
    try:

        logger.info(
            f">>>>>> stage {STAGE_NAME} started <<<<<<"
        )

        obj = (
            JSONGenerationPipeline()
        )

        obj.main()

        logger.info(
            f">>>>>> stage {STAGE_NAME} completed <<<<<<"
        )

    except Exception as e:

        logger.exception(e)

        raise e