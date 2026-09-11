from src.receipt_intelligence.config.configuration import ConfigurationManager
from src.receipt_intelligence.components.confindence_and_reliability_engine import ConfidenceEngine
from src.receipt_intelligence import logger


STAGE_NAME = "CONFIDENCE AND RELIABILITY STAGE"

class ConfidenceAndReliabilityPipeline:

    def __init__(self):
        pass

    def main(self):

        config = ConfigurationManager()

        confidence_and_reliability_config = (
            config.get_confidence_and_reliability_engine_config()
        )

        confidence_engine = ConfidenceEngine(
            config=confidence_and_reliability_config
        )

        summary = (
            confidence_engine.run()
        )

if __name__ == "__main__":
    try:

        logger.info(
            f">>>>>> stage {STAGE_NAME} started <<<<<<"
        )

        obj = (
            ConfidenceAndReliabilityPipeline()
        )

        obj.main()

        logger.info(
            f">>>>>> stage {STAGE_NAME} completed <<<<<<"
        )

    except Exception as e:

        logger.exception(e)

        raise e