from src.receipt_intelligence.config.configuration import ConfigurationManager
from src.receipt_intelligence.components.evaluation import EvaluationEngine
from src.receipt_intelligence import logger


STAGE_NAME = "EVALUATION STAGE"

class EvaluationPipeline:

    def __init__(self):
        pass

    def main(self):

        config = ConfigurationManager()

        evaluation_config = (
            config.get_evaluation_config()
        )

        evaluation = EvaluationEngine(
            config=evaluation_config
        )

        summary = (
            evaluation.run()
        )

if __name__ == "__main__":
    try:

        logger.info(
            f">>>>>> stage {STAGE_NAME} started <<<<<<"
        )

        obj = (
               EvaluationPipeline()
        )

        obj.main()

        logger.info(
            f">>>>>> stage {STAGE_NAME} completed <<<<<<"
        )

    except Exception as e:

        logger.exception(e)

        raise e