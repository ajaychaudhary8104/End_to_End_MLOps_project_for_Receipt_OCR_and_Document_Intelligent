from src.receipt_intelligence.config.configuration import ConfigurationManager
from src.receipt_intelligence.components.experiment_tracking_with_mlflow import ReceiptOCRPipeline, MLflowExperimentTracker
from src.receipt_intelligence import logger


STAGE_NAME = "MLFOW TRACKER STAGE"

class MLFlowTrackerPipeline:

    def __init__(self):
        pass

    def main(self):

        receipt_ocr_pipeline = ReceiptOCRPipeline()
        receipt_ocr_pipeline.run_pipeline()
        

if __name__ == "__main__":
    try:

        logger.info(
            f">>>>>> stage {STAGE_NAME} started <<<<<<"
        )

        obj = (
            MLFlowTrackerPipeline()
        )

        obj.main()

        logger.info(
            f">>>>>> stage {STAGE_NAME} completed <<<<<<"
        )

    except Exception as e:

        logger.exception(e)

        raise e