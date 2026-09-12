from src.receipt_intelligence.config.configuration import ConfigurationManager
from src.receipt_intelligence.components.financial_summary import FinancialSummaryEngine
from src.receipt_intelligence import logger


STAGE_NAME = "FINANCIAL SUMMARY STAGE"

class FinancialSummaryPipeline:

    def __init__(self):
        pass

    def main(self):

        config = ConfigurationManager()

        financial_summary_config = (
            config.get_financial_summary_config()
        )

        financial_summary = FinancialSummaryEngine(
            config=financial_summary_config
        )

        summary = (
            financial_summary.run()
        )

if __name__ == "__main__":
    try:

        logger.info(
            f">>>>>> stage {STAGE_NAME} started <<<<<<"
        )

        obj = (
            FinancialSummaryPipeline()
        )

        obj.main()

        logger.info(
            f">>>>>> stage {STAGE_NAME} completed <<<<<<"
        )

    except Exception as e:

        logger.exception(e)

        raise e