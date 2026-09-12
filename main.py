from src.receipt_intelligence import logger
from src.receipt_intelligence.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.receipt_intelligence.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from src.receipt_intelligence.pipeline.stage_03_image_preprocessing import ImagePreprocessingTrainingPipeline
from receipt_intelligence.pipeline.stage_04_ocr_engine import OCREngineTrainingPipeline
from receipt_intelligence.pipeline.stage_05_ocr_text_cleaning import OCRTextCleaningPipeline
from receipt_intelligence.pipeline.stage_06_field_extraction_engine import FieldExtractionPipeline
from src.receipt_intelligence.pipeline.stage_07_confidence_and_reliability_engine import ConfidenceAndReliabilityPipeline
from src.receipt_intelligence.pipeline.stage_08_json_generation import JSONGenerationPipeline
from src.receipt_intelligence.pipeline.stage_09_financial_summary import FinancialSummaryPipeline
import warnings

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning
)

warnings.filterwarnings(
    "ignore",
    category=FutureWarning
)

STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
   logger.exception(e)
   raise e

STAGE_NAME = "Data Validation stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   data_validation = DataValidationTrainingPipeline()
   data_validation.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
   logger.exception(e)
   raise e

STAGE_NAME = "Image Preprocessing stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   image_preprocessing= ImagePreprocessingTrainingPipeline()
   image_preprocessing.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
   logger.exception(e)
   raise e

STAGE_NAME = "OCR Engine stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   ocr_engine = OCREngineTrainingPipeline()
   ocr_engine.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
   logger.exception(e)
   raise e   


STAGE_NAME = "OCR Text Cleaning stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   ocr_text_cleaning = OCRTextCleaningPipeline()
   ocr_text_cleaning.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
   logger.exception(e)
   raise e   

STAGE_NAME = "Field Extraction stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   field_extraction = FieldExtractionPipeline()
   field_extraction.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
   logger.exception(e)
   raise e   

STAGE_NAME = "Confidence and Reliability stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   confidence_and_reliability = ConfidenceAndReliabilityPipeline()
   confidence_and_reliability.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
   logger.exception(e)
   raise e   

STAGE_NAME = "JSON Generation stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   json_generation = JSONGenerationPipeline()
   json_generation.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
   logger.exception(e)
   raise e   

STAGE_NAME = "Financial Summary stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   financial_summary = FinancialSummaryPipeline()
   financial_summary.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
   logger.exception(e)
   raise e   
