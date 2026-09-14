from src.receipt_intelligence.constants import *
from src.receipt_intelligence.utils.common import read_yaml, create_directories
from src.receipt_intelligence.entity.config_entity import (DataIngestionConfig, DataValidationConfig ,
                                                            ImagePreprocessingConfig, OCREngineConfig,
                                                            TextCleaningConfig, FieldExtractionConfig,
                                                              ConfidenceEngineConfig ,JSONGeneratorConfig,
                                                              FinancialSummaryConfig, EvaluationConfig)

class ConfigurationManager:
    def __init__(self, config_filepath = CONFIG_FILE_PATH, params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
    
        config = self.config.data_validation

        create_directories(
            [config.root_dir]
        )

        validation_config = (DataValidationConfig(
                root_dir=Path(config.root_dir),
                data_dir=Path(config.data_dir),
                report_dir=Path(config.report_dir),
                min_width=config.min_width,
                min_height=config.min_height,
                max_width=config.max_width,
                max_height=config.max_height,
                min_file_size_bytes=config.min_file_size_bytes,
                max_file_size_mb=config.max_file_size_mb,
                supported_extensions = tuple(config.supported_extensions),
                recursive = config.recursive,
                fail_on_empty_dataset= config.fail_on_empty_dataset,
                fail_on_corrupt_images= config.fail_on_corrupt_images,
                duplicate_detection= config.duplicate_detection,
                generate_report= config.generate_report
                
            )
        )

        return validation_config

    def get_image_preprocessing_config(self) -> ImagePreprocessingConfig:
    
        config = self.config.image_preprocessing

        create_directories(
            [config.root_dir]
        )

        image_preprocessing_config = (ImagePreprocessingConfig(
                root_dir=Path(config.root_dir),
                input_dir=Path(config.input_dir),
                output_dir=Path(config.output_dir),
                target_width=config.target_width,
                min_width=config.min_width,
                min_height=config.min_height,
                denoise=config.denoise,
                clahe=config.clahe,
                adaptive_threshold=config.adaptive_threshold,
                deskew=config.deskew,
                perspective_correction=config.perspective_correction,
                grayscale=config.grayscale,
                save_intermediate=config.save_intermediate,
                output_extension=config.output_extension,
                jpeg_quality=config.jpeg_quality,
                max_rotation_angle=config.max_rotation_angle,
                supported_extensions=tuple(config.supported_extensions)
            )
        )

        return image_preprocessing_config

    def get_ocr_engine_config(self) -> OCREngineConfig:
        ocr_engine_config = self.config.ocr_engine

        create_directories(
            [ocr_engine_config.root_dir, ocr_engine_config.output_dir]
        )

        return OCREngineConfig(
            root_dir=Path(ocr_engine_config.root_dir),
            input_dir=Path(ocr_engine_config.input_dir),
            output_dir=Path(ocr_engine_config.output_dir),
            model_name=ocr_engine_config.model_name,
            language=ocr_engine_config.language,
            device=ocr_engine_config.device,
            text_detection_score_threshold=ocr_engine_config.text_detection_score_threshold,
            text_recognition_score_threshold=ocr_engine_config.text_recognition_score_threshold,
            use_doc_orientation_classify=ocr_engine_config.use_doc_orientation_classify,
            use_doc_unwarping=ocr_engine_config.use_doc_unwarping,
            use_textline_orientation=ocr_engine_config.use_textline_orientation,
            save_visualization=ocr_engine_config.save_visualization,
            recursive=ocr_engine_config.recursive,
            supported_extensions=tuple(ocr_engine_config.supported_extensions)
        )        

    def get_ocr_text_cleaning_config(self) -> TextCleaningConfig:
        ocr_text_cleaning_config = self.config.ocr_text_cleaning
        create_directories([ocr_text_cleaning_config.root_dir,ocr_text_cleaning_config.output_dir])

        return TextCleaningConfig(
            root_dir=ocr_text_cleaning_config.root_dir,
            input_dir=ocr_text_cleaning_config.input_dir,
            output_dir=ocr_text_cleaning_config.output_dir,
            min_text_length=ocr_text_cleaning_config.min_text_length,
            normalize_unicode=ocr_text_cleaning_config.normalize_unicode,
            normalize_whitespace=ocr_text_cleaning_config.normalize_whitespace,
            normalize_currency=ocr_text_cleaning_config.normalize_currency,
            normalize_common_ocr_errors=ocr_text_cleaning_config.normalize_common_ocr_errors,
            normalize_dates=ocr_text_cleaning_config.normalize_dates,
            preserve_case=ocr_text_cleaning_config.preserve_case,
            remove_control_characters=ocr_text_cleaning_config.remove_control_characters,
            deduplicate_adjacent_lines=ocr_text_cleaning_config.deduplicate_adjacent_lines,
            remove_empty_lines=ocr_text_cleaning_config.remove_empty_lines,
            confidence_round_digits=ocr_text_cleaning_config.confidence_round_digits
        )

    def get_field_extraction_engine_config(self) -> FieldExtractionConfig:
        field_extraction_config = self.config.field_extraction_engine
        create_directories([Path(field_extraction_config.root_dir) , Path(field_extraction_config.output_dir)])
        
        fe_config = FieldExtractionConfig(
            root_dir=Path(field_extraction_config.root_dir),
            input_dir=Path(field_extraction_config.input_dir),
            output_dir=Path(field_extraction_config.output_dir),
            vendor_search_lines=field_extraction_config.vendor_search_lines,
            total_search_last_lines=field_extraction_config.total_search_last_lines,
            min_item_name_length=field_extraction_config.min_item_name_length,
            max_item_name_length=field_extraction_config.max_item_name_length,
            min_item_price=field_extraction_config.min_item_price,
            max_item_price=field_extraction_config.max_item_price,
            total_keywords=tuple(field_extraction_config.total_keywords),
            date_search_lines=field_extraction_config.date_search_lines,
            reject_keywords=tuple(field_extraction_config.reject_keywords),
            confidence_round_digits=field_extraction_config.confidence_round_digits,

        )

        return fe_config 

    def get_confidence_and_reliability_engine_config(self) -> ConfidenceEngineConfig:
        config = self.config.confidence_and_reliability_engine

        confidence_engine_config = ConfidenceEngineConfig(
            root_dir=Path(config.root_dir),
            input_dir=Path(config.input_dir),
            ocr_dir=Path(config.ocr_dir),
            cleaned_ocr_dir=Path(config.cleaned_ocr_dir),
            output_dir=Path(config.output_dir),
            low_confidence_threshold=config.low_confidence_threshold,
            medium_confidence_threshold=config.medium_confidence_threshold,
            ocr_weight=config.ocr_weight,
            pattern_weight=config.pattern_weight,
            context_weight=config.context_weight,
            conflict_penalty=config.conflict_penalty,
            item_count_mismatch_penalty=config.item_count_mismatch_penalty,
            absolute_total_tolerance=config.absolute_total_tolerance,
            relative_total_tolerance=config.relative_total_tolerance,
            confidence_digits=config.confidence_digits,
            save_manifest=config.save_manifest
        )

        return confidence_engine_config    

    def get_json_generator_config(self) -> JSONGeneratorConfig:
        config = self.config.json_generation

        json_generator_config = JSONGeneratorConfig(
            root_dir=Path(config.root_dir),
            input_dir=Path(config.input_dir),
            output_dir=Path(config.output_dir),
            include_confidence=config.include_confidence,
            include_reliability=config.include_reliability,
            include_review_flag=config.include_review_flag,
            include_evidence=config.include_evidence,
            include_source_metadata=config.include_source_metadata,
            include_extraction_metadata=config.include_extraction_metadata,
            round_amount_digits=config.round_amount_digits,
            round_confidence_digits=config.round_confidence_digits,
            medium_confidence_threshold=config.medium_confidence_threshold,
            low_confidence_threshold=config.low_confidence_threshold,
            save_manifest=config.save_manifest,
            schema_version=config.schema_version
        )

        return json_generator_config    

    def get_financial_summary_config(self) -> FinancialSummaryConfig:
        config = self.config.financial_summary

        financial_summary_config = FinancialSummaryConfig(
            root_dir=Path(config.root_dir),
            input_dir=Path(config.input_dir),
            output_dir=Path(config.output_dir),
            minimum_confidence=config.minimum_confidence,
            exclude_review_required=config.exclude_review_required,
            require_validation_passed=config.require_validation_passed,
            include_low_confidence_receipts=config.include_low_confidence_receipts,
            currency_decimals=config.currency_decimals,
            save_receipt_table=config.save_receipt_table,
            save_audit_table=config.save_audit_table,
            save_csv=config.save_csv,
            save_json=config.save_json,
            schema_version=config.schema_version
        )

        return financial_summary_config  

    def get_evaluation_config(self) -> EvaluationConfig:
        config = self.config.evaluation

        create_directories([
            config.root_dir,
            config.output_dir,
        ])

        return EvaluationConfig(
            root_dir=config.root_dir,
            prediction_dir=config.prediction_dir,
            ground_truth_dir=config.ground_truth_dir,
            ocr_prediction_dir=config.ocr_prediction_dir,
            output_dir=config.output_dir,
            ocr_prediction_text_field=config.ocr_prediction_text_field,
            ocr_prediction_fallback_fields=(
                config.ocr_prediction_fallback_fields
            ),
            allow_prediction_text_fallback=(
                config.allow_prediction_text_fallback
            ),
            text_field_in_ground_truth=(
                config.text_field_in_ground_truth
            ),
            case_sensitive=config.case_sensitive,
            normalize_whitespace=config.normalize_whitespace,
            remove_punctuation_for_text_eval=(
                config.remove_punctuation_for_text_eval
            ),
            strict_string_matching=(
                config.strict_string_matching
            ),
            string_similarity_threshold=(
                config.string_similarity_threshold
            ),
            amount_tolerance=config.amount_tolerance,
            item_price_tolerance=config.item_price_tolerance,
            item_name_similarity_threshold=(
                config.item_name_similarity_threshold
            ),
            normalize_dates=config.normalize_dates,
            date_order=config.date_order,
            cer_warning_threshold=(
                config.cer_warning_threshold
            ),
            wer_warning_threshold=(
                config.wer_warning_threshold
            ),
            item_f1_warning_threshold=(
                config.item_f1_warning_threshold
            ),
            confidence_bin_count=(
                config.confidence_bin_count
            ),
            quality_gate_enabled=(
                config.quality_gate_enabled
            ),
            max_corpus_cer=config.max_corpus_cer,
            max_corpus_wer=config.max_corpus_wer,
            min_store_accuracy=config.min_store_accuracy,
            min_date_accuracy=config.min_date_accuracy,
            min_total_accuracy=config.min_total_accuracy,
            min_item_f1=config.min_item_f1,
            save_detailed_results=(
                config.save_detailed_results
            ),
            save_csv=config.save_csv,
            save_json=config.save_json,
        )