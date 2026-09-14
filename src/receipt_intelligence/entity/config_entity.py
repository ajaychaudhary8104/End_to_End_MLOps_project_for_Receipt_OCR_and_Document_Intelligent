from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataValidationConfig:
    """
    Configuration for validating receipt image datasets.
    """
    root_dir: Path
    data_dir: Path
    report_dir: Path
    min_width: int 
    min_height: int 
    max_width: int 
    max_height: int 
    min_file_size_bytes: int 
    max_file_size_mb: int 
    supported_extensions: tuple[str, ...]
    recursive: bool 
    fail_on_empty_dataset: bool 
    fail_on_corrupt_images: bool 
    duplicate_detection: bool 
    generate_report: bool     

@dataclass(frozen=True)
class ImagePreprocessingConfig:
    """
    Configuration for receipt image preprocessing.
    """
    root_dir: Path 
    input_dir: Path
    output_dir: Path
    target_width: int
    min_width: int 
    min_height: int 
    denoise: bool 
    clahe: bool
    adaptive_threshold: bool 
    deskew: bool 
    perspective_correction: bool 
    grayscale: bool 
    save_intermediate: bool 
    output_extension: str 
    jpeg_quality: int 
    max_rotation_angle: float 
    supported_extensions: tuple[str, ...]     

@dataclass(frozen=True)
class OCREngineConfig:
    """
    Configuration for PaddleOCR inference.
    """
    root_dir: Path
    input_dir: Path
    output_dir: Path
    model_name: str 
    language: str
    device: str 
    text_detection_score_threshold: float
    text_recognition_score_threshold: float 
    use_doc_orientation_classify: bool
    use_doc_unwarping: bool 
    use_textline_orientation: bool
    save_visualization: bool 
    recursive: bool 
    supported_extensions: tuple[str, ...]

@dataclass(frozen=True)
class TextCleaningConfig:
    """
    Configuration for OCR text normalization.

    This layer is intentionally conservative. Its purpose is
    to clean OCR noise while preserving business-relevant text.
    """
    root_dir: Path
    input_dir: Path
    output_dir: Path
    min_text_length: int
    normalize_unicode: bool 
    normalize_whitespace: bool 
    normalize_currency: bool 
    normalize_common_ocr_errors: bool 
    normalize_dates: bool
    preserve_case: bool
    remove_control_characters: bool 
    deduplicate_adjacent_lines: bool 
    remove_empty_lines: bool
    confidence_round_digits: int

@dataclass(frozen=True)
class FieldExtractionConfig:
    """
    Configuration for hybrid receipt field extraction.
    """
    root_dir: Path
    input_dir: Path
    output_dir: Path
    vendor_search_lines: int
    min_item_name_length: int
    max_item_name_length: int
    min_item_price: float
    max_item_price: float
    total_keywords: tuple[str, ...]
    date_search_lines: int
    total_search_last_lines: int
    reject_keywords: tuple[str, ...]
    confidence_round_digits: int            

@dataclass(frozen=True)
class ConfidenceEngineConfig:
    """
    Configuration for field-level confidence scoring.
    """
    root_dir: Path
    input_dir: Path
    ocr_dir: Path
    cleaned_ocr_dir: Path
    output_dir: Path
    low_confidence_threshold: float 
    medium_confidence_threshold: float
    ocr_weight: float
    pattern_weight: float
    context_weight: float
    conflict_penalty: float 
    item_count_mismatch_penalty: float 
    absolute_total_tolerance: float 
    relative_total_tolerance: float 
    confidence_digits: int
    save_manifest: bool    


@dataclass(frozen=True)
class JSONGeneratorConfig:
    root_dir: Path
    input_dir: Path
    output_dir: Path
    include_confidence: bool 
    include_reliability: bool 
    include_review_flag: bool 
    include_evidence: bool 
    include_source_metadata: bool 
    include_extraction_metadata: bool 
    round_amount_digits: int
    round_confidence_digits: int
    medium_confidence_threshold: float 
    low_confidence_threshold: float 
    save_manifest: bool 
    schema_version: str 

@dataclass(frozen=True)
class FinancialSummaryConfig:
    root_dir: Path
    input_dir: Path
    output_dir: Path
    minimum_confidence: float
    exclude_review_required: bool 
    require_validation_passed: bool 
    include_low_confidence_receipts: bool 
    currency_decimals: int
    save_receipt_table: bool 
    save_audit_table: bool 
    save_csv: bool 
    save_json: bool 
    schema_version: str    

@dataclass(frozen=True)
class EvaluationConfig:
    root_dir : Path
    prediction_dir: Path
    ground_truth_dir: Path
    ocr_prediction_dir: Path
    output_dir: Path
    ocr_prediction_text_field: str 
    ocr_prediction_fallback_fields: tuple[str, ...]
    allow_prediction_text_fallback: bool
    text_field_in_ground_truth: str 
    case_sensitive: bool
    normalize_whitespace: bool
    remove_punctuation_for_text_eval: bool 
    strict_string_matching: bool 
    string_similarity_threshold: float
    amount_tolerance: float
    item_price_tolerance: float 
    item_name_similarity_threshold: float 
    normalize_dates: bool 
    date_order: str 
    cer_warning_threshold: float
    wer_warning_threshold: float 
    item_f1_warning_threshold: float 
    confidence_bin_count: int
    quality_gate_enabled: bool
    max_corpus_cer: float | None
    max_corpus_wer: float | None
    min_store_accuracy: float | None
    min_date_accuracy: float | None 
    min_total_accuracy: float | None
    min_item_f1: float | None
    save_detailed_results: bool
    save_csv: bool
    save_json: bool     

@dataclass(frozen=True)
class MLflowConfig:
    root_dir: Path
    tracking_uri: str
    experiment_name: str 
    artifact_location: str 
    run_name_prefix: str 
    local_summary_dir: str 
    local_artifact_dir: str 
    log_system_metrics: bool 
    save_local_run_summary: bool 
    log_environment: bool
    log_git_metadata: bool 
    log_python_metadata: bool
    log_mlflow_metadata: bool 
    allow_existing_active_run: bool 
    max_param_length: int
    max_tag_length: int 
    max_metric_count: int 
    hash_dataset_files: bool 
    strict_metric_validation: bool 
    strict_parameter_validation: bool 
    strict_tag_validation: bool 
    secret_key_patterns: tuple[str, ...]        