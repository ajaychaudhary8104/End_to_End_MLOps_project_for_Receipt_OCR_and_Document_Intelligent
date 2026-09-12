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