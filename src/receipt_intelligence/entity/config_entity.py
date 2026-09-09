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