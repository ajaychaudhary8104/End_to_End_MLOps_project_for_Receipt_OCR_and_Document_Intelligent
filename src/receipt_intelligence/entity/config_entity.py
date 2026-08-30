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