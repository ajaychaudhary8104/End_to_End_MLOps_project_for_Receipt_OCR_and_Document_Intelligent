from src.receipt_intelligence.constants import *
from src.receipt_intelligence.utils.common import read_yaml, create_directories
from src.receipt_intelligence.entity.config_entity import DataIngestionConfig, DataValidationConfig , ImagePreprocessingConfig

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