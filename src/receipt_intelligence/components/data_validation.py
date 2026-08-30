from __future__ import annotations
import hashlib
import json
import mimetypes
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from src.receipt_intelligence import logger
from src.receipt_intelligence.entity.config_entity import DataValidationConfig



# ============================================================
# VALIDATION RESULT
# ============================================================

@dataclass
class ImageValidationResult:
    """
    Validation result for a single image.
    """

    file_path: str

    file_name: str

    extension: str

    exists: bool

    readable: bool

    valid_format: bool

    file_size_bytes: int

    width: int | None

    height: int | None

    channels: int | None

    megapixels: float | None

    is_empty: bool

    is_corrupt: bool

    is_duplicate: bool

    duplicate_of: str | None

    mime_type: str | None

    sha256: str | None

    errors: list[str]

    warnings: list[str]


# ============================================================
# DATA VALIDATION CLASS
# ============================================================

class DataValidation:
    """
    Production-grade validation layer for receipt images.

    Validation flow:

        Dataset Directory
                ↓
        File Discovery
                ↓
        Extension Validation
                ↓
        File Size Validation
                ↓
        Image Decode Validation
                ↓
        Dimension Validation
                ↓
        Empty / Corrupt Check
                ↓
        Duplicate Detection
                ↓
        Dataset-Level Validation
                ↓
        JSON Report
    """

    def __init__(self, config: DataValidationConfig):
        self.config = config

        self.data_dir = Path(config.data_dir)
        self.report_dir = Path(config.report_dir)

        self.report_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.results: list[ImageValidationResult] = []

        self._hash_to_path: dict[str, str] = {}

        self._stats: dict[str, Any] = {}

    # ========================================================
    # FILE DISCOVERY
    # ========================================================

    def _discover_files(self) -> list[Path]:
        """
        Discover candidate files from the dataset directory.
        """

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory does not exist: {self.data_dir}"
            )

        if not self.data_dir.is_dir():
            raise NotADirectoryError(
                f"Expected directory but found: {self.data_dir}"
            )

        if self.config.recursive:
            all_files = [
                path
                for path in self.data_dir.rglob("*")
                if path.is_file()
            ]
        else:
            all_files = [
                path
                for path in self.data_dir.iterdir()
                if path.is_file()
            ]

        supported = {
            ext
            for ext in self.config.supported_extensions
        }

        image_files = sorted(
            path
            for path in all_files
            if path.suffix.lower() in supported
        )

        logger.info(
            "Discovered %d candidate receipt images.",
            len(image_files)
        )

        return image_files

    # ========================================================
    # SHA256
    # ========================================================

    @staticmethod
    def _calculate_sha256(file_path: Path) -> str:
        """
        Calculate SHA256 checksum using streaming reads.
        """

        sha256 = hashlib.sha256()

        with file_path.open("rb") as file:

            while True:

                chunk = file.read(1024 * 1024)

                if not chunk:
                    break

                sha256.update(chunk)

        return sha256.hexdigest()

    # ========================================================
    # SINGLE IMAGE VALIDATION
    # ========================================================

    def _validate_image(
        self,
        image_path: Path
    ) -> ImageValidationResult:
        """
        Validate one receipt image.
        """

        errors: list[str] = []
        warnings: list[str] = []

        exists = image_path.exists()

        file_size = 0

        width = None
        height = None
        channels = None
        megapixels = None

        readable = False
        valid_format = False
        is_empty = False
        is_corrupt = False
        is_duplicate = False
        duplicate_of = None

        sha256 = None

        extension = image_path.suffix.lower()

        mime_type, _ = mimetypes.guess_type(
            image_path.name
        )

        # ----------------------------------------------------
        # EXISTENCE
        # ----------------------------------------------------

        if not exists:

            errors.append(
                "File does not exist."
            )

            return ImageValidationResult(
                file_path=str(image_path),
                file_name=image_path.name,
                extension=extension,
                exists=False,
                readable=False,
                valid_format=False,
                file_size_bytes=0,
                width=None,
                height=None,
                channels=None,
                megapixels=None,
                is_empty=True,
                is_corrupt=True,
                is_duplicate=False,
                duplicate_of=None,
                mime_type=mime_type,
                sha256=None,
                errors=errors,
                warnings=warnings,
            )

        # ----------------------------------------------------
        # EXTENSION
        # ----------------------------------------------------

        supported_extensions = {
            ext.lower()
            for ext in self.config.supported_extensions
        }

        valid_format = (
            extension in supported_extensions
        )

        if not valid_format:

            errors.append(
                f"Unsupported image extension: {extension}"
            )

        # ----------------------------------------------------
        # FILE SIZE
        # ----------------------------------------------------

        try:
            file_size = image_path.stat().st_size

        except OSError as exc:

            errors.append(
                f"Unable to read file metadata: {exc}"
            )

        if file_size < self.config.min_file_size_bytes:

            is_empty = True

            errors.append(
                f"File size {file_size} bytes is below "
                f"minimum {self.config.min_file_size_bytes} bytes."
            )

        max_size_bytes = (
            self.config.max_file_size_mb
            * 1024
            * 1024
        )

        if file_size > max_size_bytes:

            warnings.append(
                f"File exceeds recommended maximum size: "
                f"{self.config.max_file_size_mb} MB."
            )

        # ----------------------------------------------------
        # CHECKSUM
        # ----------------------------------------------------

        if file_size > 0:

            try:

                sha256 = self._calculate_sha256(
                    image_path
                )

            except Exception as exc:

                warnings.append(
                    f"SHA256 calculation failed: {exc}"
                )

        # ----------------------------------------------------
        # DUPLICATE DETECTION
        # ----------------------------------------------------

        if (
            self.config.duplicate_detection
            and sha256 is not None
        ):

            if sha256 in self._hash_to_path:

                is_duplicate = True

                duplicate_of = (
                    self._hash_to_path[sha256]
                )

                warnings.append(
                    f"Duplicate image detected. "
                    f"Same content as: {duplicate_of}"
                )

            else:

                self._hash_to_path[sha256] = (
                    str(image_path)
                )

        # ----------------------------------------------------
        # IMAGE DECODE
        # ----------------------------------------------------

        try:

            image = cv2.imread(
                str(image_path),
                cv2.IMREAD_UNCHANGED
            )

            if image is None:

                is_corrupt = True

                errors.append(
                    "OpenCV could not decode the image."
                )

            else:

                readable = True

                shape = image.shape

                if len(shape) == 2:

                    height, width = shape
                    channels = 1

                elif len(shape) == 3:

                    height, width, channels = shape

                else:

                    is_corrupt = True

                    errors.append(
                        f"Unexpected image shape: {shape}"
                    )

        except Exception as exc:

            is_corrupt = True

            errors.append(
                f"Image decoding failed: {exc}"
            )

        # ----------------------------------------------------
        # DIMENSION VALIDATION
        # ----------------------------------------------------

        if width is not None and height is not None:

            if width <= 0 or height <= 0:

                is_empty = True

                errors.append(
                    "Image has invalid dimensions."
                )

            elif (
                width < self.config.min_width
                or height < self.config.min_height
            ):

                warnings.append(
                    f"Image resolution {width}x{height} "
                    f"is below recommended minimum "
                    f"{self.config.min_width}x"
                    f"{self.config.min_height}."
                )

            elif (
                width > self.config.max_width
                or height > self.config.max_height
            ):

                warnings.append(
                    f"Image resolution {width}x{height} "
                    f"exceeds recommended maximum "
                    f"{self.config.max_width}x"
                    f"{self.config.max_height}."
                )

            megapixels = (
                width * height / 1_000_000
            )

        # ----------------------------------------------------
        # IMAGE CHANNEL VALIDATION
        # ----------------------------------------------------

        if channels is not None:

            if channels not in (1, 3, 4):

                warnings.append(
                    f"Unusual channel count: {channels}"
                )

        # ----------------------------------------------------
        # COMPLETELY WHITE / BLACK IMAGE DETECTION
        # ----------------------------------------------------

        if readable:

            try:

                image_gray = cv2.imread(
                    str(image_path),
                    cv2.IMREAD_GRAYSCALE
                )

                if image_gray is not None:

                    min_pixel = int(
                        image_gray.min()
                    )

                    max_pixel = int(
                        image_gray.max()
                    )

                    # Completely blank image
                    if min_pixel == max_pixel:

                        is_empty = True

                        warnings.append(
                            f"Image contains a constant pixel "
                            f"value ({min_pixel}); likely blank."
                        )

            except Exception as exc:

                warnings.append(
                    f"Blank-image check failed: {exc}"
                )

        # ----------------------------------------------------
        # BUILD RESULT
        # ----------------------------------------------------

        return ImageValidationResult(
            file_path=str(image_path),
            file_name=image_path.name,
            extension=extension,
            exists=exists,
            readable=readable,
            valid_format=valid_format,
            file_size_bytes=file_size,
            width=width,
            height=height,
            channels=channels,
            megapixels=megapixels,
            is_empty=is_empty,
            is_corrupt=is_corrupt,
            is_duplicate=is_duplicate,
            duplicate_of=duplicate_of,
            mime_type=mime_type,
            sha256=sha256,
            errors=errors,
            warnings=warnings,
        )

    # ========================================================
    # DATASET LEVEL STATISTICS
    # ========================================================

    def _calculate_statistics(self) -> dict[str, Any]:
        """
        Calculate dataset-level validation statistics.
        """

        total = len(self.results)

        valid = sum(
            1
            for result in self.results
            if (
                result.exists
                and result.readable
                and result.valid_format
                and not result.is_corrupt
                and not result.is_empty
            )
        )

        invalid = total - valid

        corrupt = sum(
            result.is_corrupt
            for result in self.results
        )

        unreadable = sum(
            not result.readable
            for result in self.results
        )

        empty = sum(
            result.is_empty
            for result in self.results
        )

        duplicates = sum(
            result.is_duplicate
            for result in self.results
        )

        total_size_bytes = sum(
            result.file_size_bytes
            for result in self.results
        )

        widths = [
            result.width
            for result in self.results
            if result.width is not None
        ]

        heights = [
            result.height
            for result in self.results
            if result.height is not None
        ]

        statistics = {
            "total_files": total,
            "valid_files": valid,
            "invalid_files": invalid,
            "corrupt_files": corrupt,
            "unreadable_files": unreadable,
            "empty_files": empty,
            "duplicate_files": duplicates,
            "validity_rate": (
                valid / total
                if total > 0
                else 0.0
            ),
            "total_size_mb": (
                total_size_bytes
                / (1024 * 1024)
            ),
            "min_width": min(widths)
                if widths else None,
            "max_width": max(widths)
                if widths else None,
            "mean_width": (
                float(np.mean(widths))
                if widths else None
            ),
            "min_height": min(heights)
                if heights else None,
            "max_height": max(heights)
                if heights else None,
            "mean_height": (
                float(np.mean(heights))
                if heights else None
            ),
        }

        return statistics

    # ========================================================
    # REPORT GENERATION
    # ========================================================

    def _save_report(self) -> Path:
        """
        Save complete validation report as JSON.
        """

        timestamp = datetime.now(
            timezone.utc
        ).strftime(
            "%Y%m%dT%H%M%SZ"
        )

        report_path = (
            self.report_dir
            / f"data_validation_report_{timestamp}.json"
        )

        report = {
            "validation_timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat(),

            "dataset_directory": str(
                self.data_dir.resolve()
            ),

            "configuration": {
                key: list(value)
                if isinstance(value, tuple)
                else value
                for key, value in asdict(
                    self.config
                ).items()
            },

            "statistics": self._stats,

            "images": [
                asdict(result)
                for result in self.results
            ],
        }

        with report_path.open(
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                report,
                file,
                indent=2,
                ensure_ascii=False,
                default=str
            )

        logger.info(
            "Validation report saved to: %s",
            report_path
        )

        return report_path

    # ========================================================
    # DATA QUALITY GATE
    # ========================================================

    def _quality_gate(self) -> None:
        """
        Fail the pipeline only for configured critical conditions.
        """

        total_files = self._stats["total_files"]

        if (
            total_files == 0
            and self.config.fail_on_empty_dataset
        ):

            raise ValueError(
                "Data validation failed: "
                "no supported receipt images found."
            )

        corrupt_files = self._stats[
            "corrupt_files"
        ]

        if (
            corrupt_files > 0
            and self.config.fail_on_corrupt_images
        ):

            raise ValueError(
                f"Data validation failed: "
                f"{corrupt_files} corrupt/unreadable "
                f"image(s) detected."
            )

        valid_files = self._stats[
            "valid_files"
        ]

        if total_files > 0 and valid_files == 0:

            raise ValueError(
                "Data validation failed: "
                "no valid receipt images found."
            )

    # ========================================================
    # MAIN VALIDATION
    # ========================================================

    def validate(self) -> dict[str, Any]:
        """
        Run complete dataset validation.

        Returns:
            Dictionary containing validation statistics
            and report information.
        """

        logger.info(
            "Starting receipt dataset validation..."
        )

        image_files = self._discover_files()

        # ----------------------------------------------------
        # Validate every image
        # ----------------------------------------------------

        self.results = []

        for index, image_path in enumerate(
            image_files,
            start=1
        ):

            result = self._validate_image(
                image_path
            )

            self.results.append(result)

            if result.errors:

                logger.warning(
                    "Validation errors for %s: %s",
                    image_path,
                    result.errors
                )

            if result.warnings:

                logger.warning(
                    "Validation warnings for %s: %s",
                    image_path,
                    result.warnings
                )

            if index % 100 == 0:

                logger.info(
                    "Validated %d/%d images.",
                    index,
                    len(image_files)
                )

        # ----------------------------------------------------
        # Dataset statistics
        # ----------------------------------------------------

        self._stats = self._calculate_statistics()

        # ----------------------------------------------------
        # Quality gate
        # ----------------------------------------------------

        self._quality_gate()

        # ----------------------------------------------------
        # Report
        # ----------------------------------------------------

        report_path = None

        if self.config.generate_report:

            report_path = self._save_report()

        # ----------------------------------------------------
        # Final result
        # ----------------------------------------------------

        validation_summary = {
            "status": "PASSED",
            "statistics": self._stats,
            "report_path": (
                str(report_path)
                if report_path is not None
                else None
            ),
        }

        logger.info(
            "Data validation completed successfully."
        )

        logger.info(
            "Total files: %d",
            self._stats["total_files"]
        )

        logger.info(
            "Valid files: %d",
            self._stats["valid_files"]
        )

        logger.info(
            "Invalid files: %d",
            self._stats["invalid_files"]
        )

        logger.info(
            "Duplicate files: %d",
            self._stats["duplicate_files"]
        )

        return validation_summary

    # ========================================================
    # CONVENIENCE METHOD
    # ========================================================

    def run(self) -> dict[str, Any]:
        """
        Pipeline-friendly entry point.
        """

        return self.validate()