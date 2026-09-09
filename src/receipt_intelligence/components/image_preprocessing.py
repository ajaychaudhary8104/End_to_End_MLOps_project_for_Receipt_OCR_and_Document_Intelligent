from __future__ import annotations
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from src.receipt_intelligence.entity.config_entity import ImagePreprocessingConfig
from src.receipt_intelligence import logger


# ============================================================
# PREPROCESSING RESULT
# ============================================================

@dataclass
class PreprocessingResult:
    """
    Metadata generated for every processed image.
    """

    receipt_id: str

    input_path: str

    output_path: str | None

    status: str

    original_width: int | None

    original_height: int | None

    processed_width: int | None

    processed_height: int | None

    rotation_angle: float

    perspective_corrected: bool

    denoised: bool

    contrast_enhanced: bool

    grayscale: bool

    errors: list[str]

    warnings: list[str]

    processing_time_ms: float

    timestamp_utc: str


# ============================================================
# IMAGE PREPROCESSOR
# ============================================================

class ImagePreprocessor:
    """
    Production-grade receipt image preprocessing pipeline.

    Pipeline:

        Raw Image
            ↓
        Validate Image
            ↓
        Resize
            ↓
        Grayscale
            ↓
        Denoising
            ↓
        CLAHE
            ↓
        Perspective Correction
            ↓
        Deskew
            ↓
        Optional Adaptive Threshold
            ↓
        Save Processed Image
    """

    def __init__(self, config: ImagePreprocessingConfig):
        self.config = config

        self.input_dir = Path(
            config.input_dir
        )

        self.output_dir = Path(
            config.output_dir
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self._supported_extensions = {
            extension.lower()
            for extension in config.supported_extensions
        }

        if config.target_width <= 0:
            raise ValueError(
                "target_width must be greater than zero."
            )

        if config.min_width <= 0:
            raise ValueError(
                "min_width must be greater than zero."
            )

        if config.min_height <= 0:
            raise ValueError(
                "min_height must be greater than zero."
            )

        if not 1 <= config.jpeg_quality <= 100:
            raise ValueError(
                "jpeg_quality must be between 1 and 100."
            )

    # ========================================================
    # IMAGE DISCOVERY
    # ========================================================

    def _discover_images(self) -> list[Path]:
        """
        Discover supported receipt images recursively.
        """

        if not self.input_dir.exists():
            raise FileNotFoundError(
                f"Input directory does not exist: "
                f"{self.input_dir}"
            )

        if not self.input_dir.is_dir():
            raise NotADirectoryError(
                f"Input path is not a directory: "
                f"{self.input_dir}"
            )

        images = sorted(
            path
            for path in self.input_dir.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower()
                in self._supported_extensions
            )
        )

        logger.info(
            "Discovered %d receipt images.",
            len(images)
        )

        return images

    # ========================================================
    # RECEIPT ID
    # ========================================================

    @staticmethod
    def _receipt_id(image_path: Path) -> str:
        """
        Generate deterministic receipt ID from filename/path.
        """

        return image_path.stem

    # ========================================================
    # LOAD IMAGE
    # ========================================================

    @staticmethod
    def _load_image(
        image_path: Path
    ) -> np.ndarray:
        """
        Load image using OpenCV.
        """

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR
        )

        if image is None:
            raise ValueError(
                f"OpenCV could not decode image: "
                f"{image_path}"
            )

        if image.size == 0:
            raise ValueError(
                f"Image is empty: {image_path}"
            )

        return image

    # ========================================================
    # RESIZE
    # ========================================================

    def _resize(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """
        Resize while preserving aspect ratio.

        Images smaller than target width are not enlarged
        unnecessarily unless required by OCR resolution.
        """

        height, width = image.shape[:2]

        if width <= self.config.target_width:
            return image

        scale = (
            self.config.target_width
            / float(width)
        )

        new_width = self.config.target_width

        new_height = max(
            1,
            int(round(height * scale))
        )

        return cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

    # ========================================================
    # GRAYSCALE
    # ========================================================

    @staticmethod
    def _grayscale(
        image: np.ndarray
    ) -> np.ndarray:
        """
        Convert BGR image to grayscale.
        """

        if len(image.shape) == 2:
            return image

        return cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    # ========================================================
    # DENOISING
    # ========================================================

    @staticmethod
    def _denoise(
        image: np.ndarray
    ) -> np.ndarray:
        """
        Non-local means denoising.

        Tuned conservatively to preserve receipt characters.
        """

        return cv2.fastNlMeansDenoising(
            image,
            None,
            h=10,
            templateWindowSize=7,
            searchWindowSize=21
        )

    # ========================================================
    # CLAHE
    # ========================================================

    @staticmethod
    def _enhance_contrast(
        image: np.ndarray
    ) -> np.ndarray:
        """
        Local contrast enhancement using CLAHE.
        """

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        return clahe.apply(image)

    # ========================================================
    # BORDER / FOREGROUND DETECTION
    # ========================================================

    @staticmethod
    def _find_document_contour(
        image: np.ndarray
    ) -> np.ndarray | None:
        """
        Attempt to identify the receipt/document boundary.

        Uses edge detection + contour approximation.
        """

        blurred = cv2.GaussianBlur(
            image,
            (5, 5),
            0
        )

        edges = cv2.Canny(
            blurred,
            50,
            150
        )

        kernel = np.ones(
            (5, 5),
            dtype=np.uint8
        )

        edges = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2
        )

        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        image_area = (
            image.shape[0]
            * image.shape[1]
        )

        candidates = []

        for contour in contours:

            area = cv2.contourArea(
                contour
            )

            if area < 0.10 * image_area:
                continue

            perimeter = cv2.arcLength(
                contour,
                True
            )

            if perimeter <= 0:
                continue

            approximation = cv2.approxPolyDP(
                contour,
                0.02 * perimeter,
                True
            )

            if len(approximation) == 4:

                candidates.append(
                    (
                        area,
                        approximation.reshape(
                            4, 2
                        )
                    )
                )

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: item[0],
            reverse=True
        )

        return candidates[0][1].astype(
            np.float32
        )

    # ========================================================
    # FOUR POINT ORDERING
    # ========================================================

    @staticmethod
    def _order_points(
        points: np.ndarray
    ) -> np.ndarray:
        """
        Order points as:

            top-left
            top-right
            bottom-right
            bottom-left
        """

        points = np.asarray(
            points,
            dtype=np.float32
        )

        if points.shape != (4, 2):
            raise ValueError(
                "Expected exactly four points."
            )

        ordered = np.zeros(
            (4, 2),
            dtype=np.float32
        )

        sums = points.sum(
            axis=1
        )

        differences = (
            points[:, 1]
            - points[:, 0]
        )

        ordered[0] = points[
            np.argmin(sums)
        ]

        ordered[2] = points[
            np.argmax(sums)
        ]

        ordered[1] = points[
            np.argmin(differences)
        ]

        ordered[3] = points[
            np.argmax(differences)
        ]

        return ordered

    # ========================================================
    # PERSPECTIVE CORRECTION
    # ========================================================

    def _perspective_correct(
        self,
        image: np.ndarray
    ) -> tuple[np.ndarray, bool]:
        """
        Correct receipt perspective if a reliable 4-point
        document contour is found.
        """

        if not self.config.perspective_correction:
            return image, False

        contour = self._find_document_contour(
            image
        )

        if contour is None:
            return image, False

        points = self._order_points(
            contour
        )

        top_left, top_right, bottom_right, bottom_left = points

        width_top = np.linalg.norm(
            top_right - top_left
        )

        width_bottom = np.linalg.norm(
            bottom_right - bottom_left
        )

        max_width = max(
            int(round(width_top)),
            int(round(width_bottom))
        )

        height_right = np.linalg.norm(
            bottom_right - top_right
        )

        height_left = np.linalg.norm(
            bottom_left - top_left
        )

        max_height = max(
            int(round(height_right)),
            int(round(height_left))
        )

        if max_width <= 0 or max_height <= 0:
            return image, False

        # Avoid pathological transformations.
        if (
            max_width > image.shape[1] * 3
            or max_height > image.shape[0] * 3
        ):
            return image, False

        destination = np.array(
            [
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1],
            ],
            dtype=np.float32
        )

        matrix = cv2.getPerspectiveTransform(
            points,
            destination
        )

        corrected = cv2.warpPerspective(
            image,
            matrix,
            (max_width, max_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        if corrected is None or corrected.size == 0:
            return image, False

        return corrected, True

    # ========================================================
    # DESKEW ANGLE
    # ========================================================

    @staticmethod
    def _estimate_skew_angle(
        image: np.ndarray
    ) -> float:
        """
        Estimate dominant text skew angle.

        Uses the minimum-area rectangle around foreground pixels.
        """

        if len(image.shape) != 2:
            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )
        else:
            gray = image

        # Threshold for text/foreground.
        _, binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV
            + cv2.THRESH_OTSU
        )

        coordinates = np.column_stack(
            np.where(binary > 0)
        )

        if coordinates.shape[0] < 100:
            return 0.0

        # OpenCV expects x,y.
        coordinates = coordinates[:, ::-1].astype(
            np.float32
        )

        rect = cv2.minAreaRect(
            coordinates
        )

        angle = float(
            rect[2]
        )

        # Normalize OpenCV's angle convention.
        if angle < -45:
            angle += 90

        if angle > 45:
            angle -= 90

        return angle

    # ========================================================
    # DESKEW
    # ========================================================

    def _deskew(
        self,
        image: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """
        Rotate image to correct text skew.
        """

        if not self.config.deskew:
            return image, 0.0

        angle = self._estimate_skew_angle(
            image
        )

        if abs(angle) > self.config.max_rotation_angle:
            logger.warning(
                "Estimated rotation %.2f° exceeds "
                "maximum %.2f°. Skipping deskew.",
                angle,
                self.config.max_rotation_angle
            )

            return image, 0.0

        # Tiny angles are usually estimation noise.
        if abs(angle) < 0.5:
            return image, 0.0

        height, width = image.shape[:2]

        center = (
            width / 2.0,
            height / 2.0
        )

        matrix = cv2.getRotationMatrix2D(
            center,
            angle,
            1.0
        )

        cos = abs(matrix[0, 0])
        sin = abs(matrix[0, 1])

        new_width = int(
            height * sin
            + width * cos
        )

        new_height = int(
            height * cos
            + width * sin
        )

        matrix[0, 2] += (
            new_width / 2
            - center[0]
        )

        matrix[1, 2] += (
            new_height / 2
            - center[1]
        )

        rotated = cv2.warpAffine(
            image,
            matrix,
            (new_width, new_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        return rotated, angle

    # ========================================================
    # ADAPTIVE THRESHOLD
    # ========================================================

    @staticmethod
    def _adaptive_threshold(
        image: np.ndarray
    ) -> np.ndarray:
        """
        Generate a binarized image for difficult lighting.

        This is optional because grayscale images often work
        better with modern OCR engines than hard thresholding.
        """

        return cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            15
        )

    # ========================================================
    # NORMALIZE ORIENTATION
    # ========================================================

    @staticmethod
    def _normalize_orientation(
        image: np.ndarray
    ) -> np.ndarray:
        """
        Apply lightweight automatic orientation correction.

        This method intentionally avoids aggressive 90-degree
        guessing without OCR/layout evidence.
        """

        return image

    # ========================================================
    # SAVE IMAGE
    # ========================================================

    def _save_image(
        self,
        image: np.ndarray,
        output_path: Path
    ) -> None:
        """
        Save processed image safely.
        """

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        extension = output_path.suffix.lower()

        params: list[int] = []

        if extension in (
            ".jpg",
            ".jpeg"
        ):

            params = [
                cv2.IMWRITE_JPEG_QUALITY,
                self.config.jpeg_quality
            ]

        success = cv2.imwrite(
            str(output_path),
            image,
            params
        )

        if not success:
            raise IOError(
                f"Failed to write image: "
                f"{output_path}"
            )

        if (not output_path.exists() or output_path.stat().st_size == 0):
            raise IOError(
                f"Output image was not created correctly: "
                f"{output_path}"
            )

    # ========================================================
    # INTERMEDIATE IMAGE SAVING
    # ========================================================

    def _save_intermediate(
        self,
        image: np.ndarray,
        receipt_id: str,
        stage: str
    ) -> None:
        """
        Save intermediate processing stage for debugging.
        """

        if not self.config.save_intermediate:
            return

        debug_dir = (
            self.output_dir
            / "_intermediate"
            / receipt_id
        )

        debug_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_path = (
            debug_dir
            / f"{stage}.png"
        )

        self._save_image(
            image,
            output_path
        )

    # ========================================================
    # PROCESS SINGLE IMAGE
    # ========================================================

    def process_image(
        self,
        image_path: str | Path
    ) -> PreprocessingResult:
        """
        Process one receipt image.
        """

        import time

        start_time = time.perf_counter()

        image_path = Path(
            image_path
        )

        receipt_id = self._receipt_id(
            image_path
        )

        errors: list[str] = []
        warnings: list[str] = []

        output_path: Path | None = None

        original_width = None
        original_height = None

        processed_width = None
        processed_height = None

        rotation_angle = 0.0
        perspective_corrected = False
        denoised = False
        contrast_enhanced = False

        try:

            # ------------------------------------------------
            # Validate input
            # ------------------------------------------------

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Input image does not exist: "
                    f"{image_path}"
                )

            if (
                image_path.suffix.lower()
                not in self._supported_extensions
            ):
                raise ValueError(
                    f"Unsupported image type: "
                    f"{image_path.suffix}"
                )

            # ------------------------------------------------
            # Load
            # ------------------------------------------------

            image = self._load_image(
                image_path
            )

            original_height, original_width = (
                image.shape[:2]
            )

            if (
                original_width
                < self.config.min_width
                or original_height
                < self.config.min_height
            ):

                warnings.append(
                    "Input image dimensions are below "
                    "the configured minimum."
                )

            self._save_intermediate(
                image,
                receipt_id,
                "01_original"
            )

            # ------------------------------------------------
            # Resize
            # ------------------------------------------------

            image = self._resize(
                image
            )

            self._save_intermediate(
                image,
                receipt_id,
                "02_resized"
            )

            # ------------------------------------------------
            # Perspective correction
            # ------------------------------------------------

            image, perspective_corrected = (
                self._perspective_correct(
                    image
                )
            )

            self._save_intermediate(
                image,
                receipt_id,
                "03_perspective"
            )

            # ------------------------------------------------
            # Grayscale
            # ------------------------------------------------

            if self.config.grayscale:

                image = self._grayscale(
                    image
                )

            self._save_intermediate(
                image,
                receipt_id,
                "04_grayscale"
            )

            # ------------------------------------------------
            # Denoising
            # ------------------------------------------------

            if self.config.denoise:

                image = self._denoise(
                    image
                )

                denoised = True

            self._save_intermediate(
                image,
                receipt_id,
                "05_denoised"
            )

            # ------------------------------------------------
            # CLAHE
            # ------------------------------------------------

            if self.config.clahe:

                image = self._enhance_contrast(
                    image
                )

                contrast_enhanced = True

            self._save_intermediate(
                image,
                receipt_id,
                "06_clahe"
            )

            # ------------------------------------------------
            # Deskew
            # ------------------------------------------------

            image, rotation_angle = (
                self._deskew(
                    image
                )
            )

            self._save_intermediate(
                image,
                receipt_id,
                "07_deskewed"
            )

            # ------------------------------------------------
            # Optional thresholding
            # ------------------------------------------------

            if self.config.adaptive_threshold:

                image = self._adaptive_threshold(
                    image
                )

            self._save_intermediate(
                image,
                receipt_id,
                "08_final"
            )

            # ------------------------------------------------
            # Orientation normalization
            # ------------------------------------------------

            image = self._normalize_orientation(
                image
            )

            # ------------------------------------------------
            # Final dimensions
            # ------------------------------------------------

            processed_height, processed_width = (
                image.shape[:2]
            )

            # ------------------------------------------------
            # Output path
            # ------------------------------------------------

            relative_path = image_path.relative_to(
                self.input_dir
            )

            output_path = (
                self.output_dir
                / relative_path.parent
                / f"{relative_path.stem}"
                f"{self.config.output_extension}"
            )

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            self._save_image(
                image,
                output_path
            )

            status = "success"

        except Exception as exc:

            logger.exception(
                "Preprocessing failed for %s",
                image_path
            )

            errors.append(
                str(exc)
            )

            status = "failed"

        processing_time_ms = (
            time.perf_counter()
            - start_time
        ) * 1000

        return PreprocessingResult(
            receipt_id=receipt_id,
            input_path=str(
                image_path
            ),
            output_path=(
                str(output_path)
                if output_path is not None
                and output_path.exists()
                else None
            ),
            status=status,
            original_width=original_width,
            original_height=original_height,
            processed_width=processed_width,
            processed_height=processed_height,
            rotation_angle=float(
                rotation_angle
            ),
            perspective_corrected=(
                perspective_corrected
            ),
            denoised=denoised,
            contrast_enhanced=(
                contrast_enhanced
            ),
            grayscale=self.config.grayscale,
            errors=errors,
            warnings=warnings,
            processing_time_ms=float(
                processing_time_ms
            ),
            timestamp_utc=datetime.now(
                timezone.utc
            ).isoformat(),
        )

    # ========================================================
    # BATCH PROCESSING
    # ========================================================

    def process_batch(
        self
    ) -> dict[str, Any]:
        """
        Process the entire receipt dataset.
        """

        logger.info(
            "Starting batch image preprocessing..."
        )

        images = self._discover_images()

        results: list[PreprocessingResult] = []

        for index, image_path in enumerate(
            images,
            start=1
        ):

            result = self.process_image(
                image_path
            )

            results.append(
                result
            )

            if index % 100 == 0:

                logger.info(
                    "Processed %d/%d images.",
                    index,
                    len(images)
                )

        successful = sum(
            result.status == "success"
            for result in results
        )

        failed = sum(
            result.status == "failed"
            for result in results
        )

        processing_times = [
            result.processing_time_ms
            for result in results
        ]

        summary = {
            "timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat(),

            "input_directory": str(
                self.input_dir.resolve()
            ),

            "output_directory": str(
                self.output_dir.resolve()
            ),

            "total_images": len(results),

            "successful": successful,

            "failed": failed,

            "success_rate": (
                successful / len(results)
                if results
                else 0.0
            ),

            "average_processing_time_ms": (
                float(np.mean(processing_times))
                if processing_times
                else 0.0
            ),

            "total_processing_time_seconds": (
                float(
                    sum(processing_times)
                    / 1000
                )
            ),

            "results": [
                asdict(result)
                for result in results
            ],
        }

        # ----------------------------------------------------
        # Save manifest
        # ----------------------------------------------------

        manifest_path = (
            self.output_dir
            / "preprocessing_manifest.json"
        )

        with manifest_path.open("w", encoding="utf-8") as file:

            json.dump(
                summary,
                file,
                indent=2,
                ensure_ascii=False
            )

        logger.info(
            "Preprocessing manifest saved to: %s",
            manifest_path
        )

        logger.info(
            "Batch preprocessing completed. "
            "Success=%d, Failed=%d",
            successful,
            failed
        )

        summary["manifest_path"] = str(
            manifest_path
        )

        return summary

    # ========================================================
    # SINGLE IMAGE ENTRY POINT
    # ========================================================

    def run(
        self,
        image_path: str | Path | None = None
    ) -> PreprocessingResult | dict[str, Any]:
        """
        Run preprocessing.

        If image_path is supplied:
            Process a single image.

        Otherwise:
            Process the complete dataset.
        """

        if image_path is not None:

            return self.process_image(
                image_path
            )

        return self.process_batch()