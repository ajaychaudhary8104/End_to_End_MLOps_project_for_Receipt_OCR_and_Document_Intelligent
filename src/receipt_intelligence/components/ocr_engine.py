from __future__ import annotations
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
from src.receipt_intelligence.entity.config_entity import OCREngineConfig
from src.receipt_intelligence import logger


# ============================================================
# OCR TEXT REGION
# ============================================================

@dataclass
class OCRTextRegion:
    """
    Normalized OCR result for one detected text region.
    """

    text: str

    confidence: float

    bbox: list[list[float]]

    x_min: float

    y_min: float

    x_max: float

    y_max: float

    width: float

    height: float


# ============================================================
# OCR IMAGE RESULT
# ============================================================

@dataclass
class OCRImageResult:
    """
    OCR result for one receipt image.
    """

    receipt_id: str

    input_path: str

    status: str

    full_text: str

    regions: list[OCRTextRegion]

    mean_confidence: float

    min_confidence: float

    text_region_count: int

    processing_time_ms: float

    model_name: str

    language: str

    device: str

    errors: list[str]

    warnings: list[str]

    timestamp_utc: str


# ============================================================
# OCR ENGINE
# ============================================================

class OCREngine:
    """
    Production-grade PaddleOCR inference engine.

    Pipeline:

        Preprocessed Receipt
                ↓
        PaddleOCR
                ↓
        Text Detection
                ↓
        Text Recognition
                ↓
        Confidence Filtering
                ↓
        Bounding Box Normalization
                ↓
        Text Ordering
                ↓
        Stable JSON Artifact
    """

    def __init__(self, config: OCREngineConfig):
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
            ext.lower()
            for ext in config.supported_extensions
        }

        # ----------------------------------------------------
        # Validate configuration
        # ----------------------------------------------------

        if not 0.0 <= config.text_detection_score_threshold <= 1.0:
            raise ValueError(
                "text_detection_score_threshold "
                "must be between 0 and 1."
            )

        if not 0.0 <= config.text_recognition_score_threshold <= 1.0:
            raise ValueError(
                "text_recognition_score_threshold "
                "must be between 0 and 1."
            )

        if not config.language:
            raise ValueError(
                "language cannot be empty."
            )

        self.ocr = self._initialize_paddleocr()

    # ========================================================
    # INITIALIZE PADDLEOCR
    # ========================================================

    def _initialize_paddleocr(self):
        """
        Initialize PaddleOCR.

        Supports the current PaddleOCR 3.x style API.
        """

        try:

            from paddleocr import PaddleOCR

        except ImportError as exc:

            raise ImportError(
                "PaddleOCR is not installed. "
                "Install it using your project's "
                "PaddleOCR/PaddlePaddle dependencies."
            ) from exc

        logger.info(
            "Initializing PaddleOCR..."
        )

        logger.info(
            "Model: %s",
            self.config.model_name
        )

        logger.info(
            "Language: %s",
            self.config.language
        )

        logger.info(
            "Device: %s",
            self.config.device
        )

        # ----------------------------------------------------
        # Current PaddleOCR 3.x interface
        # ----------------------------------------------------

        try:

            ocr = PaddleOCR(
                lang=self.config.language,
                device=self.config.device,

                # Keep document-level transformations disabled
                # because our preprocessing layer already handles
                # perspective/deskew operations.
                use_doc_orientation_classify=(
                    self.config.use_doc_orientation_classify
                ),

                use_doc_unwarping=(
                    self.config.use_doc_unwarping
                ),

                use_textline_orientation=(
                    self.config.use_textline_orientation
                ),
                use_angle_cls=True, 
            )

            logger.info(
                "PaddleOCR initialized successfully."
            )

            return ocr

        except TypeError:

            # ------------------------------------------------
            # Compatibility fallback for installations where
            # some newer arguments are unavailable.
            # ------------------------------------------------

            logger.warning(
                "PaddleOCR initialization with extended "
                "configuration failed. Falling back to "
                "minimal configuration."
            )

            try:

                ocr = PaddleOCR(
                    lang=self.config.language,
                    device=self.config.device
                )

                logger.info(
                    "PaddleOCR fallback initialization succeeded."
                )

                return ocr

            except Exception as exc:

                logger.exception(
                    "PaddleOCR initialization failed."
                )

                raise RuntimeError(
                    "Unable to initialize PaddleOCR."
                ) from exc

        except Exception as exc:

            logger.exception(
                "PaddleOCR initialization failed."
            )

            raise RuntimeError(
                "Unable to initialize PaddleOCR."
            ) from exc

    # ========================================================
    # FILE DISCOVERY
    # ========================================================

    def _discover_images(self) -> list[Path]:
        """
        Discover preprocessed receipt images.
        """

        if not self.input_dir.exists():
            raise FileNotFoundError(
                f"OCR input directory does not exist: "
                f"{self.input_dir}"
            )

        if not self.input_dir.is_dir():
            raise NotADirectoryError(
                f"OCR input path is not a directory: "
                f"{self.input_dir}"
            )

        if self.config.recursive:

            images = sorted(
                path
                for path in self.input_dir.rglob("*")
                if (
                    path.is_file()
                    and path.suffix.lower()
                    in self._supported_extensions
                )
            )

        else:

            images = sorted(
                path
                for path in self.input_dir.iterdir()
                if (
                    path.is_file()
                    and path.suffix.lower()
                    in self._supported_extensions
                )
            )

        logger.info(
            "Discovered %d images for OCR.",
            len(images)
        )

        return images

    # ========================================================
    # RECEIPT ID
    # ========================================================

    @staticmethod
    def _receipt_id(
        image_path: Path
    ) -> str:
        """
        Stable receipt ID based on filename.
        """

        return image_path.stem

    # ========================================================
    # FLOAT CONVERSION
    # ========================================================

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        """
        Convert arbitrary numeric values to float safely.
        """

        try:

            if value is None:
                return None

            numeric = float(value)

            if not np.isfinite(numeric):
                return None

            return numeric

        except (
            TypeError,
            ValueError
        ):

            return None

    # ========================================================
    # RECURSIVE OBJECT CONVERSION
    # ========================================================

    @staticmethod
    def _to_python(
        value: Any
    ) -> Any:
        """
        Convert PaddleOCR/numpy objects into standard
        Python objects recursively.
        """

        if isinstance(
            value,
            np.ndarray
        ):

            return value.tolist()

        if isinstance(
            value,
            np.generic
        ):

            return value.item()

        if isinstance(
            value,
            dict
        ):

            return {
                str(key): OCREngine._to_python(
                    val
                )
                for key, val in value.items()
            }

        if isinstance(
            value,
            (list, tuple)
        ):

            return [
                OCREngine._to_python(
                    item
                )
                for item in value
            ]

        return value

    # ========================================================
    # EXTRACT RAW RESULT DICTIONARY
    # ========================================================

    def _extract_result_dict(
        self,
        result: Any
    ) -> dict[str, Any]:
        """
        Convert a PaddleOCR result object into a regular
        Python dictionary.
        """

        # ----------------------------------------------------
        # Result objects in PaddleOCR 3.x commonly expose
        # a .json property/method or .res payload.
        # ----------------------------------------------------

        # Try .json attribute
        try:

            json_value = getattr(
                result,
                "json",
                None
            )

            if json_value is not None:

                if callable(json_value):
                    json_value = json_value()

                json_value = self._to_python(
                    json_value
                )

                if isinstance(
                    json_value,
                    dict
                ):

                    return json_value

        except Exception:
            pass

        # Try .res
        try:

            res_value = getattr(
                result,
                "res",
                None
            )

            if res_value is not None:

                res_value = self._to_python(
                    res_value
                )

                if isinstance(
                    res_value,
                    dict
                ):

                    return res_value

        except Exception:
            pass

        # Try __dict__
        try:

            raw_dict = getattr(
                result,
                "__dict__",
                {}
            )

            raw_dict = self._to_python(
                raw_dict
            )

            if isinstance(
                raw_dict,
                dict
            ):

                return raw_dict

        except Exception:
            pass

        return {}
    def _parse_paddle_result(
        self,
        result: Any
    ) -> list[OCRTextRegion]:
        """
        Parse PaddleOCR output into stable OCRTextRegion objects.

        Supports:

        1. Classic PaddleOCR output:
        [
            [
                [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],
                ("TEXT", score)
            ],
            ...
        ]

        2. PaddleOCR 3.x dictionary outputs:
        {
            "dt_polys": ...,
            "rec_texts": ...,
            "rec_scores": ...
        }
        """

        # ============================================================
        # CLASSIC PADDLEOCR FORMAT
        # ============================================================

        if isinstance(result, list):

            classic_regions = []

            is_classic_format = False

            for item in result:

                if not isinstance(item, (list, tuple)):
                    continue

                if len(item) < 2:
                    continue

                bbox = item[0]
                text_info = item[1]

                if (
                    isinstance(text_info, (list, tuple))
                    and len(text_info) >= 2
                ):
                    is_classic_format = True

                    text = text_info[0]
                    score = text_info[1]

                    region = self._build_region(
                        text=text,
                        confidence=score,
                        bbox=bbox
                    )

                    if region is not None:
                        classic_regions.append(region)

            if is_classic_format:
                return classic_regions

        # ============================================================
        # EXISTING PADDLEOCR 3.x PARSER
        # ============================================================

        data = self._extract_result_dict(result)

        if (
            "res" in data
            and isinstance(data["res"], dict)
        ):
            data = data["res"]

        polygons = (
            data.get("dt_polys")
            or data.get("rec_polys")
            or data.get("text_region")
            or data.get("text_regions")
            or []
        )

        texts = (
            data.get("rec_texts")
            or data.get("texts")
            or data.get("text")
            or []
        )

        scores = (
            data.get("rec_scores")
            or data.get("scores")
            or data.get("text_scores")
            or []
        )

        polygons = self._to_python(polygons)
        texts = self._to_python(texts)
        scores = self._to_python(scores)

        if texts is None:
            texts = []

        if scores is None:
            scores = []

        if polygons is None:
            polygons = []

        if isinstance(data.get("ocr"), list):

            regions = []

            for item in data["ocr"]:

                if not isinstance(item, dict):
                    continue

                text = (
                    item.get("text")
                    or item.get("rec_text")
                    or ""
                )

                confidence = (
                    item.get("confidence")
                    or item.get("score")
                    or item.get("rec_score")
                    or 0.0
                )

                bbox = (
                    item.get("bbox")
                    or item.get("polygon")
                    or item.get("points")
                    or []
                )

                region = self._build_region(
                    text=text,
                    confidence=confidence,
                    bbox=bbox
                )

                if region is not None:
                    regions.append(region)

            return regions

        if not isinstance(texts, list):
            texts = [texts]

        if not isinstance(scores, list):
            scores = [scores]

        if not isinstance(polygons, list):
            polygons = [polygons]

        region_count = max(
            len(texts),
            len(polygons),
            len(scores)
        )

        regions = []

        for index in range(region_count):

            text = (
                texts[index]
                if index < len(texts)
                else ""
            )

            bbox = (
                polygons[index]
                if index < len(polygons)
                else []
            )

            score = (
                scores[index]
                if index < len(scores)
                else 0.0
            )

            region = self._build_region(
                text=text,
                confidence=score,
                bbox=bbox
            )

            if region is not None:
                regions.append(region)

        return regions

    # ========================================================
    # REGION BUILDER
    # ========================================================

    def _build_region(
        self,
        text: Any,
        confidence: Any,
        bbox: Any
    ) -> OCRTextRegion | None:
        """
        Create normalized OCR region.
        """

        if text is None:
            text = ""

        text = str(
            text
        ).strip()

        score = self._safe_float(
            confidence
        )

        if score is None:
            score = 0.0

        # ----------------------------------------------------
        # Normalize polygon
        # ----------------------------------------------------

        bbox = self._to_python(
            bbox
        )

        if bbox is None:
            bbox = []

        # Expected:
        #
        # [[x1,y1],
        #  [x2,y2],
        #  [x3,y3],
        #  [x4,y4]]
        #
        # Also tolerate flat:
        # [x1,y1,x2,y2,...]
        # ----------------------------------------------------

        points = []

        try:

            if (
                isinstance(bbox, list)
                and bbox
                and isinstance(
                    bbox[0],
                    (int, float)
                )
            ):

                if len(bbox) >= 4:

                    flat = bbox

                    for i in range(
                        0,
                        len(flat) - 1,
                        2
                    ):

                        x = self._safe_float(
                            flat[i]
                        )

                        y = self._safe_float(
                            flat[i + 1]
                        )

                        if (
                            x is not None
                            and y is not None
                        ):

                            points.append(
                                [x, y]
                            )

            else:

                for point in bbox:

                    if not isinstance(
                        point,
                        (list, tuple)
                    ):
                        continue

                    if len(point) < 2:
                        continue

                    x = self._safe_float(
                        point[0]
                    )

                    y = self._safe_float(
                        point[1]
                    )

                    if (
                        x is not None
                        and y is not None
                    ):

                        points.append(
                            [x, y]
                        )

        except Exception:

            points = []

        # ----------------------------------------------------
        # No geometry: still allow text result.
        # ----------------------------------------------------

        if points:

            x_values = [
                point[0]
                for point in points
            ]

            y_values = [
                point[1]
                for point in points
            ]

            x_min = float(
                min(x_values)
            )

            y_min = float(
                min(y_values)
            )

            x_max = float(
                max(x_values)
            )

            y_max = float(
                max(y_values)
            )

        else:

            x_min = 0.0
            y_min = 0.0
            x_max = 0.0
            y_max = 0.0

        width = max(
            0.0,
            x_max - x_min
        )

        height = max(
            0.0,
            y_max - y_min
        )

        # ----------------------------------------------------
        # Recognition score filtering
        # ----------------------------------------------------

        if (
            score
            < self.config.text_recognition_score_threshold
        ):

            return None

        # Ignore completely blank OCR strings.
        if not text:

            return None

        return OCRTextRegion(
            text=text,
            confidence=float(score),
            bbox=points,
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
            width=width,
            height=height,
        )

    # ========================================================
    # TEXT ORDERING
    # ========================================================

    @staticmethod
    def _sort_regions(
        regions: list[OCRTextRegion]
    ) -> list[OCRTextRegion]:
        """
        Sort OCR regions approximately in reading order.

        Primary:
            top-to-bottom

        Secondary:
            left-to-right
        """

        if not regions:
            return []

        # Estimate line tolerance dynamically.
        median_height = float(
            np.median(
                [
                    region.height
                    for region in regions
                    if region.height > 0
                ]
            )
        ) if any(
            region.height > 0
            for region in regions
        ) else 20.0

        line_tolerance = max(
            10.0,
            median_height * 0.5
        )

        def sort_key(
            region: OCRTextRegion
        ):
            line_number = int(
                round(
                    region.y_min
                    / line_tolerance
                )
            )

            return (
                line_number,
                region.x_min
            )

        return sorted(
            regions,
            key=sort_key
        )

    # ========================================================
    # FULL TEXT
    # ========================================================

    @staticmethod
    def _build_full_text(
        regions: list[OCRTextRegion]
    ) -> str:
        """
        Combine OCR regions into reading-order text.
        """

        return "\n".join(
            region.text
            for region in regions
            if region.text.strip()
        ).strip()

    # ========================================================
    # SAVE OCR JSON
    # ========================================================

    def _save_result(
        self,
        result: OCRImageResult
    ) -> Path:
        """
        Persist normalized OCR output.
        """

        output_path = (
            self.output_dir
            / f"{result.receipt_id}.json"
        )

        payload = asdict(
            result
        )

        with output_path.open(
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                payload,
                file,
                indent=2,
                ensure_ascii=False
            )

        return output_path

    # ========================================================
    # PROCESS SINGLE IMAGE
    # ========================================================

    def process_image(
        self,
        image_path: str | Path
    ) -> OCRImageResult:
        """
        Run OCR on one preprocessed receipt.
        """

        start = time.perf_counter()

        image_path = Path(
            image_path
        )

        receipt_id = self._receipt_id(
            image_path
        )

        errors: list[str] = []
        warnings: list[str] = []

        try:

            if not image_path.exists():

                raise FileNotFoundError(
                    f"OCR input image not found: "
                    f"{image_path}"
                )

            if (
                image_path.suffix.lower()
                not in self._supported_extensions
            ):

                raise ValueError(
                    f"Unsupported OCR image extension: "
                    f"{image_path.suffix}"
                )

            logger.info(
                "Running OCR: %s",
                image_path
            )

            # ------------------------------------------------
            # PaddleOCR inference
            # ------------------------------------------------

            predictions = self.ocr.ocr(str(image_path), cls=True)

            # PaddleOCR returns an iterable/list of results.
            prediction_list = list(
                predictions
            )
            print(prediction_list)

            if not prediction_list:

                warnings.append(
                    "PaddleOCR returned no prediction result."
                )

                regions = []

            else:

                # One image -> normally one result.
                # If multiple result objects appear, combine them.
                regions = []

                for prediction in prediction_list:

                    parsed_regions = (
                        self._parse_paddle_result(
                            prediction
                        )
                    )

                    regions.extend(
                        parsed_regions
                    )

            # ------------------------------------------------
            # Sort regions
            # ------------------------------------------------

            regions = self._sort_regions(
                regions
            )

            # ------------------------------------------------
            # Detection threshold
            #
            # Recognition threshold is already applied.
            # We additionally protect against invalid scores.
            # ------------------------------------------------

            regions = [
                region
                for region in regions
                if (
                    0.0
                    <= region.confidence
                    <= 1.0
                    and region.confidence
                    >= self.config.text_detection_score_threshold
                )
            ]

            # ------------------------------------------------
            # Full OCR text
            # ------------------------------------------------

            full_text = self._build_full_text(
                regions
            )

            # ------------------------------------------------
            # Confidence statistics
            # ------------------------------------------------

            confidences = [
                region.confidence
                for region in regions
            ]

            mean_confidence = (
                float(np.mean(confidences))
                if confidences
                else 0.0
            )

            min_confidence = (
                float(np.min(confidences))
                if confidences
                else 0.0
            )

            # ------------------------------------------------
            # Warning conditions
            # ------------------------------------------------

            if not regions:

                warnings.append(
                    "No text regions survived OCR confidence filtering."
                )

            if mean_confidence < 0.60 and regions:

                warnings.append(
                    "Mean OCR confidence is below 0.60."
                )

            status = "success"

        except Exception as exc:

            logger.exception(
                "OCR failed for %s",
                image_path
            )

            errors.append(
                str(exc)
            )

            full_text = ""
            regions = []
            mean_confidence = 0.0
            min_confidence = 0.0
            status = "failed"

        processing_time_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        result = OCRImageResult(
            receipt_id=receipt_id,
            input_path=str(
                image_path
            ),
            status=status,
            full_text=full_text,
            regions=regions,
            mean_confidence=float(
                mean_confidence
            ),
            min_confidence=float(
                min_confidence
            ),
            text_region_count=len(
                regions
            ),
            processing_time_ms=float(
                processing_time_ms
            ),
            model_name=self.config.model_name,
            language=self.config.language,
            device=self.config.device,
            errors=errors,
            warnings=warnings,
            timestamp_utc=datetime.now(
                timezone.utc
            ).isoformat(),
        )

        # ----------------------------------------------------
        # Persist normalized result
        # ----------------------------------------------------

        try:

            output_path = self._save_result(
                result
            )

            logger.info(
                "OCR result saved: %s",
                output_path
            )

        except Exception as exc:

            logger.exception(
                "Failed to save OCR result for %s",
                image_path
            )

            result.errors.append(
                f"OCR result save failed: {exc}"
            )

            result.status = "failed"

        return result

    # ========================================================
    # BATCH OCR
    # ========================================================

    def process_batch(self) -> dict[str, Any]:
        """
        Run OCR over the complete preprocessed dataset.
        """

        logger.info(
            "Starting batch OCR..."
        )

        images = self._discover_images()

        results: list[OCRImageResult] = []

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

            if index % 50 == 0:

                logger.info(
                    "OCR processed %d/%d images.",
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

        all_confidences = [
            result.mean_confidence
            for result in results
            if (
                result.status == "success"
                and result.text_region_count > 0
            )
        ]

        total_regions = sum(
            result.text_region_count
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

            "model_name": self.config.model_name,

            "language": self.config.language,

            "device": self.config.device,

            "total_images": len(results),

            "successful": successful,

            "failed": failed,

            "success_rate": (
                successful / len(results)
                if results
                else 0.0
            ),

            "total_text_regions": total_regions,

            "average_mean_ocr_confidence": (
                float(np.mean(all_confidences))
                if all_confidences
                else 0.0
            ),

            "average_processing_time_ms": (
                float(np.mean(processing_times))
                if processing_times
                else 0.0
            ),

            "total_processing_time_seconds": (
                float(
                    sum(processing_times) / 1000.0
                )
            ),

            "results": [
                asdict(result)
                for result in results
            ],
        }

        # ----------------------------------------------------
        # Save batch manifest
        # ----------------------------------------------------

        manifest_path = (
            self.output_dir
            / "ocr_manifest.json"
        )

        with manifest_path.open(
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                summary,
                file,
                indent=2,
                ensure_ascii=False
            )

        logger.info(
            "OCR manifest saved: %s",
            manifest_path
        )

        summary["manifest_path"] = str(
            manifest_path
        )

        logger.info(
            "Batch OCR completed. "
            "Success=%d, Failed=%d",
            successful,
            failed
        )

        return summary

    # ========================================================
    # PUBLIC ENTRY POINT
    # ========================================================

    def run(
        self,
        image_path: str | Path | None = None
    ) -> OCRImageResult | dict[str, Any]:
        """
        Run OCR on one image or the full dataset.
        """

        if image_path is not None:

            return self.process_image(
                image_path
            )

        return self.process_batch()