from __future__ import annotations

import json
import shutil
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from typing import Any, Iterable
from src.receipt_intelligence.config.configuration import ConfigurationManager
from src.receipt_intelligence.components.image_preprocessing import ImagePreprocessor
from src.receipt_intelligence.components.ocr_engine import OCREngine
from src.receipt_intelligence.components.ocr_text_cleaning import OCRTextCleaner
from src.receipt_intelligence.components.field_extraction_engine import FieldExtractor
from src.receipt_intelligence.components.confindence_and_reliability_engine import ConfidenceEngine
from src.receipt_intelligence.components.json_generation import JSONGenerator
from src.receipt_intelligence.entity.config_entity import InferenceConfig
from src.receipt_intelligence import logger


@dataclass(frozen=True)
class InferenceResult:
    status: str
    receipt_id: str
    input_path: str
    processing_time_ms: float
    prediction: dict[str, Any] | None
    overall_confidence: float | None
    reliability: str | None
    review_required: bool
    error: str | None
    timestamp_utc: str


class ReceiptInferencePipeline:
    """
    Production-grade inference pipeline.

    Flow:

        Input Receipt Image
                ↓
        Image Preprocessing
                ↓
        OCR
                ↓
        OCR Text Cleaning
                ↓
        Field Extraction
                ↓
        Confidence & Reliability
                ↓
        JSON Generation
                ↓
        Final Prediction
    """

    def __init__(
        self,
        config: ConfigurationManager,
        inference_config:InferenceConfig,
        *,
        mlflow_tracker: Any | None = None,
    ):
        self.config = config
        self.inference_config = inference_config
        self.mlflow_tracker = mlflow_tracker

        self._validate_configuration()
        self._initialize_directories()


    # =========================================================
    # VALIDATION
    # =========================================================

    def _validate_configuration(self) -> None:
        config = self.inference_config

        if not str(config.root_dir).strip():
            raise ValueError(
                "inference.root_dir cannot be empty."
            )

        if not str(config.input_dir).strip():
            raise ValueError(
                "inference.input_dir cannot be empty."
            )

        if not str(config.output_dir).strip():
            raise ValueError(
                "inference.output_dir cannot be empty."
            )

        if not str(
            config.prediction_filename
        ).strip():
            raise ValueError(
                "inference.prediction_filename cannot be empty."
            )

        if not config.prediction_filename.lower().endswith(
            ".json"
        ):
            raise ValueError(
                "inference.prediction_filename must end with .json."
            )

        if not str(
            config.mlflow_run_name
        ).strip():
            raise ValueError(
                "inference.mlflow_run_name cannot be empty."
            )

    def _initialize_directories(self) -> None:
        config = self.inference_config

        if config.create_output_directory:
            Path(
                config.root_dir
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

            Path(
                config.output_dir
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

    # =========================================================
    # GENERAL HELPERS
    # =========================================================

    @staticmethod
    def _utc_timestamp() -> str:
        from datetime import datetime, timezone

        return datetime.now(
            timezone.utc
        ).isoformat()

    @staticmethod
    def _to_dict(
        value: Any,
    ) -> Any:
        if value is None:
            return None

        if is_dataclass(value):
            return asdict(value)

        if isinstance(value, dict):
            return {
                str(key): ReceiptInferencePipeline._to_dict(
                    item
                )
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple)):
            return [
                ReceiptInferencePipeline._to_dict(
                    item
                )
                for item in value
            ]

        if isinstance(value, Path):
            return str(value)

        if hasattr(value, "tolist"):
            try:
                return value.tolist()
            except Exception:
                pass

        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass

        return value

    @staticmethod
    def _safe_receipt_id(
        image_path: Path,
    ) -> str:
        stem = image_path.stem.strip()

        if not stem:
            return (
                f"receipt_{uuid.uuid4().hex[:12]}"
            )

        value = "".join(
            character
            if character.isalnum()
            or character in ("_", "-")
            else "_"
            for character in stem
        )

        return value[:200]

    def _validate_input_image(
        self,
        image_path: Path,
    ) -> None:
        if not image_path.exists():
            raise FileNotFoundError(
                f"Receipt image does not exist: "
                f"{image_path}"
            )

        if not image_path.is_file():
            raise ValueError(
                f"Receipt input is not a file: "
                f"{image_path}"
            )

        if (
            image_path.suffix.lower()
            not in self.inference_config.supported_extensions
        ):
            raise ValueError(
                f"Unsupported receipt image extension: "
                f"{image_path.suffix}"
            )

        if image_path.stat().st_size <= 0:
            raise ValueError(
                f"Receipt image is empty: "
                f"{image_path}"
            )

    # =========================================================
    # CONFIG CLONING
    # =========================================================

    @staticmethod
    def _replace_config(
        config: Any,
        **changes: Any,
    ) -> Any:
        if is_dataclass(config):
            valid_fields = {
                field.name
                for field in config.__dataclass_fields__.values()
            }

            filtered_changes = {
                key: value
                for key, value in changes.items()
                if key in valid_fields
            }

            if filtered_changes:
                return replace(
                    config,
                    **filtered_changes,
                )

        return config

    # =========================================================
    # STAGE CONFIGURATION
    # =========================================================

    def _build_workspace(
        self,
    ) -> tempfile.TemporaryDirectory:
        return tempfile.TemporaryDirectory(
            prefix="receipt_inference_"
        )

    def _prepare_workspace(
        self,
        workspace: Path,
        image_path: Path,
    ) -> dict[str, Path]:
        input_dir = (
            workspace / "input"
        )
        preprocessing_dir = (
            workspace / "preprocessed"
        )
        ocr_dir = (
            workspace / "ocr"
        )
        cleaning_dir = (
            workspace / "cleaning"
        )
        extraction_dir = (
            workspace / "extracted"
        )
        confidence_dir = (
            workspace / "confidence"
        )
        json_dir = (
            workspace / "json"
        )

        for directory in (
            input_dir,
            preprocessing_dir,
            ocr_dir,
            cleaning_dir,
            extraction_dir,
            confidence_dir,
            json_dir,
        ):
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )

        staged_input = (
            input_dir / image_path.name
        )

        shutil.copy2(
            image_path,
            staged_input,
        )

        return {
            "input": input_dir,
            "preprocessed": preprocessing_dir,
            "ocr": ocr_dir,
            "cleaning": cleaning_dir,
            "extracted": extraction_dir,
            "confidence": confidence_dir,
            "json": json_dir,
            "staged_input": staged_input,
        }

    # =========================================================
    # COMPONENT BUILDERS
    # =========================================================

    def _create_preprocessing_engine(
        self,
        paths: dict[str, Path],
    ) -> Any:
        config = self.config.get_image_preprocessing_config()

        stage_config = self._replace_config(
            config,
            input_dir=str(
                paths["input"]
            ),
            output_dir=str(
                paths["preprocessed"]
            ),
        )

        return ImagePreprocessor(
            config=stage_config
        )

    def _create_ocr_engine(
        self,
        paths: dict[str, Path],
    ) -> Any:
        config = self.config.get_ocr_engine_config()

        stage_config = self._replace_config(
            config,
            input_dir=str(
                paths["preprocessed"]
            ),
            output_dir=str(
                paths["ocr"]
            ),
        )

        return OCREngine(
            config=stage_config
        )

    def _create_cleaning_engine(
        self,
        paths: dict[str, Path],
    ) -> Any:
        config = self.config.get_ocr_text_cleaning_config()

        stage_config = self._replace_config(
            config,
            input_dir=str(
                paths["ocr"]
            ),
            output_dir=str(
                paths["cleaning"]
            ),
        )

        return OCRTextCleaner(
            config=stage_config
        )

    def _create_extraction_engine(
        self,
        paths: dict[str, Path],
    ) -> Any:
        config = self.config.get_field_extraction_engine_config()

        stage_config = self._replace_config(
            config,
            input_dir=str(
                paths["cleaning"]
            ),
            output_dir=str(
                paths["extracted"]
            ),
        )

        return FieldExtractor(
            config=stage_config
        )

    def _create_confidence_engine(
        self,
        paths: dict[str, Path],
    ) -> Any:
        config = self.config.get_confidence_and_reliability_engine_config()

        stage_config = self._replace_config(
            config,
            input_dir=str(
                paths["extracted"]
            ),
            ocr_dir=str(
                paths["ocr"]
            ),
            cleaned_ocr_dir=str(
                paths["cleaning"]
            ),
            output_dir=str(
                paths["confidence"]
            ),
        )

        return ConfidenceEngine(
            config=stage_config
        )

    def _create_json_generator(
        self,
        paths: dict[str, Path],
    ) -> Any:
        config = self.config.get_json_generator_config()

        stage_config = self._replace_config(
            config,
            input_dir=str(
                paths["confidence"]
            ),
            output_dir=str(
                paths["json"]
            ),
        )

        return JSONGenerator(
            config=stage_config
        )

    # =========================================================
    # STAGE EXECUTION
    # =========================================================

    @staticmethod
    def _execute_engine(
        engine: Any,
        *,
        input_value: Any = None,
    ) -> Any:
        if not hasattr(
            engine,
            "run",
        ):
            raise AttributeError(
                f"{type(engine).__name__} "
                "does not expose a run() method."
            )

        if input_value is not None:
            try:
                return engine.run(
                    input_value
                )
            except TypeError:
                return engine.run()

        return engine.run()
    
    def _run_preprocessing(
        self,
        paths: dict[str, Path],
    ) -> Any:

        engine = (
            self._create_preprocessing_engine(
                paths
            )
        )

        return self._execute_engine(
            engine,
            input_value=paths["staged_input"],
        )

    def _run_ocr(
        self,
        paths: dict[str, Path],
    ) -> Any:
        engine = self._create_ocr_engine(
            paths
        )

        return self._execute_engine(
            engine
        )

    def _run_cleaning(
        self,
        paths: dict[str, Path],
    ) -> Any:
        engine = self._create_cleaning_engine(
            paths
        )

        return self._execute_engine(
            engine
        )

    def _run_extraction(
        self,
        paths: dict[str, Path],
    ) -> Any:
        engine = self._create_extraction_engine(
            paths
        )

        return self._execute_engine(
            engine
        )

    def _run_confidence(
        self,
        paths: dict[str, Path],
    ) -> Any:
        engine = self._create_confidence_engine(
            paths
        )

        return self._execute_engine(
            engine
        )

    def _run_json_generation(
        self,
        paths: dict[str, Path],
    ) -> Any:
        engine = self._create_json_generator(
            paths
        )

        return self._execute_engine(
            engine
        )

    # =========================================================
    # FINAL JSON DISCOVERY
    # =========================================================

    @staticmethod
    def _discover_json_files(
        directory: Path,
    ) -> list[Path]:
        if not directory.exists():
            return []

        return sorted(
            [
                path
                for path in directory.rglob("*.json")
                if path.is_file()
            ],
            key=lambda path: path.as_posix().lower(),
        )

    def _load_final_prediction(
        self,
        json_directory: Path,
        receipt_id: str,
    ) -> dict[str, Any]:
        files = self._discover_json_files(
            json_directory
        )

        if not files:
            raise FileNotFoundError(
                "JSON generation completed but "
                "no prediction JSON was produced."
            )

        preferred_names = (
            f"{receipt_id}.json",
            "prediction.json",
            "result.json",
        )

        selected_file = None

        for preferred_name in preferred_names:
            for file_path in files:
                if (
                    file_path.name.lower()
                    == preferred_name.lower()
                ):
                    selected_file = file_path
                    break

            if selected_file is not None:
                break

        if selected_file is None:
            selected_file = files[0]

        with selected_file.open(
            "r",
            encoding="utf-8",
        ) as file:
            payload = json.load(file)

        if not isinstance(
            payload,
            dict,
        ):
            raise ValueError(
                "Final inference JSON must contain "
                "a JSON object."
            )

        if (
            "results" in payload
            and isinstance(
                payload["results"],
                list,
            )
        ):
            for result in payload["results"]:
                if not isinstance(
                    result,
                    dict,
                ):
                    continue

                if str(
                    result.get(
                        "receipt_id",
                        "",
                    )
                ) == receipt_id:
                    return result

            if len(
                payload["results"]
            ) == 1 and isinstance(
                payload["results"][0],
                dict,
            ):
                return payload["results"][0]

        return payload

    # =========================================================
    # CONFIDENCE EXTRACTION
    # =========================================================

    @staticmethod
    def _extract_confidence(
        prediction: dict[str, Any],
    ) -> float | None:
        candidates = (
            prediction.get(
                "overall_confidence"
            ),
            prediction.get(
                "confidence"
            ),
            prediction.get(
                "prediction_confidence"
            ),
        )

        for value in candidates:
            if value is None:
                continue

            try:
                number = float(value)
            except (
                TypeError,
                ValueError,
            ):
                continue

            if 0.0 <= number <= 1.0:
                return number

        return None

    @staticmethod
    def _extract_reliability(
        prediction: dict[str, Any],
    ) -> str | None:
        value = prediction.get(
            "reliability"
        )

        if value is None:
            return None

        value = str(
            value
        ).strip().lower()

        if value in {
            "high",
            "medium",
            "low",
        }:
            return value

        return None

    @staticmethod
    def _extract_review_required(
        prediction: dict[str, Any],
    ) -> bool:
        value = prediction.get(
            "review_required",
            False,
        )

        if isinstance(
            value,
            bool,
        ):
            return value

        if isinstance(
            value,
            str,
        ):
            return value.strip().lower() in {
                "true",
                "1",
                "yes",
                "y",
            }

        if isinstance(
            value,
            (int, float),
        ):
            return bool(value)

        return False

    # =========================================================
    # SINGLE RECEIPT INFERENCE
    # =========================================================

    def predict_one(
        self,
        image_path: str | Path,
        *,
        save_prediction: bool | None = None,
    ) -> dict[str, Any]:
        started_at = time.perf_counter()

        source_path = (
            Path(image_path)
            .expanduser()
            .resolve()
        )

        self._validate_input_image(
            source_path
        )

        receipt_id = (
            self._safe_receipt_id(
                source_path
            )
        )

        effective_save_prediction = (
            self.inference_config.save_predictions
            if save_prediction is None
            else save_prediction
        )

        workspace_context = None

        try:
            workspace_context = (
                self._build_workspace()
            )

            workspace = Path(
                workspace_context.name
            )

            paths = self._prepare_workspace(
                workspace,
                source_path,
            )

            stage_timings: dict[str, float] = {}

            stage_start = time.perf_counter()

            preprocessing_result = (
                self._run_preprocessing(
                    paths,
                )
            )

            stage_timings[
                "preprocessing_ms"
            ] = round(
                (
                    time.perf_counter()
                    - stage_start
                ) * 1000.0,
                3,
            )

            stage_start = time.perf_counter()

            ocr_result = self._run_ocr(
                paths
            )

            stage_timings[
                "ocr_ms"
            ] = round(
                (
                    time.perf_counter()
                    - stage_start
                ) * 1000.0,
                3,
            )

            stage_start = time.perf_counter()

            cleaning_result = (
                self._run_cleaning(
                    paths
                )
            )

            stage_timings[
                "text_cleaning_ms"
            ] = round(
                (
                    time.perf_counter()
                    - stage_start
                ) * 1000.0,
                3,
            )

            stage_start = time.perf_counter()

            extraction_result = (
                self._run_extraction(
                    paths
                )
            )

            stage_timings[
                "field_extraction_ms"
            ] = round(
                (
                    time.perf_counter()
                    - stage_start
                ) * 1000.0,
                3,
            )

            stage_start = time.perf_counter()

            confidence_result = (
                self._run_confidence(
                    paths
                )
            )

            stage_timings[
                "confidence_ms"
            ] = round(
                (
                    time.perf_counter()
                    - stage_start
                ) * 1000.0,
                3,
            )

            stage_start = time.perf_counter()

            json_result = (
                self._run_json_generation(
                    paths
                )
            )

            stage_timings[
                "json_generation_ms"
            ] = round(
                (
                    time.perf_counter()
                    - stage_start
                ) * 1000.0,
                3,
            )

            prediction = (
                self._load_final_prediction(
                    paths["json"],
                    receipt_id,
                )
            )

            overall_confidence = (
                self._extract_confidence(
                    prediction
                )
            )

            reliability = (
                self._extract_reliability(
                    prediction
                )
            )

            review_required = (
                self._extract_review_required(
                    prediction
                )
            )

            total_processing_time_ms = round(
                (
                    time.perf_counter()
                    - started_at
                ) * 1000.0,
                3,
            )

            response = {
                "status": "success",
                "receipt_id": receipt_id,
                "input_path": str(
                    source_path
                ),
                "processing_time_ms": (
                    total_processing_time_ms
                ),
                "stage_timings_ms": stage_timings,
                "prediction": prediction,
                "overall_confidence": (
                    overall_confidence
                ),
                "reliability": reliability,
                "review_required": review_required,
                "timestamp_utc": (
                    self._utc_timestamp()
                ),
            }

            if effective_save_prediction:
                output_path = (
                    Path(
                        self.inference_config.output_dir
                    )
                    / f"{receipt_id}.json"
                )

                self.save_prediction(
                    response,
                    output_path,
                )

                response[
                    "prediction_path"
                ] = str(
                    output_path
                )

            self._log_mlflow_single(
                response
            )

            return response

        except Exception as exc:
            elapsed_ms = round(
                (
                    time.perf_counter()
                    - started_at
                ) * 1000.0,
                3,
            )

            failure_response = {
                "status": "failed",
                "receipt_id": receipt_id,
                "input_path": str(
                    source_path
                ),
                "processing_time_ms": elapsed_ms,
                "prediction": None,
                "overall_confidence": None,
                "reliability": None,
                "review_required": True,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
                "timestamp_utc": (
                    self._utc_timestamp()
                ),
            }

            self._log_mlflow_single_failure(
                failure_response
            )

            logger.exception(
                "Receipt inference failed: %s",
                source_path,
            )

            if self.inference_config.continue_on_error:
                return failure_response

            raise

        finally:
            if workspace_context is not None:
                workspace_context.cleanup()

    # =========================================================
    # BATCH INFERENCE
    # =========================================================

    def predict_batch(
        self,
        image_paths: Iterable[
            str | Path
        ],
        *,
        save_summary: bool = True,
    ) -> dict[str, Any]:
        started_at = time.perf_counter()

        paths = [
            Path(
                path
            ).expanduser().resolve()
            for path in image_paths
        ]

        if not paths:
            result = {
                "status": "empty",
                "total": 0,
                "successful": 0,
                "failed": 0,
                "processing_time_ms": 0.0,
                "results": [],
                "errors": [],
                "timestamp_utc": (
                    self._utc_timestamp()
                ),
            }

            if save_summary:
                self.save_prediction(
                    result,
                    Path(
                        self.inference_config.output_dir
                    )
                    / self.inference_config.prediction_filename,
                )

            return result

        results = []
        errors = []
        seen_ids: set[str] = set()

        for image_path in paths:
            receipt_id = (
                self._safe_receipt_id(
                    image_path
                )
            )

            if receipt_id in seen_ids:
                duplicate_error = {
                    "status": "failed",
                    "receipt_id": receipt_id,
                    "input_path": str(
                        image_path
                    ),
                    "error": {
                        "type": "DuplicateReceiptId",
                        "message": (
                            "Multiple input files resolve "
                            "to the same receipt_id."
                        ),
                    },
                }

                errors.append(
                    duplicate_error
                )

                if not self.inference_config.continue_on_error:
                    raise ValueError(
                        duplicate_error[
                            "error"
                        ]["message"]
                    )

                continue

            seen_ids.add(
                receipt_id
            )

            result = self.predict_one(
                image_path
            )

            if result.get(
                "status"
            ) == "success":
                results.append(
                    result
                )
            else:
                errors.append(
                    result
                )

                if not self.inference_config.continue_on_error:
                    raise RuntimeError(
                        result.get(
                            "error",
                            {},
                        ).get(
                            "message",
                            "Inference failed.",
                        )
                    )

        successful = len(
            results
        )

        failed = len(
            errors
        )

        total = len(
            paths
        )

        if failed == 0:
            status = "success"
        elif successful > 0:
            status = "partial"
        else:
            status = "failed"

        total_processing_time_ms = round(
            (
                time.perf_counter()
                - started_at
            ) * 1000.0,
            3,
        )

        batch_result = {
            "status": status,
            "total": total,
            "successful": successful,
            "failed": failed,
            "processing_time_ms": (
                total_processing_time_ms
            ),
            "results": results,
            "errors": errors,
            "timestamp_utc": (
                self._utc_timestamp()
            ),
        }

        if save_summary:
            output_path = (
                Path(
                    self.inference_config.output_dir
                )
                / self.inference_config.prediction_filename
            )

            self.save_prediction(
                batch_result,
                output_path,
            )

            batch_result[
                "prediction_path"
            ] = str(
                output_path
            )

        self._log_mlflow_batch(
            batch_result
        )

        return batch_result

    # =========================================================
    # DIRECTORY INFERENCE
    # =========================================================

    def predict_directory(
        self,
        input_directory: str | Path | None = None,
    ) -> dict[str, Any]:
        directory = Path(
            input_directory
            if input_directory is not None
            else self.inference_config.input_dir
        ).expanduser().resolve()

        if not directory.exists():
            raise FileNotFoundError(
                f"Inference input directory does not exist: "
                f"{directory}"
            )

        if not directory.is_dir():
            raise ValueError(
                f"Inference input path is not a directory: "
                f"{directory}"
            )

        if self.inference_config.recursive:
            candidates = directory.rglob("*")
        else:
            candidates = directory.glob("*")

        image_paths = sorted(
            [
                path
                for path in candidates
                if path.is_file()
                and path.suffix.lower()
                in self.inference_config.supported_extensions
            ],
            key=lambda path: path.as_posix().lower(),
        )

        return self.predict_batch(
            image_paths
        )

    # =========================================================
    # SAVE
    # =========================================================

    def save_prediction(
        self,
        result: dict[str, Any],
        output_path: str | Path,
    ) -> Path:
        destination = (
            Path(output_path)
            .expanduser()
            .resolve()
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if (
            destination.exists()
            and not self.inference_config.overwrite_predictions
        ):
            raise FileExistsError(
                f"Prediction already exists: "
                f"{destination}"
            )

        temporary_path = destination.with_suffix(
            destination.suffix + ".tmp"
        )

        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self._to_dict(result),
                file,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            file.write("\n")

        temporary_path.replace(
            destination
        )

        return destination

    # =========================================================
    # MLFLOW
    # =========================================================

    def _log_mlflow_single(
        self,
        result: dict[str, Any],
    ) -> None:
        if not self.inference_config.enable_mlflow:
            return

        if self.mlflow_tracker is None:
            logger.warning(
                "Inference MLflow logging is enabled, "
                "but no MLflow tracker was supplied."
            )
            return

        try:
            self.mlflow_tracker.log_tags(
                {
                    "inference_type": "single",
                    "inference_status": result[
                        "status"
                    ],
                }
            )

            self.mlflow_tracker.log_metrics(
                {
                    "inference_processing_time_ms": float(
                        result.get(
                            "processing_time_ms",
                            0.0,
                        )
                    ),
                }
            )

            confidence = result.get(
                "overall_confidence"
            )

            if confidence is not None:
                self.mlflow_tracker.log_metric(
                    "inference_confidence",
                    float(confidence),
                )

            self.mlflow_tracker.log_json_artifact(
                result,
                artifact_file=(
                    "inference/"
                    f"{result['receipt_id']}.json"
                ),
            )

        except Exception:
            logger.exception(
                "MLflow inference logging failed."
            )

    def _log_mlflow_single_failure(
        self,
        result: dict[str, Any],
    ) -> None:
        if not self.inference_config.enable_mlflow:
            return

        if self.mlflow_tracker is None:
            return

        try:
            self.mlflow_tracker.log_tag(
                "inference_status",
                "failed",
            )

            self.mlflow_tracker.log_json_artifact(
                result,
                artifact_file=(
                    "inference/errors/"
                    f"{result['receipt_id']}.json"
                ),
            )

        except Exception:
            logger.exception(
                "MLflow inference failure logging failed."
            )

    def _log_mlflow_batch(
        self,
        result: dict[str, Any],
    ) -> None:
        if not self.inference_config.enable_mlflow:
            return

        if self.mlflow_tracker is None:
            return

        try:
            self.mlflow_tracker.log_tags(
                {
                    "inference_type": "batch",
                    "inference_status": result[
                        "status"
                    ],
                }
            )

            self.mlflow_tracker.log_metrics(
                {
                    "inference_total": float(
                        result.get(
                            "total",
                            0,
                        )
                    ),
                    "inference_successful": float(
                        result.get(
                            "successful",
                            0,
                        )
                    ),
                    "inference_failed": float(
                        result.get(
                            "failed",
                            0,
                        )
                    ),
                    "inference_processing_time_ms": float(
                        result.get(
                            "processing_time_ms",
                            0.0,
                        )
                    ),
                }
            )

            self.mlflow_tracker.log_json_artifact(
                result,
                artifact_file=(
                    "inference/"
                    "batch_result.json"
                ),
            )

        except Exception:
            logger.exception(
                "MLflow batch logging failed."
            )