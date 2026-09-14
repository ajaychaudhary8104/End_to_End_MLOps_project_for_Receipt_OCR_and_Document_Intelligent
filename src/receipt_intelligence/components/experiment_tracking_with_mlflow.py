from __future__ import annotations
import dataclasses
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import mlflow
    from mlflow import MlflowClient
    from mlflow.entities import Run
except ImportError as exc:
    mlflow = None
    MlflowClient = None
    Run = Any
    _MLFLOW_IMPORT_ERROR = exc
else:
    _MLFLOW_IMPORT_ERROR = None

from src.receipt_intelligence.entity.config_entity import MLflowConfig
from src.receipt_intelligence import logger
from src.receipt_intelligence.config.configuration import ConfigurationManager
from src.receipt_intelligence.components.ocr_engine import OCREngine
from src.receipt_intelligence.components.ocr_text_cleaning import OCRTextCleaner
from src.receipt_intelligence.components.field_extraction_engine import FieldExtractor
from src.receipt_intelligence.components.confindence_and_reliability_engine import ConfidenceEngine
from src.receipt_intelligence.components.json_generation import JSONGenerator
from src.receipt_intelligence.components.financial_summary import FinancialSummaryEngine
from src.receipt_intelligence.components.evaluation import EvaluationEngine


@dataclass
class MLflowRunHandle:
    run_id: str
    experiment_id: str
    run_name: str
    started_at_utc: str
    owner_thread_id: int


class MLflowExperimentTracker:
    def __init__(self, config: MLflowConfig):
        self.config = config
        self.active_run: MLflowRunHandle | None = None
        self._lock = threading.RLock()
        self._metric_count = 0
        self._run_started_monotonic: float | None = None
        self._ensure_mlflow_available()
        self._configure_mlflow()
        self._ensure_local_directories()

    def _ensure_mlflow_available(self) -> None:
        if mlflow is None:
            raise ImportError(
                "MLflow is required for MLflowExperimentTracker. "
                "Install it with `pip install mlflow`."
            ) from _MLFLOW_IMPORT_ERROR

    def _configure_mlflow(self) -> None:
        tracking_uri = str(
            self.config.tracking_uri
        ).strip()

        if not tracking_uri:
            raise ValueError(
                "tracking_uri cannot be empty"
            )

        mlflow.set_tracking_uri(
            tracking_uri
        )

        client = MlflowClient(
            tracking_uri=tracking_uri
        )

        experiment = client.get_experiment_by_name(
            self.config.experiment_name
        )

        if experiment is None:
            create_kwargs = {
                "name": self.config.experiment_name
            }

            if self.config.artifact_location:
                create_kwargs[
                    "artifact_location"
                ] = self.config.artifact_location

            client.create_experiment(
                **create_kwargs
            )

        elif experiment.lifecycle_stage == "deleted":
            client.restore_experiment(
                experiment.experiment_id
            )

        mlflow.set_experiment(
            self.config.experiment_name
        )    

    def _ensure_local_directories(self) -> None:
        Path(self.config.local_summary_dir).mkdir(
            parents=True,
            exist_ok=True,
        )
        Path(self.config.local_artifact_dir).mkdir(
            parents=True,
            exist_ok=True,
        )

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _utc_iso(dt: datetime | None = None) -> str:
        value = dt or datetime.now(timezone.utc)
        return value.astimezone(timezone.utc).isoformat()

    @staticmethod
    def _sanitize_mlflow_name(value: Any, max_length: int = 250) -> str:
        text = str(value).strip()
        text = re.sub(r"[^A-Za-z0-9_.\-/ ]+", "_", text)
        text = re.sub(r"\s+", "_", text)
        text = text.strip("._-/")
        if not text:
            text = "unnamed"
        return text[:max_length]

    def _is_secret_key(self, key: str) -> bool:
        normalized = re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")
        return any(
            pattern in normalized
            for pattern in self.config.secret_key_patterns
        )

    @staticmethod
    def _redact_sensitive_urls(value: str) -> str:
        text = value

        text = re.sub(
            r"([a-zA-Z][a-zA-Z0-9+.-]*://)([^/@\s:]+):([^/@\s]+)@",
            r"\1***:***@",
            text,
        )

        text = re.sub(
            r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=-]+",
            r"\1 ***",
            text,
        )

        text = re.sub(
            r"(?i)\b(token|api[_-]?key|secret)\s*=\s*([^\s&;]+)",
            r"\1=***",
            text,
        )

        return text

    def _redact(self, value: Any, key: str | None = None) -> Any:
        if key is not None and self._is_secret_key(key):
            return "***REDACTED***"

        if value is None:
            return None

        if isinstance(value, Mapping):
            return {
                str(k): self._redact(v, str(k))
                for k, v in value.items()
            }

        if is_dataclass(value) and not isinstance(value, type):
            return self._redact(asdict(value))

        if isinstance(value, (list, tuple, set)):
            return [
                self._redact(item)
                for item in value
            ]

        if isinstance(value, Path):
            return str(value)

        if isinstance(value, bytes):
            return f"<bytes:{len(value)}>"

        if isinstance(value, str):
            return self._redact_sensitive_urls(value)

        if isinstance(value, (str, int, float, bool)):
            return value

        try:
            return self._redact(str(value))
        except Exception:
            return "<unserializable>"

    def _to_jsonable(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, bool)):
            return value

        if isinstance(value, float):
            if not math.isfinite(value):
                return None
            return value

        if isinstance(value, Path):
            return str(value)

        if isinstance(value, datetime):
            return self._utc_iso(value)

        if isinstance(value, Mapping):
            return {
                str(k): self._to_jsonable(v)
                for k, v in value.items()
            }

        if isinstance(value, (list, tuple, set)):
            return [
                self._to_jsonable(v)
                for v in value
            ]

        if is_dataclass(value) and not isinstance(value, type):
            return self._to_jsonable(asdict(value))

        if hasattr(value, "tolist"):
            try:
                return self._to_jsonable(value.tolist())
            except Exception:
                pass

        if hasattr(value, "item"):
            try:
                return self._to_jsonable(value.item())
            except Exception:
                pass

        return str(value)

    def _serialize_value(
        self,
        value: Any,
        max_length: int,
        strict: bool,
        field_name: str,
    ) -> str:
        redacted = self._redact(value)
        jsonable = self._to_jsonable(redacted)

        if isinstance(jsonable, str):
            text = jsonable
        else:
            try:
                text = json.dumps(
                    jsonable,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            except (TypeError, ValueError) as exc:
                if strict:
                    raise ValueError(
                        f"Unable to serialize MLflow {field_name}: {field_name}"
                    ) from exc
                text = str(jsonable)

        if len(text) > max_length:
            if strict:
                raise ValueError(
                    f"MLflow {field_name} '{field_name}' exceeds "
                    f"max length {max_length}"
                )
            text = text[: max_length - 3] + "..."

        return text

    def _normalize_params(
        self,
        params: Mapping[str, Any] | None,
    ) -> dict[str, str]:
        if not params:
            return {}

        normalized: dict[str, str] = {}

        for raw_key, raw_value in params.items():
            key = self._sanitize_mlflow_name(raw_key)
            if not key:
                continue

            normalized[key] = self._serialize_value(
                raw_value,
                self.config.max_param_length,
                self.config.strict_parameter_validation,
                "parameter",
            )

        return normalized

    def _normalize_tags(
        self,
        tags: Mapping[str, Any] | None,
    ) -> dict[str, str]:
        if not tags:
            return {}

        normalized: dict[str, str] = {}

        for raw_key, raw_value in tags.items():
            key = self._sanitize_mlflow_name(raw_key)
            if not key:
                continue

            normalized[key] = self._serialize_value(
                raw_value,
                self.config.max_tag_length,
                self.config.strict_tag_validation,
                "tag",
            )

        return normalized

    def _normalize_metrics(
        self,
        metrics: Mapping[str, Any] | None,
    ) -> dict[str, float]:
        if not metrics:
            return {}

        normalized: dict[str, float] = {}

        for raw_key, raw_value in metrics.items():
            if self._metric_count >= self.config.max_metric_count:
                raise RuntimeError(
                    f"Maximum MLflow metric count "
                    f"{self.config.max_metric_count} exceeded"
                )

            key = self._sanitize_mlflow_name(raw_key)
            if not key:
                continue

            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                if self.config.strict_metric_validation:
                    raise ValueError(
                        f"Metric '{raw_key}' must be numeric"
                    ) from exc
                continue

            if not math.isfinite(value):
                if self.config.strict_metric_validation:
                    raise ValueError(
                        f"Metric '{raw_key}' must be finite"
                    )
                continue

            normalized[key] = value

        return normalized

    @staticmethod
    def _run_info_payload(run_info: Any) -> dict[str, Any]:
        return {
            "run_id": getattr(run_info, "run_id", None),
            "run_name": getattr(run_info, "run_name", None),
            "experiment_id": getattr(run_info, "experiment_id", None),
            "status": getattr(run_info, "status", None),
            "start_time": getattr(run_info, "start_time", None),
            "end_time": getattr(run_info, "end_time", None),
            "artifact_uri": getattr(run_info, "artifact_uri", None),
            "lifecycle_stage": getattr(
                run_info,
                "lifecycle_stage",
                None,
            ),
        }

    @staticmethod
    def _run_data_payload(run_data: Any) -> dict[str, Any]:
        return {
            "params": dict(getattr(run_data, "params", {}) or {}),
            "metrics": dict(getattr(run_data, "metrics", {}) or {}),
            "tags": dict(getattr(run_data, "tags", {}) or {}),
        }

    @staticmethod
    def _git_command(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            value = completed.stdout.strip()
            return value or None
        except (
            OSError,
            subprocess.SubprocessError,
            TimeoutError,
        ):
            return None

    def _git_metadata(self) -> dict[str, Any]:
        commit = self._git_command(
            "rev-parse",
            "HEAD",
        )
        branch = self._git_command(
            "rev-parse",
            "--abbrev-ref",
            "HEAD",
        )
        status = self._git_command(
            "status",
            "--porcelain",
        )

        return {
            "git_commit": commit,
            "git_branch": branch,
            "git_dirty": bool(status) if status is not None else None,
        }

    @staticmethod
    def _package_version(package_name: str) -> str | None:
        try:
            return importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            return None
        except Exception:
            return None

    def _environment_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "os_name": os.name,
            "machine": platform.machine(),
            "processor": platform.processor(),
            "system": platform.system(),
            "release": platform.release(),
        }

        package_names = (
            "mlflow",
            "numpy",
            "pandas",
            "scipy",
            "opencv-python",
            "paddleocr",
            "paddlepaddle",
            "fastapi",
            "pydantic",
        )

        for package_name in package_names:
            version = self._package_version(package_name)
            if version:
                key = re.sub(
                    r"[^a-zA-Z0-9]+",
                    "_",
                    package_name,
                ).strip("_")
                metadata[f"{key}_version"] = version

        return metadata

    @staticmethod
    def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(chunk_size)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    def _sha256_directory(self, path: Path) -> str:
        digest = hashlib.sha256()

        files = [
            file_path
            for file_path in path.rglob("*")
            if file_path.is_file()
        ]

        for file_path in sorted(
            files,
            key=lambda item: item.relative_to(path).as_posix(),
        ):
            relative_path = file_path.relative_to(path).as_posix()
            digest.update(relative_path.encode("utf-8"))

            file_hash = self._sha256_file(file_path)
            digest.update(file_hash.encode("ascii"))

        return digest.hexdigest()

    def _dataset_metadata(
        self,
        dataset_path: str | Path | None,
        dataset_name: str | None,
        dataset_version: str | None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {}

        if dataset_name:
            metadata["dataset_name"] = dataset_name

        if dataset_version:
            metadata["dataset_version"] = dataset_version

        if dataset_path:
            path = Path(dataset_path).expanduser().resolve()

            if not path.exists():
                raise FileNotFoundError(
                    f"Dataset path does not exist: {path}"
                )

            metadata["dataset_path"] = str(path)

            if self.config.hash_dataset_files:
                if path.is_file():
                    metadata["dataset_sha256"] = self._sha256_file(path)
                elif path.is_dir():
                    metadata["dataset_sha256"] = self._sha256_directory(
                        path
                    )

        return metadata

    def _assert_run_available(self) -> MLflowRunHandle:
        if self.active_run is None:
            raise RuntimeError(
                "No active MLflow run. Call start_run() first."
            )

        current_thread_id = threading.get_ident()

        if self.active_run.owner_thread_id != current_thread_id:
            raise RuntimeError(
                "The active MLflow run belongs to a different thread. "
                "Create one tracker instance per request/thread."
            )

        return self.active_run

    def start_run(
        self,
        run_name: str | None = None,
        params: Mapping[str, Any] | None = None,
        tags: Mapping[str, Any] | None = None,
        dataset_path: str | Path | None = None,
        dataset_name: str | None = None,
        dataset_version: str | None = None,
    ) -> MLflowRunHandle:
        with self._lock:
            current_thread_id = threading.get_ident()

            if self.active_run is not None:
                raise RuntimeError(
                    f"Tracker already owns active run "
                    f"{self.active_run.run_id}"
                )

            existing_mlflow_run = mlflow.active_run()

            if (
                existing_mlflow_run is not None
                and not self.config.allow_existing_active_run
            ):
                raise RuntimeError(
                    "An MLflow run is already active in the current context: "
                    f"{existing_mlflow_run.info.run_id}. "
                    "Do not nest runs unintentionally."
                )

            final_run_name = (
                run_name.strip()
                if isinstance(run_name, str) and run_name.strip()
                else (
                    f"{self.config.run_name_prefix}-"
                    f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
                    f"{uuid.uuid4().hex[:8]}"
                )
            )

            final_run_name = self._sanitize_mlflow_name(
                final_run_name,
                max_length=250,
            )

            self._metric_count = 0
            self._run_started_monotonic = time.monotonic()

            started_at = self._utc_now()

            active_mlflow_run = mlflow.start_run(
                run_name=final_run_name,
                log_system_metrics=self.config.log_system_metrics,
            )

            run_info = active_mlflow_run.info

            handle = MLflowRunHandle(
                run_id=run_info.run_id,
                experiment_id=str(run_info.experiment_id),
                run_name=final_run_name,
                started_at_utc=self._utc_iso(started_at),
                owner_thread_id=current_thread_id,
            )

            self.active_run = handle

            try:
                base_tags: dict[str, Any] = {
                    "run_id": handle.run_id,
                    "run_name": final_run_name,
                    "pipeline": "receipt_ocr",
                    "tracker_version": "2.0",
                }

                if self.config.log_python_metadata:
                    env_metadata = self._environment_metadata()
                    base_tags.update(
                        {
                            f"env_{key}": value
                            for key, value in env_metadata.items()
                        }
                    )

                if self.config.log_git_metadata:
                    git_metadata = self._git_metadata()
                    base_tags.update(
                        {
                            key: value
                            for key, value in git_metadata.items()
                            if value is not None
                        }
                    )

                dataset_metadata = self._dataset_metadata(
                    dataset_path=dataset_path,
                    dataset_name=dataset_name,
                    dataset_version=dataset_version,
                )

                base_tags.update(dataset_metadata)

                merged_tags = {
                    **base_tags,
                    **(tags or {}),
                }

                normalized_tags = self._normalize_tags(merged_tags)
                if normalized_tags:
                    mlflow.set_tags(normalized_tags)

                normalized_params = self._normalize_params(params)
                if normalized_params:
                    mlflow.log_params(normalized_params)

                if self.config.log_python_metadata:
                    environment = self._redact(
                        self._environment_metadata()
                    )
                    self.log_json_artifact(
                        environment,
                        artifact_file="metadata/environment.json",
                    )

                if self.config.log_git_metadata:
                    git_metadata = self._redact(
                        self._git_metadata()
                    )
                    self.log_json_artifact(
                        git_metadata,
                        artifact_file="metadata/git.json",
                    )

                self.log_json_artifact(
                    {
                        "dataset": self._redact(
                            dataset_metadata
                        ),
                    },
                    artifact_file="metadata/dataset.json",
                )

                self.log_json_artifact(
                    self._redact(
                        asdict(self.config)
                    ),
                    artifact_file="metadata/tracker_config.json",
                )

                return handle

            except Exception:
                try:
                    mlflow.end_run(status="FAILED")
                finally:
                    self.active_run = None
                    self._run_started_monotonic = None
                raise

    def log_params(
        self,
        params: Mapping[str, Any],
    ) -> None:
        self._assert_run_available()
        normalized = self._normalize_params(params)
        if normalized:
            mlflow.log_params(normalized)

    def log_param(
        self,
        key: str,
        value: Any,
    ) -> None:
        self.log_params({key: value})

    def log_metrics(
        self,
        metrics: Mapping[str, Any],
    ) -> None:
        self._assert_run_available()
        normalized = self._normalize_metrics(metrics)

        if normalized:
            mlflow.log_metrics(normalized)
            self._metric_count += len(normalized)

    def log_metric(
        self,
        key: str,
        value: Any,
    ) -> None:
        self.log_metrics({key: value})

    def log_tags(
        self,
        tags: Mapping[str, Any],
    ) -> None:
        self._assert_run_available()
        normalized = self._normalize_tags(tags)
        if normalized:
            mlflow.set_tags(normalized)

    def log_tag(
        self,
        key: str,
        value: Any,
    ) -> None:
        self.log_tags({key: value})

    def log_json_artifact(
        self,
        payload: Any,
        artifact_file: str,
    ) -> None:
        self._assert_run_available()

        artifact_file = artifact_file.replace("\\", "/").lstrip("/")

        if not artifact_file.endswith(".json"):
            artifact_file = f"{artifact_file}.json"

        artifact_path = Path(artifact_file)

        if artifact_path.is_absolute():
            raise ValueError(
                "artifact_file must be a relative MLflow artifact path"
            )

        parent = artifact_path.parent
        artifact_dir = (
            None
            if str(parent) == "."
            else str(parent)
        )

        safe_payload = self._redact(
            self._to_jsonable(payload)
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            local_file = temp_root / artifact_path.name

            local_file.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with local_file.open(
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    safe_payload,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                handle.write("\n")

            mlflow.log_artifact(
                str(local_file),
                artifact_path=artifact_dir,
            )

    def log_text_artifact(
        self,
        text: str,
        artifact_file: str,
    ) -> None:
        self._assert_run_available()

        artifact_file = artifact_file.replace("\\", "/").lstrip("/")

        artifact_path = Path(artifact_file)

        if artifact_path.is_absolute():
            raise ValueError(
                "artifact_file must be a relative MLflow artifact path"
            )

        parent = artifact_path.parent
        artifact_dir = (
            None
            if str(parent) == "."
            else str(parent)
        )

        safe_text = self._redact(text)

        with tempfile.TemporaryDirectory() as temp_dir:
            local_file = Path(temp_dir) / artifact_path.name

            local_file.write_text(
                str(safe_text),
                encoding="utf-8",
            )

            mlflow.log_artifact(
                str(local_file),
                artifact_path=artifact_dir,
            )

    def log_local_file(
        self,
        file_path: str | Path,
        artifact_path: str | None = None,
    ) -> None:
        self._assert_run_available()

        source = Path(file_path).expanduser().resolve()

        if not source.exists():
            raise FileNotFoundError(
                f"Artifact file does not exist: {source}"
            )

        if not source.is_file():
            raise ValueError(
                f"Artifact path is not a file: {source}"
            )

        if artifact_path is None:
            destination_dir = None
        else:
            normalized = artifact_path.replace("\\", "/").strip("/")
            destination_dir = normalized or None

        mlflow.log_artifact(
            str(source),
            artifact_path=destination_dir,
        )

    def log_local_directory(
        self,
        directory_path: str | Path,
        artifact_path: str | None = None,
    ) -> None:
        self._assert_run_available()

        source = Path(directory_path).expanduser().resolve()

        if not source.exists():
            raise FileNotFoundError(
                f"Artifact directory does not exist: {source}"
            )

        if not source.is_dir():
            raise ValueError(
                f"Artifact path is not a directory: {source}"
            )

        normalized = (
            artifact_path.replace("\\", "/").strip("/")
            if artifact_path
            else None
        )

        mlflow.log_artifacts(
            str(source),
            artifact_path=normalized,
        )

    def log_evaluation_result(
        self,
        evaluation_result: Any,
        artifact_file: str = "evaluation/evaluation_result.json",
        metric_prefix: str = "evaluation",
    ) -> None:
        self._assert_run_available()

        payload = self._to_jsonable(
            evaluation_result
        )

        self.log_json_artifact(
            payload,
            artifact_file=artifact_file,
        )

        if isinstance(payload, Mapping):
            numeric_metrics: dict[str, float] = {}

            for key, value in payload.items():
                if isinstance(value, bool):
                    continue

                if isinstance(value, (int, float)):
                    if isinstance(value, float) and not math.isfinite(value):
                        continue

                    metric_name = (
                        f"{metric_prefix}_{key}"
                        if metric_prefix
                        else str(key)
                    )
                    numeric_metrics[metric_name] = float(value)

            if numeric_metrics:
                self.log_metrics(numeric_metrics)

            for status_key in (
                "status",
                "quality_gate_status",
                "reliability",
                "review_required",
            ):
                if status_key in payload and payload[status_key] is not None:
                    self.log_tag(
                        f"{metric_prefix}_{status_key}",
                        payload[status_key],
                    )

    def log_quality_gate(
        self,
        passed: bool,
        status: str,
        reasons: Sequence[str] | None = None,
    ) -> None:
        self._assert_run_available()

        clean_reasons = []
        for reason in reasons or []:
            text = str(reason).strip()
            if text:
                clean_reasons.append(text)

        self.log_metrics(
            {
                "quality_gate_passed": 1.0 if passed else 0.0,
                "quality_gate_failure_count": float(
                    len(clean_reasons)
                ),
            }
        )

        self.log_tags(
            {
                "quality_gate_status": str(status),
                "quality_gate_passed": str(bool(passed)).lower(),
                "quality_gate_reasons": json.dumps(
                    clean_reasons,
                    ensure_ascii=False,
                ),
            }
        )

    def log_pipeline_status(
        self,
        status: str,
        error: str | None = None,
    ) -> None:
        self._assert_run_available()

        clean_status = str(status).strip().lower()

        allowed_statuses = {
            "success",
            "partial",
            "failed",
            "empty",
            "no_ground_truth",
            "skipped",
        }

        if clean_status not in allowed_statuses:
            raise ValueError(
                f"Unsupported pipeline status: {clean_status}"
            )

        tags = {
            "pipeline_status": clean_status,
        }

        if error:
            safe_error = self._redact_sensitive_urls(
                str(error)
            )
            tags["pipeline_error"] = safe_error

        self.log_tags(tags)

    def _get_current_mlflow_run(self):
        handle = self._assert_run_available()

        run = MlflowClient().get_run(handle.run_id)
        return run

    def save_run_summary(
        self,
        extra_payload: Mapping[str, Any] | None = None,
    ) -> str:
        handle = self._assert_run_available()

        run = self._get_current_mlflow_run()

        payload: dict[str, Any] = {
            "generated_at_utc": self._utc_iso(),
            "run": {
                "info": self._run_info_payload(
                    run.info
                ),
                "data": self._run_data_payload(
                    run.data
                ),
            },
            "tracker": {
                "tracking_uri": self.config.tracking_uri,
                "experiment_name": self.config.experiment_name,
            },
        }

        if self._run_started_monotonic is not None:
            payload["tracker"]["elapsed_seconds"] = round(
                time.monotonic() - self._run_started_monotonic,
                6,
            )

        if extra_payload:
            payload["extra"] = self._redact(
                self._to_jsonable(
                    extra_payload
                )
            )

        local_directory = Path(
            self.config.local_summary_dir
        )
        local_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        local_file = (
            local_directory
            / f"{handle.run_id}.json"
        )

        temp_file = local_file.with_suffix(
            local_file.suffix + ".tmp"
        )

        with temp_file.open(
            "w",
            encoding="utf-8",
        ) as file_handle:
            json.dump(
                self._redact(
                    self._to_jsonable(payload)
                ),
                file_handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            file_handle.write("\n")

        os.replace(
            temp_file,
            local_file,
        )

        self.log_json_artifact(
            payload,
            artifact_file=(
                f"run_metadata/{handle.run_id}.json"
            ),
        )

        return str(local_file)

    def end_run(
        self,
        status: str = "FINISHED",
        extra_summary: Mapping[str, Any] | None = None,
    ) -> str | None:
        with self._lock:
            handle = self.active_run

            if handle is None:
                return None

            try:
                if self.config.save_local_run_summary:
                    self.save_run_summary(
                        extra_payload=extra_summary
                    )

                normalized_status = str(status).upper().strip()

                allowed_statuses = {
                    "FINISHED",
                    "FAILED",
                    "KILLED",
                }

                if normalized_status not in allowed_statuses:
                    raise ValueError(
                        f"Unsupported MLflow run status: "
                        f"{normalized_status}"
                    )

                run_id = handle.run_id
                mlflow.end_run(
                    status=normalized_status
                )
                return run_id

            finally:
                self.active_run = None
                self._run_started_monotonic = None
                self._metric_count = 0

    def fail_run(
        self,
        error: BaseException | str,
        extra_summary: Mapping[str, Any] | None = None,
    ) -> str | None:
        with self._lock:
            handle = self.active_run

            if handle is None:
                return None

            try:
                error_text = (
                    str(error)
                    if not isinstance(error, BaseException)
                    else (
                        f"{type(error).__name__}: "
                        f"{error}"
                    )
                )

                self.log_tags(
                    {
                        "run_status": "failed",
                        "error_type": (
                            type(error).__name__
                            if isinstance(error, BaseException)
                            else "RuntimeError"
                        ),
                        "error_message": (
                            self._redact_sensitive_urls(
                                error_text
                            )
                        ),
                    }
                )

                self.log_text_artifact(
                    (
                        self._redact_sensitive_urls(
                            error_text
                        )
                    ),
                    artifact_file=(
                        "errors/run_error.txt"
                    ),
                )

                return self.end_run(
                    status="FAILED",
                    extra_summary=extra_summary,
                )

            except Exception:
                try:
                    mlflow.end_run(
                        status="FAILED"
                    )
                finally:
                    self.active_run = None
                    self._run_started_monotonic = None
                    self._metric_count = 0
                raise

    def register_model(
        self,
        artifact_path: str,
        registered_model_name: str,
        await_registration_for: int = 300,
        tags: Mapping[str, Any] | None = None,
    ) -> Any:
        self._assert_run_available()

        clean_name = str(
            registered_model_name
        ).strip()

        if not clean_name:
            raise ValueError(
                "registered_model_name cannot be empty"
            )

        model_uri = (
            f"runs:/{self.active_run.run_id}/"
            f"{artifact_path.strip('/')}"
        )

        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=clean_name,
            await_registration_for=await_registration_for,
        )

        if tags:
            client = MlflowClient()
            version = client.get_model_version(
                name=clean_name,
                version=str(
                    registered_model.version
                ),
            )

            normalized_tags = self._normalize_tags(tags)

            for key, value in normalized_tags.items():
                client.set_model_version_tag(
                    name=clean_name,
                    version=version.version,
                    key=key,
                    value=value,
                )

        return registered_model

    def restore_local_artifact_copy(
        self,
        source_file: str | Path,
        destination_relative_path: str,
    ) -> str:
        source = Path(source_file).expanduser().resolve()

        if not source.is_file():
            raise FileNotFoundError(
                f"Source file does not exist: {source}"
            )

        destination_root = Path(
            self.config.local_artifact_dir
        ).resolve()

        relative = Path(
            destination_relative_path
        )

        if relative.is_absolute():
            raise ValueError(
                "destination_relative_path must be relative"
            )

        destination = (
            destination_root / relative
        ).resolve()

        if destination_root not in destination.parents and (
            destination != destination_root
        ):
            raise ValueError(
                "Destination path escapes local artifact directory"
            )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temp_destination = destination.with_suffix(
            destination.suffix + ".tmp"
        )

        shutil.copy2(
            source,
            temp_destination,
        )

        os.replace(
            temp_destination,
            destination,
        )

        return str(destination)

    def __enter__(self) -> "MLflowExperimentTracker":
        self.start_run()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> bool:
        if exc_value is not None:
            self.fail_run(
                exc_value
            )
            return False

        self.end_run(
            status="FINISHED"
        )
        return False


def create_mlflow_tracker(
    config: MLflowConfig | None = None,
) -> MLflowExperimentTracker:
    return MLflowExperimentTracker(
        config or MLflowConfig()
    )


def run_receipt_pipeline_with_mlflow(
    pipeline_callable: Any,
    *,
    run_name: str | None = None,
    params: Mapping[str, Any] | None = None,
    tags: Mapping[str, Any] | None = None,
    dataset_path: str | Path | None = None,
    dataset_name: str | None = None,
    dataset_version: str | None = None,
) -> Any:
   
    config = ConfigurationManager()
    mlflow_config = config.get_mlflow_config()

    tracker = MLflowExperimentTracker(mlflow_config)

    tracker.start_run(
        run_name=run_name,
        params=params,
        tags=tags,
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
    )

    try:
        result = pipeline_callable()

        if result is not None:
            result_payload = (
                asdict(result)
                if is_dataclass(result)
                else result
            )

            if isinstance(result_payload, Mapping):
                status = result_payload.get(
                    "status"
                )

                if status is not None:
                    tracker.log_pipeline_status(
                        str(status)
                    )

                tracker.log_json_artifact(
                    result_payload,
                    artifact_file=(
                        "pipeline/pipeline_result.json"
                    ),
                )

                numeric_metrics = {
                    f"pipeline_{key}": value
                    for key, value in result_payload.items()
                    if isinstance(value, (int, float))
                    and not isinstance(value, bool)
                }

                if numeric_metrics:
                    tracker.log_metrics(
                        numeric_metrics
                    )

        tracker.end_run(
            status="FINISHED"
        )

        return result

    except Exception as exc:
        tracker.fail_run(
            exc
        )
        raise



class ReceiptOCRPipeline:
    def __init__(self):
        self.config = ConfigurationManager()
        
    def _result_to_dict(self, result: Any) -> Any:
        if result is None:
            return None

        if is_dataclass(result):
            return asdict(result)

        if isinstance(result, dict):
            return result

        if isinstance(result, (list, tuple)):
            return [
                self._result_to_dict(item)
                for item in result
            ]

        return result

    def run_ocr(self) -> Any:
        start = time.perf_counter()
        ocr_config = self.config.get_ocr_engine_config()
        ocr_engine = OCREngine(
            config=ocr_config
        )

        result = ocr_engine.run()

        elapsed_ms = round(
            (time.perf_counter() - start) * 1000.0,
            3,
        )

        logger.info(
            "OCR stage completed in %.3f ms",
            elapsed_ms,
        )

        return result

    def run_text_cleaning(
        self,
        ocr_result: Any,
    ) -> Any:
        start = time.perf_counter()

        cleaning_config = self.config.get_ocr_text_cleaning_config()
        text_cleaning_engine = OCRTextCleaner(
            config=cleaning_config
        )

        result = text_cleaning_engine.run()

        elapsed_ms = round(
            (time.perf_counter() - start) * 1000.0,
            3,
        )

        logger.info(
            "Text cleaning stage completed in %.3f ms",
            elapsed_ms,
        )

        return result

    def run_field_extraction(
        self,
        cleaning_result: Any,
    ) -> Any:
        start = time.perf_counter()

        extraction_config = self.config.get_field_extraction_engine_config()
        extraction_engine = FieldExtractor(
            config=extraction_config
        )

        result = extraction_engine.run()

        elapsed_ms = round(
            (time.perf_counter() - start) * 1000.0,
            3,
        )

        logger.info(
            "Field extraction stage completed in %.3f ms",
            elapsed_ms,
        )

        return result

    def run_confidence_engine(
        self,
        extraction_result: Any,
        cleaning_result: Any,
    ) -> Any:
        start = time.perf_counter()

        confidence_config = self.config.get_confidence_and_reliability_engine_config()
        confidence_engine = ConfidenceEngine(
            config=confidence_config
        )

        result = confidence_engine.run()

        elapsed_ms = round(
            (time.perf_counter() - start) * 1000.0,
            3,
        )

        logger.info(
            "Confidence stage completed in %.3f ms",
            elapsed_ms,
        )

        return result

    def run_json_generation(
        self,
        confidence_result: Any,
    ) -> Any:
        start = time.perf_counter()

        json_config = self.config.get_json_generator_config()
        json_generator = JSONGenerator(
            config=json_config
        )

        result = json_generator.run()

        elapsed_ms = round(
            (time.perf_counter() - start) * 1000.0,
            3,
        )

        logger.info(
            "JSON generation stage completed in %.3f ms",
            elapsed_ms,
        )

        return result

    def run_financial_summary(
        self,
        json_result: Any,
    ) -> Any:
        start = time.perf_counter()

        financial_config = self.config.get_financial_summary_config()
        financial_engine = FinancialSummaryEngine(
            config=financial_config
        )

        result = financial_engine.run()

        elapsed_ms = round(
            (time.perf_counter() - start) * 1000.0,
            3,
        )

        logger.info(
            "Financial summary stage completed in %.3f ms",
            elapsed_ms,
        )

        return result

    def run_evaluation(
        self,
        json_result: Any,
        cleaning_result: Any,
    ) -> Any:
        start = time.perf_counter()

        evaluation_config = self.config.get_evaluation_config()
        evaluation_engine = EvaluationEngine(
            config=evaluation_config
        )

        result = evaluation_engine.run()

        elapsed_ms = round(
            (time.perf_counter() - start) * 1000.0,
            3,
        )

        logger.info(
            "Evaluation stage completed in %.3f ms",
            elapsed_ms,
        )

        return result

    def run_pipeline(self) -> dict[str, Any]:
        pipeline_start = time.perf_counter()

        stage_results: dict[str, Any] = {}
        stage_timings: dict[str, float] = {}

        mlflow_tracker = None

        try:
            mlflow_config = self.config.get_mlflow_config()

            mlflow_tracker = MLflowExperimentTracker(
                config=mlflow_config
            )

            mlflow_tracker.start_run(
                run_name="receipt-ocr-full-pipeline",
                params={
                    "pipeline_name": "receipt_ocr_idp",
                    "ocr_engine": "PaddleOCR",
                    "text_cleaning_enabled": True,
                    "field_extraction_enabled": True,
                    "confidence_engine_enabled": True,
                    "json_generation_enabled": True,
                    "financial_summary_enabled": True,
                    "evaluation_enabled": True,
                },
                tags={
                    "project": "receipt_ocr_document_intelligent",
                    "pipeline_type": "end_to_end",
                    "environment": "development",
                    "task": "receipt_ocr_and_information_extraction",
                },
                dataset_path="artifacts/data_ingestion/AI-OCR dataset",
                dataset_name="AI-OCR dataset",
                dataset_version="v1.0",
            )

            stages = (
                (
                    "ocr",
                    self.run_ocr,
                    "stage_ocr_processing_time_ms",
                ),
                (
                    "text_cleaning",
                    lambda: self.run_text_cleaning(
                        stage_results["ocr"]
                    ),
                    "stage_text_cleaning_processing_time_ms",
                ),
                (
                    "field_extraction",
                    lambda: self.run_field_extraction(
                        stage_results["text_cleaning"]
                    ),
                    "stage_field_extraction_processing_time_ms",
                ),
                (
                    "confidence_engine",
                    lambda: self.run_confidence_engine(
                        stage_results["field_extraction"],
                        stage_results["text_cleaning"],
                    ),
                    "stage_confidence_engine_processing_time_ms",
                ),
                (
                    "json_generation",
                    lambda: self.run_json_generation(
                        stage_results["confidence_engine"]
                    ),
                    "stage_json_generation_processing_time_ms",
                ),
                (
                    "financial_summary",
                    lambda: self.run_financial_summary(
                        stage_results["json_generation"]
                    ),
                    "stage_financial_summary_processing_time_ms",
                ),
                (
                    "evaluation",
                    lambda: self.run_evaluation(
                        stage_results["json_generation"],
                        stage_results["text_cleaning"],
                    ),
                    "stage_evaluation_processing_time_ms",
                ),
            )

            for stage_name, stage_callable, metric_name in stages:
                stage_start = time.perf_counter()

                result = stage_callable()

                elapsed_ms = round(
                    (time.perf_counter() - stage_start) * 1000.0,
                    3,
                )

                stage_results[stage_name] = result
                stage_timings[stage_name] = elapsed_ms

                mlflow_tracker.log_metric(
                    metric_name,
                    elapsed_ms,
                )

                mlflow_tracker.log_tag(
                    f"stage_{stage_name}_status",
                    "success",
                )

            total_processing_time_ms = round(
                (time.perf_counter() - pipeline_start) * 1000.0,
                3,
            )

            pipeline_result = {
                "status": "success",
                "pipeline_name": "receipt_ocr_idp",
                "processing_time_ms": total_processing_time_ms,
                "stage_timings_ms": stage_timings,
                "stages": {
                    stage_name: self._result_to_dict(
                        stage_results[stage_name]
                    )
                    for stage_name in stage_results
                },
            }

            mlflow_tracker.log_metrics(
                {
                    "pipeline_processing_time_ms": (
                        total_processing_time_ms
                    ),
                    "pipeline_stage_count": float(
                        len(stage_results)
                    ),
                    "pipeline_success": 1.0,
                }
            )

            evaluation_result = stage_results.get(
                "evaluation"
            )

            if evaluation_result is not None:
                mlflow_tracker.log_evaluation_result(
                    self._result_to_dict(
                        evaluation_result
                    ),
                    artifact_file=(
                        "evaluation/evaluation_result.json"
                    ),
                    metric_prefix="evaluation",
                )

            mlflow_tracker.log_pipeline_status(
                status="success"
            )

            mlflow_tracker.log_json_artifact(
                pipeline_result,
                artifact_file=(
                    "pipeline/pipeline_result.json"
                ),
            )

            mlflow_tracker.log_json_artifact(
                {
                    "stage_timings_ms": stage_timings,
                    "stage_status": {
                        stage_name: "success"
                        for stage_name in stage_results
                    },
                },
                artifact_file=(
                    "pipeline/stage_summary.json"
                ),
            )

            mlflow_tracker.end_run(
                status="FINISHED",
                extra_summary={
                    "pipeline_status": "success",
                    "processing_time_ms": (
                        total_processing_time_ms
                    ),
                    "stage_timings_ms": stage_timings,
                },
            )

            return pipeline_result

        except Exception as exc:
            total_processing_time_ms = round(
                (time.perf_counter() - pipeline_start) * 1000.0,
                3,
            )

            failure_result = {
                "status": "failed",
                "pipeline_name": "receipt_ocr_idp",
                "processing_time_ms": (
                    total_processing_time_ms
                ),
                "stage_timings_ms": stage_timings,
                "completed_stages": list(
                    stage_results.keys()
                ),
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }

            if mlflow_tracker is not None:
                try:
                    mlflow_tracker.log_json_artifact(
                        failure_result,
                        artifact_file=(
                            "pipeline/pipeline_failure.json"
                        ),
                    )

                    mlflow_tracker.fail_run(
                        exc,
                        extra_summary=failure_result,
                    )
                except Exception:
                    logger.exception(
                        "Failed to finalize MLflow failure state."
                    )

            logger.exception(
                "Receipt OCR pipeline failed."
            )

            raise
