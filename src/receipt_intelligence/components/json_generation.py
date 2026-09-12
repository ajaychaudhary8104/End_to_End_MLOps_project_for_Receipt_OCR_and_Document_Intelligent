from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
from src.receipt_intelligence.entity.config_entity import JSONGeneratorConfig
from src.receipt_intelligence import logger

class JSONGenerator:
    """
    Production-grade final JSON generator.

    Pipeline:

        Confidence Result
                ↓
        Input Validation
                ↓
        Type Normalization
                ↓
        Monetary Normalization
                ↓
        Date Validation
                ↓
        Confidence Normalization
                ↓
        Reliability Normalization
                ↓
        Assignment JSON
                ↓
        Confidence Metadata
                ↓
        Final Schema Validation
                ↓
        JSON Artifact
                ↓
        Batch Manifest

    Important:
        This class does NOT perform OCR.
        This class does NOT perform field extraction.
        This class does NOT invent missing values.

    It converts already-produced confidence results into
    a stable, downstream-safe JSON representation.
    """

    REQUIRED_TOP_LEVEL_FIELDS = (
        "store_name",
        "date",
        "items",
        "total_amount",
    )

    ALLOWED_RELIABILITY = frozenset(
        {
            "high",
            "medium",
            "low",
        }
    )

    # ========================================================
    # INIT
    # ========================================================

    def __init__(
        self,
        config: JSONGeneratorConfig,
    ):
        self.config = config

        self.input_dir = Path(
            config.input_dir
        )

        self.output_dir = Path(
            config.output_dir
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Validate configuration.
        # ----------------------------------------------------

        if config.round_amount_digits < 0:
            raise ValueError(
                "round_amount_digits cannot be negative."
            )

        if config.round_confidence_digits < 0:
            raise ValueError(
                "round_confidence_digits cannot be negative."
            )

        if not (
            0.0
            <= config.low_confidence_threshold
            <= 1.0
        ):
            raise ValueError(
                "low_confidence_threshold must be between 0 and 1."
            )

        if not (
            0.0
            <= config.medium_confidence_threshold
            <= 1.0
        ):
            raise ValueError(
                "medium_confidence_threshold must be between 0 and 1."
            )

        if (
            config.low_confidence_threshold
            >= config.medium_confidence_threshold
        ):
            raise ValueError(
                "low_confidence_threshold must be lower than "
                "medium_confidence_threshold."
            )

        if not config.schema_version.strip():
            raise ValueError(
                "schema_version cannot be empty."
            )

    # ========================================================
    # DISCOVER INPUT FILES
    # ========================================================

    def _discover_files(self) -> list[Path]:
        """
        Discover confidence-result JSON files.
        """

        if not self.input_dir.exists():
            raise FileNotFoundError(
                f"Confidence directory does not exist: "
                f"{self.input_dir}"
            )

        if not self.input_dir.is_dir():
            raise NotADirectoryError(
                f"Expected directory: "
                f"{self.input_dir}"
            )

        files = sorted(
            path
            for path in self.input_dir.rglob("*.json")
            if path.name
            not in {
                "confidence_manifest.json",
                "json_manifest.json",
            }
        )

        logger.info(
            "Discovered %d confidence result files.",
            len(files),
        )

        return files

    # ========================================================
    # LOAD JSON
    # ========================================================

    @staticmethod
    def _load_json(
        path: Path,
    ) -> dict[str, Any]:
        """
        Load one JSON object safely.
        """

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        if not isinstance(
            data,
            dict,
        ):
            raise ValueError(
                f"Expected JSON object: {path}"
            )

        return data

    # ========================================================
    # SAFE FLOAT
    # ========================================================

    @staticmethod
    def _safe_float(
        value: Any,
        default: float | None = None,
    ) -> float | None:
        """
        Convert arbitrary input to a finite float.

        NaN and Infinity are rejected.
        """

        try:
            if value is None:
                return default

            number = float(
                value
            )

            if not np.isfinite(
                number
            ):
                return default

            return float(number)

        except (
            TypeError,
            ValueError,
        ):
            return default

    # ========================================================
    # SAFE BOOLEAN
    # ========================================================

    @staticmethod
    def _normalize_bool(
        value: Any,
        default: bool = False,
    ) -> bool:
        """
        Normalize common boolean representations.

        Examples:

            True
            False
            "true"
            "false"
            1
            0
        """

        if isinstance(
            value,
            bool,
        ):
            return value

        if isinstance(
            value,
            (int, float),
        ):

            if value == 1:
                return True

            if value == 0:
                return False

        if isinstance(
            value,
            str,
        ):

            normalized = value.strip().casefold()

            if normalized in {
                "true",
                "yes",
                "y",
                "1",
            }:
                return True

            if normalized in {
                "false",
                "no",
                "n",
                "0",
            }:
                return False

        return default

    # ========================================================
    # NORMALIZE STRING
    # ========================================================

    @staticmethod
    def _normalize_string(
        value: Any,
    ) -> str | None:
        """
        Normalize a textual value.

        Empty strings become None.
        """

        if value is None:
            return None

        value = str(
            value
        ).strip()

        return value if value else None

    # ========================================================
    # NORMALIZE MONEY
    # ========================================================

    def _normalize_amount(
        self,
        value: Any,
    ) -> float | None:
        """
        Convert monetary values into JSON-safe numeric values.

        Supported examples:

            "40"
            "40.00"
            "₹40.00"
            "Rs. 40.00"
            "Rs 40.00"
            "INR 40.00"
            "1,250.50"

        Negative and non-finite values become None.
        """

        if value is None:
            return None

        if isinstance(
            value,
            str,
        ):

            cleaned = value.strip()

            cleaned = re.sub(
                r"(?i)(?:₹|rs\.?|inr)",
                "",
                cleaned,
            )

            cleaned = cleaned.replace(
                ",",
                "",
            )

            cleaned = cleaned.strip()

            value = cleaned

        number = self._safe_float(
            value
        )

        if number is None:
            return None

        if number < 0:
            return None

        return round(
            number,
            self.config.round_amount_digits,
        )

    # ========================================================
    # NORMALIZE CONFIDENCE
    # ========================================================

    def _normalize_confidence(
        self,
        value: Any,
    ) -> tuple[float, bool]:
        """
        Normalize confidence into [0, 1].

        Returns:

            (normalized_value, input_was_valid)

        Missing confidence is treated as 0.0 but does not
        count as an invalid numerical value.

        Out-of-range confidence is clipped and marked invalid.
        """

        if value is None:
            return (
                0.0,
                True,
            )

        number = self._safe_float(
            value
        )

        if number is None:
            return (
                0.0,
                False,
            )

        if not (
            0.0
            <= number
            <= 1.0
        ):
            clipped = float(
                np.clip(
                    number,
                    0.0,
                    1.0,
                )
            )

            return (
                round(
                    clipped,
                    self.config.round_confidence_digits,
                ),
                False,
            )

        return (
            round(
                number,
                self.config.round_confidence_digits,
            ),
            True,
        )

    # ========================================================
    # DERIVE RELIABILITY
    # ========================================================

    def _derive_reliability(
        self,
        confidence: float,
    ) -> str:
        """
        Derive reliability deterministically from confidence.
        """

        if (
            confidence
            >= self.config.medium_confidence_threshold
        ):
            return "high"

        if (
            confidence
            >= self.config.low_confidence_threshold
        ):
            return "medium"

        return "low"

    # ========================================================
    # NORMALIZE RELIABILITY
    # ========================================================

    def _normalize_reliability(
        self,
        value: Any,
        confidence: float,
    ) -> tuple[str, bool]:
        """
        Preserve valid upstream reliability.

        Invalid or missing reliability is derived from confidence.

        Returns:

            (reliability, input_was_valid)
        """

        if value is None:
            return (
                self._derive_reliability(
                    confidence
                ),
                True,
            )

        reliability = str(
            value
        ).strip().casefold()

        if reliability in self.ALLOWED_RELIABILITY:
            return (
                reliability,
                True,
            )

        return (
            self._derive_reliability(
                confidence
            ),
            False,
        )

    # ========================================================
    # NORMALIZE EXPECTED ITEM COUNT
    # ========================================================

    @staticmethod
    def _normalize_expected_item_count(
        value: Any,
    ) -> tuple[int | None, bool]:
        """
        Normalize expected item count.

        Valid range:
            0 ... 10000
        """

        if value is None:
            return (
                None,
                True,
            )

        try:
            number = float(
                value
            )

            if not np.isfinite(
                number
            ):
                return (
                    None,
                    False,
                )

            if not number.is_integer():
                return (
                    None,
                    False,
                )

            number_int = int(
                number
            )

            if not (
                0
                <= number_int
                <= 10000
            ):
                return (
                    None,
                    False,
                )

            return (
                number_int,
                True,
            )

        except (
            TypeError,
            ValueError,
        ):
            return (
                None,
                False,
            )

    # ========================================================
    # DATE VALIDATION
    # ========================================================

    @staticmethod
    def _validate_date_value(
        value: Any,
    ) -> tuple[bool, list[str]]:
        """
        Validate strict YYYY-MM-DD date format and calendar value.
        """

        warnings: list[str] = []

        if value is None:
            return (
                True,
                warnings,
            )

        text = str(
            value
        ).strip()

        if not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}",
            text,
        ):
            warnings.append(
                "date_format_invalid"
            )
            return (
                False,
                warnings,
            )

        try:
            datetime.strptime(
                text,
                "%Y-%m-%d",
            )

        except ValueError:
            warnings.append(
                "date_invalid_calendar_value"
            )

            return (
                False,
                warnings,
            )

        return (
            True,
            warnings,
        )

    # ========================================================
    # NORMALIZE ITEMS
    # ========================================================

    def _normalize_items(
        self,
        items: Any,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """
        Normalize extracted item records.

        Required item representation:

            {
                "name": "...",
                "price": 40.0
            }

        Confidence/reliability metadata is preserved whenever
        available from the confidence engine.
        """

        warnings: list[str] = []

        if items is None:
            return (
                [],
                warnings,
            )

        if not isinstance(
            items,
            list,
        ):
            warnings.append(
                "items_field_malformed"
            )

            return (
                [],
                warnings,
            )

        normalized_items: list[
            dict[str, Any]
        ] = []

        for index, item in enumerate(
            items
        ):

            if not isinstance(
                item,
                dict,
            ):
                warnings.append(
                    f"items[{index}]_not_object"
                )
                continue

            name = self._normalize_string(
                item.get(
                    "name"
                )
            )

            # ------------------------------------------------
            # Do not invent missing item names.
            # ------------------------------------------------

            if name is None:
                warnings.append(
                    f"items[{index}]_name_missing"
                )
                continue

            price_raw = item.get(
                "price"
            )

            price = self._normalize_amount(
                price_raw
            )

            # ------------------------------------------------
            # Preserve core item fields.
            # ------------------------------------------------

            normalized_item: dict[str, Any] = {
                "name": name,
                "price": price,
            }

            # ------------------------------------------------
            # Confidence
            # ------------------------------------------------

            confidence, confidence_valid = (
                self._normalize_confidence(
                    item.get(
                        "confidence"
                    )
                )
            )

            if (
                not confidence_valid
            ):
                warnings.append(
                    f"items[{index}]_confidence_invalid"
                )

            if self.config.include_confidence:

                normalized_item[
                    "confidence"
                ] = confidence

            # ------------------------------------------------
            # Reliability
            # ------------------------------------------------

            reliability, reliability_valid = (
                self._normalize_reliability(
                    item.get(
                        "reliability"
                    ),
                    confidence,
                )
            )

            if (
                not reliability_valid
            ):
                warnings.append(
                    f"items[{index}]_reliability_invalid"
                )

            if self.config.include_reliability:

                normalized_item[
                    "reliability"
                ] = reliability

            # ------------------------------------------------
            # Review flag
            # ------------------------------------------------

            if self.config.include_review_flag:

                normalized_item[
                    "review_required"
                ] = self._normalize_bool(
                    item.get(
                        "review_required"
                    ),
                    default=(
                        confidence
                        < self.config.low_confidence_threshold
                    ),
                )

            # ------------------------------------------------
            # Evidence
            # ------------------------------------------------

            if self.config.include_evidence:

                raw_evidence = item.get(
                    "evidence",
                    [],
                )

                if isinstance(
                    raw_evidence,
                    list,
                ):

                    normalized_item[
                        "evidence"
                    ] = [
                        str(
                            evidence
                        )
                        for evidence in raw_evidence
                        if str(
                            evidence
                        ).strip()
                    ]

                else:

                    normalized_item[
                        "evidence"
                    ] = []

                    warnings.append(
                        f"items[{index}]_evidence_malformed"
                    )

            # ------------------------------------------------
            # Extraction metadata
            # ------------------------------------------------

            if self.config.include_extraction_metadata:

                normalized_item[
                    "source_text"
                ] = self._normalize_string(
                    item.get(
                        "source_text"
                    )
                )

                normalized_item[
                    "extraction_method"
                ] = self._normalize_string(
                    item.get(
                        "extraction_method"
                    )
                )

                region_index = item.get(
                    "region_index"
                )

                if region_index is None:

                    normalized_item[
                        "region_index"
                    ] = None

                else:

                    try:

                        region_float = float(
                            region_index
                        )

                        if (
                            np.isfinite(
                                region_float
                            )
                            and region_float.is_integer()
                            and region_float >= 0
                        ):

                            normalized_item[
                                "region_index"
                            ] = int(
                                region_float
                            )

                        else:

                            normalized_item[
                                "region_index"
                            ] = None

                            warnings.append(
                                f"items[{index}]_region_index_invalid"
                            )

                    except (
                        TypeError,
                        ValueError,
                    ):

                        normalized_item[
                            "region_index"
                        ] = None

                        warnings.append(
                            f"items[{index}]_region_index_invalid"
                        )

            # ------------------------------------------------
            # Preserve detailed confidence components.
            # ------------------------------------------------

            if self.config.include_source_metadata:

                for confidence_name in (
                    "ocr_confidence",
                    "pattern_confidence",
                    "context_confidence",
                ):

                    component, component_valid = (
                        self._normalize_confidence(
                            item.get(
                                confidence_name
                            )
                        )
                    )

                    normalized_item[
                        confidence_name
                    ] = component

                    if not component_valid:
                        warnings.append(
                            f"items[{index}]_"
                            f"{confidence_name}_invalid"
                        )

                normalized_item[
                    "conflict_detected"
                ] = self._normalize_bool(
                    item.get(
                        "conflict_detected"
                    ),
                    default=False,
                )

                raw_warnings = item.get(
                    "warnings",
                    [],
                )

                if isinstance(
                    raw_warnings,
                    list,
                ):

                    normalized_item[
                        "warnings"
                    ] = [
                        str(
                            warning
                        )
                        for warning in raw_warnings
                        if str(
                            warning
                        ).strip()
                    ]

                else:

                    normalized_item[
                        "warnings"
                    ] = []

            normalized_items.append(
                normalized_item
            )

        return (
            normalized_items,
            warnings,
        )

    # ========================================================
    # BUILD CONFIDENCE FIELD
    # ========================================================

    def _confidence_field(
        self,
        field: dict[str, Any] | None,
        fallback_value: Any = None,
    ) -> tuple[dict[str, Any], list[str]]:
        """
        Build a stable confidence-aware representation for
        store_name, date, and total_amount.
        """

        warnings: list[str] = []

        if not isinstance(
            field,
            dict,
        ):
            field = {}

        value = field.get(
            "value",
            fallback_value,
        )

        if value is None:
            missing_value = True

        else:
            missing_value = False

        confidence, confidence_valid = (
            self._normalize_confidence(
                field.get(
                    "confidence"
                )
            )
        )

        if not confidence_valid:
            warnings.append(
                "field_confidence_invalid"
            )

        reliability, reliability_valid = (
            self._normalize_reliability(
                field.get(
                    "reliability"
                ),
                confidence,
            )
        )

        if not reliability_valid:
            warnings.append(
                "field_reliability_invalid"
            )

        result: dict[str, Any] = {
            "value": value,
        }

        if self.config.include_confidence:

            result[
                "confidence"
            ] = confidence

        if self.config.include_reliability:

            result[
                "reliability"
            ] = reliability

        if self.config.include_review_flag:

            result[
                "review_required"
            ] = self._normalize_bool(
                field.get(
                    "review_required"
                ),
                default=(
                    missing_value
                    or confidence
                    < self.config.low_confidence_threshold
                ),
            )

        if self.config.include_evidence:

            raw_evidence = field.get(
                "evidence",
                [],
            )

            if isinstance(
                raw_evidence,
                list,
            ):

                result[
                    "evidence"
                ] = [
                    str(
                        evidence
                    )
                    for evidence in raw_evidence
                    if str(
                        evidence
                    ).strip()
                ]

            else:

                result[
                    "evidence"
                ] = []

                warnings.append(
                    "field_evidence_malformed"
                )

        if self.config.include_source_metadata:

            for confidence_name in (
                "ocr_confidence",
                "pattern_confidence",
                "context_confidence",
            ):

                component, component_valid = (
                    self._normalize_confidence(
                        field.get(
                            confidence_name
                        )
                    )
                )

                result[
                    confidence_name
                ] = component

                if not component_valid:
                    warnings.append(
                        f"{confidence_name}_invalid"
                    )

            result[
                "conflict_detected"
            ] = self._normalize_bool(
                field.get(
                    "conflict_detected"
                ),
                default=False,
            )

        return (
            result,
            warnings,
        )

    # ========================================================
    # BUILD FINAL OUTPUT
    # ========================================================

    def _build_output(
        self,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Convert confidence-engine output into final assignment JSON.
        """

        generation_warnings: list[str] = []

        # ----------------------------------------------------
        # Receipt ID
        # ----------------------------------------------------

        receipt_id = self._normalize_string(
            data.get(
                "receipt_id"
            )
        )

        # ----------------------------------------------------
        # Confidence fields
        # ----------------------------------------------------

        fields = data.get(
            "fields",
            {},
        )

        if not isinstance(
            fields,
            dict,
        ):
            fields = {}

            generation_warnings.append(
                "fields_object_malformed"
            )

        store_field = fields.get(
            "store_name"
        )

        date_field = fields.get(
            "date"
        )

        total_field = fields.get(
            "total_amount"
        )

        # ----------------------------------------------------
        # Fallback values
        # ----------------------------------------------------

        fallback_store_name = (
            data.get(
                "store_name"
            )
            if data.get(
                "store_name"
            ) is not None
            else data.get(
                "vendor_name"
            )
        )

        fallback_date = (
            data.get(
                "date"
            )
            if data.get(
                "date"
            ) is not None
            else data.get(
                "transaction_date"
            )
        )

        fallback_total = data.get(
            "total_amount"
        )

        # ----------------------------------------------------
        # Extract field values
        # ----------------------------------------------------

        if isinstance(
            store_field,
            dict,
        ):

            store_name = store_field.get(
                "value",
                fallback_store_name,
            )

        else:

            store_name = fallback_store_name

        if isinstance(
            date_field,
            dict,
        ):

            transaction_date = date_field.get(
                "value",
                fallback_date,
            )

        else:

            transaction_date = fallback_date

        if isinstance(
            total_field,
            dict,
        ):

            total_amount = total_field.get(
                "value",
                fallback_total,
            )

        else:

            total_amount = fallback_total

        # ----------------------------------------------------
        # Normalize fields
        # ----------------------------------------------------

        store_name = self._normalize_string(
            store_name
        )

        transaction_date = self._normalize_string(
            transaction_date
        )

        total_amount = self._normalize_amount(
            total_amount
        )

        # ----------------------------------------------------
        # Validate date
        # ----------------------------------------------------

        date_valid, date_warnings = (
            self._validate_date_value(
                transaction_date
            )
        )

        if not date_valid:

            generation_warnings.extend(
                date_warnings
            )

        # ----------------------------------------------------
        # Normalize items
        # ----------------------------------------------------

        items, item_warnings = (
            self._normalize_items(
                data.get(
                    "items",
                    [],
                )
            )
        )

        generation_warnings.extend(
            item_warnings
        )

        # ----------------------------------------------------
        # Expected item count
        #
        # Preserve the distinction between:
        #
        # expected_item_count
        #     receipt-reported count
        #
        # item_count
        #     successfully normalized named items
        # ----------------------------------------------------

        expected_raw = data.get(
            "expected_item_count"
        )

        if expected_raw is None:

            metadata = data.get(
                "metadata"
            )

            if isinstance(
                metadata,
                dict,
            ):
                expected_raw = metadata.get(
                    "expected_item_count"
                )

        expected_item_count, expected_count_valid = (
            self._normalize_expected_item_count(
                expected_raw
            )
        )

        if not expected_count_valid:

            generation_warnings.append(
                "expected_item_count_invalid"
            )

        item_count = len(
            items
        )

        # ----------------------------------------------------
        # Overall confidence
        # ----------------------------------------------------

        overall_confidence, overall_confidence_valid = (
            self._normalize_confidence(
                data.get(
                    "overall_confidence"
                )
            )
        )

        if not overall_confidence_valid:

            generation_warnings.append(
                "overall_confidence_invalid"
            )

        # ----------------------------------------------------
        # Overall reliability
        # ----------------------------------------------------

        reliability, reliability_valid = (
            self._normalize_reliability(
                data.get(
                    "reliability"
                ),
                overall_confidence,
            )
        )

        if not reliability_valid:

            generation_warnings.append(
                "overall_reliability_invalid"
            )

        # ----------------------------------------------------
        # Review flag
        # ----------------------------------------------------

        confidence_review = (
            overall_confidence
            < self.config.low_confidence_threshold
        )

        field_review = False

        for field_value in (
            store_field,
            date_field,
            total_field,
        ):

            if isinstance(
                field_value,
                dict,
            ):

                if self._normalize_bool(
                    field_value.get(
                        "review_required"
                    ),
                    default=False,
                ):
                    field_review = True

        source_review_required = (
            self._normalize_bool(
                data.get(
                    "review_required"
                ),
                default=False,
            )
        )

        review_required = (
            confidence_review
            or field_review
            or source_review_required
        )

        # ----------------------------------------------------
        # Build confidence-aware field objects
        # ----------------------------------------------------

        store_confidence, store_confidence_warnings = (
            self._confidence_field(
                store_field,
                fallback_value=store_name,
            )
        )

        date_confidence, date_confidence_warnings = (
            self._confidence_field(
                date_field,
                fallback_value=transaction_date,
            )
        )

        total_confidence, total_confidence_warnings = (
            self._confidence_field(
                total_field,
                fallback_value=total_amount,
            )
        )

        generation_warnings.extend(
            store_confidence_warnings
        )

        generation_warnings.extend(
            date_confidence_warnings
        )

        generation_warnings.extend(
            total_confidence_warnings
        )

        # ----------------------------------------------------
        # Assignment-facing core schema
        # ----------------------------------------------------

        output: dict[str, Any] = {
            "store_name": store_name,

            "date": transaction_date,

            "items": items,

            "total_amount": total_amount,

            "expected_item_count": expected_item_count,
        }

        # ----------------------------------------------------
        # Confidence metadata
        # ----------------------------------------------------

        if self.config.include_confidence:

            output[
                "confidence"
            ] = {
                "store_name": store_confidence,
                "date": date_confidence,
                "total_amount": total_confidence,
            }

        if self.config.include_confidence:

            output[
                "overall_confidence"
            ] = overall_confidence

        if self.config.include_reliability:

            output[
                "reliability"
            ] = reliability

        if self.config.include_review_flag:

            output[
                "review_required"
            ] = review_required

        # ----------------------------------------------------
        # Conflicts and warnings
        # ----------------------------------------------------

        source_conflicts = data.get(
            "conflicts",
            [],
        )

        if isinstance(
            source_conflicts,
            list,
        ):

            conflicts = [
                str(
                    conflict
                )
                for conflict in source_conflicts
                if str(
                    conflict
                ).strip()
            ]

        else:

            conflicts = []

            generation_warnings.append(
                "conflicts_field_malformed"
            )

        source_warnings = data.get(
            "warnings",
            [],
        )

        if isinstance(
            source_warnings,
            list,
        ):

            warnings = [
                str(
                    warning
                )
                for warning in source_warnings
                if str(
                    warning
                ).strip()
            ]

        else:

            warnings = []

            generation_warnings.append(
                "warnings_field_malformed"
            )

        warnings.extend(
            generation_warnings
        )

        # De-duplicate while preserving deterministic ordering.
        conflicts = sorted(
            set(conflicts)
        )

        warnings = sorted(
            set(warnings)
        )

        if self.config.include_source_metadata:

            output[
                "conflicts"
            ] = conflicts

            output[
                "warnings"
            ] = warnings

        # ----------------------------------------------------
        # Metadata / provenance
        # ----------------------------------------------------

        if self.config.include_source_metadata:

            item_sum = None

            source_item_sum = data.get(
                "item_sum"
            )

            if source_item_sum is None:

                item_sum = (
                    round(
                        sum(
                            item["price"]
                            for item in items
                            if item.get(
                                "price"
                            ) is not None
                        ),
                        self.config.round_amount_digits,
                    )
                    if any(
                        item.get(
                            "price"
                        ) is not None
                        for item in items
                    )
                    else None
                )

            else:

                item_sum = self._normalize_amount(
                    source_item_sum
                )

            item_total_difference = self._safe_float(
                data.get(
                    "item_total_difference"
                )
            )

            if item_total_difference is not None:

                item_total_difference = round(
                    item_total_difference,
                    self.config.round_amount_digits,
                )

            output[
                "metadata"
            ] = {
                "schema_version": self.config.schema_version,

                "receipt_id": receipt_id,

                "source_extraction_file": (
                    self._normalize_string(
                        data.get(
                            "input_path"
                        )
                    )
                ),

                "source_status": (
                    self._normalize_string(
                        data.get(
                            "status"
                        )
                    )
                ),

                "generated_at_utc": (
                    datetime.now().astimezone().isoformat()
                ),

                "item_count": item_count,

                "expected_item_count": expected_item_count,

                "item_sum": item_sum,

                "item_total_difference": (
                    item_total_difference
                ),
            }

        return output

    # ========================================================
    # SCHEMA VALIDATION
    # ========================================================

    def _validate_schema(
        self,
        output: dict[str, Any],
    ) -> list[str]:
        """
        Validate the final JSON schema and value constraints.
        """

        errors: list[str] = []

        # ====================================================
        # TOP LEVEL REQUIRED FIELDS
        # ====================================================

        for field in self.REQUIRED_TOP_LEVEL_FIELDS:

            if field not in output:

                errors.append(
                    f"Missing required field: {field}"
                )

        # ====================================================
        # STORE NAME
        # ====================================================

        if (
            "store_name" in output
            and output[
                "store_name"
            ] is not None
            and not isinstance(
                output[
                    "store_name"
                ],
                str,
            )
        ):

            errors.append(
                "store_name must be string or null."
            )

        # ====================================================
        # DATE
        # ====================================================

        if (
            "date" in output
            and output[
                "date"
            ] is not None
        ):

            date_value = output[
                "date"
            ]

            if not isinstance(
                date_value,
                str,
            ):

                errors.append(
                    "date must be string or null."
                )

            else:

                date_valid, date_errors = (
                    self._validate_date_value(
                        date_value
                    )
                )

                if not date_valid:

                    errors.extend(
                        [
                            f"date: {error}"
                            for error in date_errors
                        ]
                    )

        # ====================================================
        # ITEMS
        # ====================================================

        if not isinstance(
            output.get(
                "items"
            ),
            list,
        ):

            errors.append(
                "items must be a list."
            )

        else:

            for index, item in enumerate(
                output[
                    "items"
                ]
            ):

                if not isinstance(
                    item,
                    dict,
                ):

                    errors.append(
                        f"items[{index}] must be an object."
                    )

                    continue

                # ------------------------------------------------
                # Name
                # ------------------------------------------------

                if (
                    "name" not in item
                    or not isinstance(
                        item.get(
                            "name"
                        ),
                        str,
                    )
                    or not item.get(
                        "name"
                    ).strip()
                ):

                    errors.append(
                        f"items[{index}].name "
                        "must be a non-empty string."
                    )

                # ------------------------------------------------
                # Price
                # ------------------------------------------------

                price = item.get(
                    "price"
                )

                if price is not None:

                    if not isinstance(
                        price,
                        (int, float),
                    ):

                        errors.append(
                            f"items[{index}].price "
                            "must be numeric or null."
                        )

                    else:

                        price_float = self._safe_float(
                            price
                        )

                        if price_float is None:

                            errors.append(
                                f"items[{index}].price "
                                "must be finite."
                            )

                        elif price_float < 0:

                            errors.append(
                                f"items[{index}].price "
                                "cannot be negative."
                            )

                # ------------------------------------------------
                # Item confidence
                # ------------------------------------------------

                if (
                    "confidence" in item
                ):

                    confidence = item.get(
                        "confidence"
                    )

                    if (
                        not isinstance(
                            confidence,
                            (int, float),
                        )
                        or not np.isfinite(
                            float(
                                confidence
                            )
                        )
                        or not (
                            0.0
                            <= float(
                                confidence
                            )
                            <= 1.0
                        )
                    ):

                        errors.append(
                            f"items[{index}].confidence "
                            "must be between 0 and 1."
                        )

                # ------------------------------------------------
                # Item reliability
                # ------------------------------------------------

                if (
                    "reliability" in item
                    and item.get(
                        "reliability"
                    ) not in self.ALLOWED_RELIABILITY
                ):

                    errors.append(
                        f"items[{index}].reliability "
                        "must be high, medium, or low."
                    )

                # ------------------------------------------------
                # Review flag
                # ------------------------------------------------

                if (
                    "review_required" in item
                    and not isinstance(
                        item.get(
                            "review_required"
                        ),
                        bool,
                    )
                ):

                    errors.append(
                        f"items[{index}].review_required "
                        "must be boolean."
                    )

                # ------------------------------------------------
                # Detailed confidence components
                # ------------------------------------------------

                for component in (
                    "ocr_confidence",
                    "pattern_confidence",
                    "context_confidence",
                ):

                    if component not in item:
                        continue

                    component_value = item.get(
                        component
                    )

                    if (
                        not isinstance(
                            component_value,
                            (int, float),
                        )
                        or not np.isfinite(
                            float(
                                component_value
                            )
                        )
                        or not (
                            0.0
                            <= float(
                                component_value
                            )
                            <= 1.0
                        )
                    ):

                        errors.append(
                            f"items[{index}].{component} "
                            "must be between 0 and 1."
                        )

        # ====================================================
        # TOTAL AMOUNT
        # ====================================================

        total_amount = output.get(
            "total_amount"
        )

        if total_amount is not None:

            if not isinstance(
                total_amount,
                (int, float),
            ):

                errors.append(
                    "total_amount must be numeric or null."
                )

            else:

                total_float = self._safe_float(
                    total_amount
                )

                if total_float is None:

                    errors.append(
                        "total_amount must be finite."
                    )

                elif total_float < 0:

                    errors.append(
                        "total_amount cannot be negative."
                    )

        # ====================================================
        # EXPECTED ITEM COUNT
        # ====================================================

        if (
            "expected_item_count" in output
            and output[
                "expected_item_count"
            ] is not None
        ):

            expected_count = output[
                "expected_item_count"
            ]

            if not isinstance(
                expected_count,
                int,
            ):

                errors.append(
                    "expected_item_count must be integer or null."
                )

            elif not (
                0
                <= expected_count
                <= 10000
            ):

                errors.append(
                    "expected_item_count must be between "
                    "0 and 10000."
                )

        # ====================================================
        # OVERALL CONFIDENCE
        # ====================================================

        if (
            "overall_confidence" in output
        ):

            overall_confidence = output[
                "overall_confidence"
            ]

            if (
                not isinstance(
                    overall_confidence,
                    (int, float),
                )
                or not np.isfinite(
                    float(
                        overall_confidence
                    )
                )
                or not (
                    0.0
                    <= float(
                        overall_confidence
                    )
                    <= 1.0
                )
            ):

                errors.append(
                    "overall_confidence must be "
                    "between 0 and 1."
                )

        # ====================================================
        # OVERALL RELIABILITY
        # ====================================================

        if (
            "reliability" in output
            and output[
                "reliability"
            ] not in self.ALLOWED_RELIABILITY
        ):

            errors.append(
                "reliability must be "
                "high, medium, or low."
            )

        # ====================================================
        # OVERALL REVIEW FLAG
        # ====================================================

        if (
            "review_required" in output
            and not isinstance(
                output[
                    "review_required"
                ],
                bool,
            )
        ):

            errors.append(
                "review_required must be boolean."
            )

        # ====================================================
        # CONFIDENCE OBJECT
        # ====================================================

        if (
            "confidence" in output
        ):

            confidence_object = output[
                "confidence"
            ]

            if not isinstance(
                confidence_object,
                dict,
            ):

                errors.append(
                    "confidence must be an object."
                )

            else:

                for field_name in (
                    "store_name",
                    "date",
                    "total_amount",
                ):

                    if field_name not in confidence_object:
                        continue

                    field_object = confidence_object[
                        field_name
                    ]

                    if not isinstance(
                        field_object,
                        dict,
                    ):

                        errors.append(
                            f"confidence.{field_name} "
                            "must be an object."
                        )

                        continue

                    field_confidence = field_object.get(
                        "confidence"
                    )

                    if field_confidence is not None:

                        if (
                            not isinstance(
                                field_confidence,
                                (int, float),
                            )
                            or not np.isfinite(
                                float(
                                    field_confidence
                                )
                            )
                            or not (
                                0.0
                                <= float(
                                    field_confidence
                                )
                                <= 1.0
                            )
                        ):

                            errors.append(
                                f"confidence.{field_name}."
                                "confidence must be between 0 and 1."
                            )

                    field_reliability = field_object.get(
                        "reliability"
                    )

                    if (
                        field_reliability is not None
                        and field_reliability
                        not in self.ALLOWED_RELIABILITY
                    ):

                        errors.append(
                            f"confidence.{field_name}."
                            "reliability must be high, medium, or low."
                        )

                    field_review = field_object.get(
                        "review_required"
                    )

                    if (
                        field_review is not None
                        and not isinstance(
                            field_review,
                            bool,
                        )
                    ):

                        errors.append(
                            f"confidence.{field_name}."
                            "review_required must be boolean."
                        )

        return sorted(
            set(errors)
        )

    # ========================================================
    # VALIDATION FAILURE BUILDER
    # ========================================================

    @staticmethod
    def _attach_validation(
        output: dict[str, Any],
        status: str,
        errors: list[str],
    ) -> None:
        """
        Attach validation status to output.
        """

        output[
            "_validation"
        ] = {
            "status": status,
            "errors": sorted(
                set(
                    errors
                )
            ),
        }

    # ========================================================
    # PROCESS SINGLE FILE
    # ========================================================

    def process_file(
        self,
        json_path: str | Path,
    ) -> dict[str, Any]:
        """
        Process one confidence JSON file.
        """

        start = time.perf_counter()

        json_path = Path(
            json_path
        )

        try:

            # ------------------------------------------------
            # Load source
            # ------------------------------------------------

            data = self._load_json(
                json_path
            )

            # ------------------------------------------------
            # Build final output
            # ------------------------------------------------

            output = self._build_output(
                data
            )

            # ------------------------------------------------
            # Validate final schema
            # ------------------------------------------------

            validation_errors = (
                self._validate_schema(
                    output
                )
            )

            if validation_errors:

                self._attach_validation(
                    output,
                    status="failed",
                    errors=validation_errors,
                )

                logger.error(
                    "JSON schema validation failed for %s: %s",
                    json_path,
                    validation_errors,
                )

            else:

                self._attach_validation(
                    output,
                    status="passed",
                    errors=[],
                )

            # ------------------------------------------------
            # Processing metadata
            # ------------------------------------------------

            processing_time_ms = round(
                (
                    time.perf_counter()
                    - start
                )
                * 1000.0,
                3,
            )

            output[
                "_processing"
            ] = {
                "processing_time_ms": processing_time_ms,

                "generated_at_utc": (
                    datetime.now().astimezone().isoformat()
                ),
            }

            return output

        except Exception as exc:

            logger.exception(
                "JSON generation failed for %s",
                json_path,
            )

            processing_time_ms = round(
                (
                    time.perf_counter()
                    - start
                )
                * 1000.0,
                3,
            )

            # ------------------------------------------------
            # Fail-safe output.
            #
            # Required assignment fields are still present.
            # Nothing is fabricated.
            # ------------------------------------------------

            output = {
                "store_name": None,

                "date": None,

                "items": [],

                "total_amount": None,

                "expected_item_count": None,

                "confidence": {},

                "overall_confidence": 0.0,

                "reliability": "low",

                "review_required": True,

                "conflicts": [
                    "json_generation_exception"
                ],

                "warnings": [
                    str(exc)
                ],

                "_validation": {
                    "status": "failed",

                    "errors": [
                        str(exc)
                    ],
                },

                "_processing": {
                    "processing_time_ms": (
                        processing_time_ms
                    ),

                    "generated_at_utc": (
                        datetime.now().astimezone().isoformat()
                    ),
                },
            }

            return output

    # ========================================================
    # SAVE JSON
    # ========================================================

    def _save_json(
        self,
        output: dict[str, Any],
        receipt_id: str,
    ) -> Path:
        """
        Save one final JSON artifact.

        allow_nan=False guarantees that invalid floating-point
        values cannot be written to the final JSON document.
        """

        safe_receipt_id = (
            self._normalize_string(
                receipt_id
            )
            or "unknown_receipt"
        )

        output_path = (
            self.output_dir
            / f"{safe_receipt_id}.json"
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                output,
                file,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )

        return output_path

    # ========================================================
    # GET RECEIPT ID
    # ========================================================

    def _get_receipt_id(
        self,
        output: dict[str, Any],
        fallback_path: Path,
    ) -> str:
        """
        Resolve a stable receipt ID.
        """

        metadata = output.get(
            "metadata",
            {}
        )

        if isinstance(
            metadata,
            dict,
        ):

            receipt_id = self._normalize_string(
                metadata.get(
                    "receipt_id"
                )
            )

            if receipt_id:
                return receipt_id

        return (
            self._normalize_string(
                output.get(
                    "receipt_id"
                )
            )
            or fallback_path.stem
        )

    # ========================================================
    # BATCH PROCESSING
    # ========================================================

    def process_batch(
        self,
    ) -> dict[str, Any]:
        """
        Process all confidence-result JSON files.

        Duplicate receipt IDs within this batch are rejected
        instead of overwriting one another.
        """

        logger.info(
            "Starting final JSON generation..."
        )

        files = self._discover_files()

        results: list[
            dict[str, Any]
        ] = []

        successful = 0

        failed = 0

        review_required = 0

        processing_times: list[
            float
        ] = []

        seen_receipt_ids: set[
            str
        ] = set()

        # ----------------------------------------------------
        # Process every receipt.
        # ----------------------------------------------------

        for index, json_path in enumerate(
            files,
            start=1,
        ):

            output = self.process_file(
                json_path
            )

            receipt_id = self._get_receipt_id(
                output,
                json_path,
            )

            duplicate_id = (
                receipt_id
                in seen_receipt_ids
            )

            if duplicate_id:

                # --------------------------------------------
                # DO NOT OVERWRITE.
                # --------------------------------------------

                duplicate_error = (
                    "Duplicate receipt_id detected "
                    f"within batch: {receipt_id}"
                )

                output[
                    "warnings"
                ] = sorted(
                    set(
                        output.get(
                            "warnings",
                            [],
                        )
                        + [
                            duplicate_error
                        ]
                    )
                )

                self._attach_validation(
                    output,
                    status="failed",
                    errors=[
                        duplicate_error
                    ],
                )

                save_succeeded = False

                output_path: Path | None = None

                failed += 1

            else:

                seen_receipt_ids.add(
                    receipt_id
                )

                # --------------------------------------------
                # Save exactly once.
                # --------------------------------------------

                try:

                    output_path = (
                        self._save_json(
                            output,
                            receipt_id,
                        )
                    )

                    save_succeeded = True

                    logger.info(
                        "Final JSON saved: %s",
                        output_path,
                    )

                except Exception as exc:

                    save_succeeded = False

                    output_path = None

                    failed += 1

                    logger.exception(
                        "Could not save final JSON for %s",
                        receipt_id,
                    )

                    output[
                        "warnings"
                    ] = sorted(
                        set(
                            output.get(
                                "warnings",
                                [],
                            )
                            + [
                                f"Save failed: {exc}"
                            ]
                        )
                    )

                    self._attach_validation(
                        output,
                        status="failed",
                        errors=[
                            f"Save failed: {exc}"
                        ],
                    )

            # ------------------------------------------------
            # Count success ONLY ONCE.
            # ------------------------------------------------

            validation_status = (
                output.get(
                    "_validation",
                    {},
                ).get(
                    "status"
                )
            )

            if (
                validation_status == "passed"
                and save_succeeded
                and not duplicate_id
            ):

                successful += 1

                receipt_status = "success"

            else:

                if not duplicate_id and save_succeeded:
                    failed += 1

                receipt_status = "failed"

            # ------------------------------------------------
            # Review count.
            # ------------------------------------------------

            needs_review = self._normalize_bool(
                output.get(
                    "review_required"
                ),
                default=True,
            )

            if needs_review:
                review_required += 1

            # ------------------------------------------------
            # Processing time.
            # ------------------------------------------------

            processing_time = self._safe_float(
                output.get(
                    "_processing",
                    {},
                ).get(
                    "processing_time_ms",
                    0.0,
                )
                if isinstance(
                    output.get(
                        "_processing",
                        {}
                    ),
                    dict,
                )
                else 0.0
            )

            if processing_time is None:
                processing_time = 0.0

            processing_times.append(
                float(
                    processing_time
                )
            )

            # ------------------------------------------------
            # Batch result record.
            # ------------------------------------------------

            results.append(
                {
                    "receipt_id": receipt_id,

                    "status": receipt_status,

                    "output_path": (
                        str(
                            output_path
                        )
                        if output_path is not None
                        else None
                    ),

                    "overall_confidence": (
                        output.get(
                            "overall_confidence",
                            0.0,
                        )
                    ),

                    "review_required": needs_review,

                    "validation_status": (
                        validation_status
                    ),
                }
            )

            if index % 100 == 0:

                logger.info(
                    "Generated JSON for %d/%d receipts.",
                    index,
                    len(files),
                )

        # ----------------------------------------------------
        # Batch totals.
        # ----------------------------------------------------

        total = len(
            files
        )

        # ----------------------------------------------------
        # Rates.
        # ----------------------------------------------------

        success_rate = (
            successful / total
            if total
            else 0.0
        )

        review_rate = (
            review_required / total
            if total
            else 0.0
        )

        average_processing_time_ms = (
            float(
                np.mean(
                    processing_times
                )
            )
            if processing_times
            else 0.0
        )

        # ----------------------------------------------------
        # Manifest.
        # ----------------------------------------------------

        manifest: dict[str, Any] = {
            "schema_version": (
                self.config.schema_version
            ),

            "generated_at_utc": (
                datetime.now().astimezone().isoformat()
            ),

            "input_directory": str(
                self.input_dir.resolve()
            ),

            "output_directory": str(
                self.output_dir.resolve()
            ),

            "total_receipts": total,

            "successful": successful,

            "failed": failed,

            "review_required": review_required,

            "success_rate": success_rate,

            "review_rate": review_rate,

            "average_processing_time_ms": (
                round(
                    average_processing_time_ms,
                    3,
                )
            ),

            "results": results,
        }

        # ----------------------------------------------------
        # Save manifest.
        #
        # IMPORTANT:
        # manifest_path is added BEFORE json.dump().
        # Therefore the saved manifest contains it too.
        # ----------------------------------------------------

        if self.config.save_manifest:

            manifest_path = (
                self.output_dir
                / "json_manifest.json"
            )

            manifest[
                "manifest_path"
            ] = str(
                manifest_path
            )

            with manifest_path.open(
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    manifest,
                    file,
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=False,
                )

            logger.info(
                "JSON manifest saved: %s",
                manifest_path,
            )

        logger.info(
            "Final JSON generation completed. "
            "Success=%d, Failed=%d, Review=%d",
            successful,
            failed,
            review_required,
        )

        return manifest

    # ========================================================
    # PUBLIC ENTRY POINT
    # ========================================================

    def run(
        self,
        json_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """
        Process a single file or the complete batch.
        """

        if json_path is not None:

            json_path = Path(
                json_path
            )

            output = self.process_file(
                json_path
            )

            receipt_id = self._get_receipt_id(
                output,
                json_path,
            )

            self._save_json(
                output,
                receipt_id,
            )

            return output

        return self.process_batch()