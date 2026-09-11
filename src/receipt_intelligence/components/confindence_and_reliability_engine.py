from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
from src.receipt_intelligence.entity.config_entity import ConfidenceEngineConfig
from src.receipt_intelligence import logger


# ============================================================
# FIELD CONFIDENCE
# ============================================================

@dataclass
class FieldConfidence:
    """
    Final confidence object for one field.
    """

    field_name: str

    value: Any

    confidence: float

    reliability: str

    review_required: bool

    ocr_confidence: float

    extraction_confidence: float

    pattern_confidence: float

    context_confidence: float

    source_text: str | None

    region_index: int | None

    conflict_detected: bool

    evidence: list[str]

    warnings: list[str]


# ============================================================
# ITEM CONFIDENCE
# ============================================================

@dataclass
class ItemConfidence:
    """
    Confidence-aware extracted line item.
    """

    name: str

    price: float | None

    confidence: float

    reliability: str

    review_required: bool

    ocr_confidence: float

    extraction_confidence: float

    pattern_confidence: float

    context_confidence: float

    source_text: str | None

    region_index: int | None

    evidence: list[str]

    warnings: list[str]


# ============================================================
# RECEIPT RESULT
# ============================================================

@dataclass
class ReceiptConfidenceResult:
    """
    Complete confidence-aware receipt result.
    """

    receipt_id: str

    status: str

    overall_confidence: float

    reliability: str

    review_required: bool

    expected_item_count: int | None

    extracted_item_count: int

    fields: dict[str, FieldConfidence]

    items: list[ItemConfidence]

    conflicts: list[str]

    warnings: list[str]

    financial_consistency: dict[str, Any]

    source_files: dict[str, str | None]

    processing_time_ms: float

    timestamp_utc: str


# ============================================================
# ENGINE
# ============================================================

class ConfidenceEngine:
    """
    Production-grade confidence engine for the exact receipt
    artifacts currently produced by this project.

    Main flow:

        Field Extraction
              +
        Cleaned OCR Regions
              +
        Raw OCR
              ↓
        Source Evidence Matching
              ↓
        OCR Confidence
              ↓
        Pattern Validation
              ↓
        Context / Heuristic Validation
              ↓
        Conflict Detection
              ↓
        Business Consistency
              ↓
        Weighted Confidence
              ↓
        Reliability
              ↓
        Review Queue Flag
    """

    # --------------------------------------------------------
    # Known receipt count pattern.
    # --------------------------------------------------------

    ITEM_COUNT_PATTERN = re.compile(
        r"(?i)"
        r"(?:#?\s*ITEMS?\s*SOLD)"
        r"\s*[:#-]?\s*"
        r"(\d+)"
    )

    # --------------------------------------------------------
    # Date validation.
    # --------------------------------------------------------

    DATE_PATTERN = re.compile(
        r"^\d{4}-\d{2}-\d{2}$"
    )

    # --------------------------------------------------------
    # Common financial labels.
    # --------------------------------------------------------

    TOTAL_KEYWORDS = (
        "TOTAL",
        "GRAND TOTAL",
        "TOTAL AMOUNT",
        "AMOUNT PAYABLE",
        "NET AMOUNT",
        "TOTAL DUE",
        "BALANCE DUE",
    )

    # ========================================================
    # INIT
    # ========================================================

    def __init__(
        self,
        config: ConfidenceEngineConfig
    ):
        self.config = config

        self.input_dir = Path(
            config.input_dir
        )

        self.ocr_dir = Path(
            config.ocr_dir
        )

        self.cleaned_ocr_dir = Path(
            config.cleaned_ocr_dir
        )

        self.output_dir = Path(
            config.output_dir
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self._validate_config()

    # ========================================================
    # CONFIG VALIDATION
    # ========================================================

    def _validate_config(self) -> None:

        weights = (
            self.config.ocr_weight
            + self.config.pattern_weight
            + self.config.context_weight
        )

        if not np.isclose(
            weights,
            1.0,
            atol=1e-6
        ):
            raise ValueError(
                "ocr_weight + pattern_weight + "
                "context_weight must equal 1.0."
            )

        if not (
            0.0
            <= self.config.low_confidence_threshold
            < self.config.medium_confidence_threshold
            <= 1.0
        ):
            raise ValueError(
                "Confidence thresholds must satisfy "
                "0 <= low < medium <= 1."
            )

        if not (
            0.0
            <= self.config.conflict_penalty
            <= 1.0
        ):
            raise ValueError(
                "conflict_penalty must be between 0 and 1."
            )

        if not (
            0.0
            <= self.config.item_count_mismatch_penalty
            <= 1.0
        ):
            raise ValueError(
                "item_count_mismatch_penalty must be "
                "between 0 and 1."
            )

    # ========================================================
    # JSON
    # ========================================================

    @staticmethod
    def _load_json(
        path: Path
    ) -> dict[str, Any]:

        with path.open(
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(
                file
            )

        if not isinstance(
            data,
            dict
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
        default: float = 0.0
    ) -> float:

        try:

            number = float(
                value
            )

            if not np.isfinite(
                number
            ):
                return default

            return float(
                number
            )

        except (
            TypeError,
            ValueError
        ):

            return default

    # ========================================================
    # BOUNDED FLOAT
    # ========================================================

    @staticmethod
    def _bound(
        value: float
    ) -> float:

        return float(
            np.clip(
                value,
                0.0,
                1.0
            )
        )

    # ========================================================
    # TEXT NORMALIZATION
    # ========================================================

    @staticmethod
    def _normalize_text(
        value: Any
    ) -> str:

        if value is None:
            return ""

        value = str(
            value
        ).strip()

        value = re.sub(
            r"\s+",
            " ",
            value
        )

        return value.casefold()

    # ========================================================
    # DISCOVER EXTRACTION FILES
    # ========================================================

    def _discover_extraction_files(
        self
    ) -> list[Path]:

        if not self.input_dir.exists():

            raise FileNotFoundError(
                f"Extraction directory does not exist: "
                f"{self.input_dir}"
            )

        files = sorted(
            self.input_dir.rglob(
                "*.json"
            )
        )

        ignored = {
            "extraction_manifest.json",
            "confidence_manifest.json",
        }

        files = [
            file
            for file in files
            if file.name not in ignored
        ]

        logger.info(
            "Discovered %d extraction files.",
            len(files)
        )

        return files

    # ========================================================
    # OCR SOURCE PATH
    # ========================================================

    def _ocr_source_path(
        self,
        receipt_id: str
    ) -> Path | None:

        path = (
            self.ocr_dir
            / f"{receipt_id}.json"
        )

        return (
            path
            if path.exists()
            else None
        )

    # ========================================================
    # CLEANED OCR PATH
    # ========================================================

    def _cleaned_ocr_source_path(
        self,
        receipt_id: str
    ) -> Path | None:

        path = (
            self.cleaned_ocr_dir
            / f"{receipt_id}.json"
        )

        return (
            path
            if path.exists()
            else None
        )

    # ========================================================
    # LOAD OCR SOURCES
    # ========================================================

    def _load_sources(
        self,
        receipt_id: str
    ) -> tuple[
        dict[str, Any] | None,
        dict[str, Any] | None,
        dict[str, str | None]
    ]:

        raw_path = (
            self._ocr_source_path(
                receipt_id
            )
        )

        cleaned_path = (
            self._cleaned_ocr_source_path(
                receipt_id
            )
        )

        raw_ocr = None

        cleaned_ocr = None

        if raw_path:

            try:
                raw_ocr = self._load_json(
                    raw_path
                )
            except Exception as exc:

                logger.warning(
                    "Could not load raw OCR for %s: %s",
                    receipt_id,
                    exc
                )

        if cleaned_path:

            try:
                cleaned_ocr = self._load_json(
                    cleaned_path
                )
            except Exception as exc:

                logger.warning(
                    "Could not load cleaned OCR for %s: %s",
                    receipt_id,
                    exc
                )

        return (
            raw_ocr,
            cleaned_ocr,
            {
                "raw_ocr": (
                    str(raw_path)
                    if raw_path
                    else None
                ),
                "cleaned_ocr": (
                    str(cleaned_path)
                    if cleaned_path
                    else None
                ),
            }
        )

    # ========================================================
    # OCR REGIONS
    # ========================================================

    @staticmethod
    def _regions(
        cleaned_ocr: dict[str, Any] | None
    ) -> list[dict[str, Any]]:

        if not cleaned_ocr:
            return []

        regions = cleaned_ocr.get(
            "regions",
            []
        )

        if not isinstance(
            regions,
            list
        ):
            return []

        return [
            region
            for region in regions
            if isinstance(
                region,
                dict
            )
        ]

    # ========================================================
    # RAW OCR REGIONS
    # ========================================================

    @staticmethod
    def _raw_regions(
        raw_ocr: dict[str, Any] | None
    ) -> list[dict[str, Any]]:

        if not raw_ocr:
            return []

        regions = raw_ocr.get(
            "regions",
            []
        )

        if not isinstance(
            regions,
            list
        ):
            return []

        return [
            region
            for region in regions
            if isinstance(
                region,
                dict
            )
        ]

    # ========================================================
    # REGION BY INDEX
    # ========================================================

    def _region_by_index(
        self,
        regions: list[dict[str, Any]],
        index: int | None
    ) -> dict[str, Any] | None:

        if index is None:
            return None

        try:

            index = int(
                index
            )

        except (
            TypeError,
            ValueError
        ):

            return None

        if (
            index < 0
            or index >= len(regions)
        ):
            return None

        return regions[index]

    # ========================================================
    # REGION TEXT MATCHING
    # ========================================================

    def _find_best_region(
        self,
        value: Any,
        source_text: Any,
        regions: list[dict[str, Any]],
        region_index: int | None = None
    ) -> tuple[
        float,
        dict[str, Any] | None,
        str
    ]:

        # ----------------------------------------------------
        # First trust the extraction region index.
        # In the current project this is the OCR reading-order
        # index produced before field extraction.
        # ----------------------------------------------------

        indexed_region = (
            self._region_by_index(
                regions,
                region_index
            )
        )

        normalized_value = (
            self._normalize_text(
                value
            )
        )

        normalized_source = (
            self._normalize_text(
                source_text
            )
        )

        if indexed_region:

            indexed_text = self._normalize_text(
                indexed_region.get(
                    "cleaned_text",
                    indexed_region.get(
                        "original_text",
                        ""
                    )
                )
            )

            indexed_confidence = self._bound(
                self._safe_float(
                    indexed_region.get(
                        "confidence",
                        0.0
                    )
                )
            )

            if (
                normalized_source
                and indexed_text
                == normalized_source
            ):

                return (
                    indexed_confidence,
                    indexed_region,
                    "indexed_exact_source"
                )

            if (
                normalized_value
                and indexed_text
                == normalized_value
            ):

                return (
                    indexed_confidence,
                    indexed_region,
                    "indexed_exact_value"
                )

            if (
                normalized_value
                and (
                    normalized_value
                    in indexed_text
                    or indexed_text
                    in normalized_value
                )
            ):

                return (
                    indexed_confidence,
                    indexed_region,
                    "indexed_partial_match"
                )

        # ----------------------------------------------------
        # Global exact matching.
        # ----------------------------------------------------

        candidates = []

        for region in regions:

            text = self._normalize_text(
                region.get(
                    "cleaned_text",
                    region.get(
                        "original_text",
                        ""
                    )
                )
            )

            confidence = self._bound(
                self._safe_float(
                    region.get(
                        "confidence",
                        0.0
                    )
                )
            )

            if not text:
                continue

            if (
                normalized_source
                and text
                == normalized_source
            ):

                candidates.append(
                    (
                        1.0,
                        confidence,
                        region,
                        "global_exact_source"
                    )
                )

            elif (
                normalized_value
                and text
                == normalized_value
            ):

                candidates.append(
                    (
                        0.95,
                        confidence,
                        region,
                        "global_exact_value"
                    )
                )

            elif (
                normalized_value
                and (
                    normalized_value in text
                    or text in normalized_value
                )
            ):

                candidates.append(
                    (
                        0.80,
                        confidence,
                        region,
                        "global_partial_match"
                    )
                )

        if candidates:

            candidates.sort(
                key=lambda item: (
                    item[0],
                    item[1]
                ),
                reverse=True
            )

            _, confidence, region, method = (
                candidates[0]
            )

            return (
                confidence,
                region,
                method
            )

        # ----------------------------------------------------
        # No field-specific match.
        # Use receipt mean OCR confidence as weak evidence.
        # ----------------------------------------------------

        receipt_confidences = [
            self._bound(
                self._safe_float(
                    region.get(
                        "confidence",
                        0.0
                    )
                )
            )
            for region in regions
        ]

        if receipt_confidences:

            return (
                float(
                    np.mean(
                        receipt_confidences
                    )
                ) * 0.60,
                None,
                "receipt_level_fallback"
            )

        return (
            0.0,
            None,
            "ocr_evidence_unavailable"
        )

    # ========================================================
    # EXTRACTION CONFIDENCE
    # ========================================================

    def _extraction_confidence(
        self,
        candidate: dict[str, Any] | None
    ) -> float:

        if not candidate:
            return 0.0

        return self._bound(
            self._safe_float(
                candidate.get(
                    "confidence",
                    0.0
                )
            )
        )

    # ========================================================
    # VENDOR PATTERN
    # ========================================================

    @staticmethod
    def _vendor_pattern(
        value: Any
    ) -> tuple[float, list[str]]:

        if value is None:

            return (
                0.0,
                [
                    "vendor_missing"
                ]
            )

        text = str(
            value
        ).strip()

        if not text:

            return (
                0.0,
                [
                    "vendor_empty"
                ]
            )

        alpha_count = sum(
            char.isalpha()
            for char in text
        )

        digit_count = sum(
            char.isdigit()
            for char in text
        )

        if alpha_count >= 2 and alpha_count >= digit_count:

            return (
                1.0,
                [
                    "vendor_contains_valid_text"
                ]
            )

        return (
            0.40,
            [
                "vendor_text_is_suspicious"
            ]
        )

    # ========================================================
    # DATE PATTERN
    # ========================================================

    @staticmethod
    def _date_pattern(
        value: Any
    ) -> tuple[float, list[str]]:

        if value is None:

            return (
                0.0,
                [
                    "date_missing"
                ]
            )

        text = str(
            value
        ).strip()

        if not ConfidenceEngine.DATE_PATTERN.fullmatch(
            text
        ):

            return (
                0.30,
                [
                    "invalid_date_format"
                ]
            )

        try:

            datetime.strptime(
                text,
                "%Y-%m-%d"
            )

            return (
                1.0,
                [
                    "valid_date_format",
                    "valid_calendar_date"
                ]
            )

        except ValueError:

            return (
                0.10,
                [
                    "invalid_calendar_date"
                ]
            )

    # ========================================================
    # AMOUNT PATTERN
    # ========================================================

    @staticmethod
    def _amount_pattern(
        value: Any
    ) -> tuple[float, list[str]]:

        if value is None:

            return (
                0.0,
                [
                    "amount_missing"
                ]
            )

        try:

            amount = float(
                value
            )

        except (
            TypeError,
            ValueError
        ):

            return (
                0.0,
                [
                    "amount_not_numeric"
                ]
            )

        if not np.isfinite(
            amount
        ):

            return (
                0.0,
                [
                    "amount_not_finite"
                ]
            )

        if amount < 0:

            return (
                0.10,
                [
                    "negative_amount"
                ]
            )

        if amount >= 0:

            return (
                1.0,
                [
                    "valid_numeric_amount"
                ]
            )

        return (
            0.0,
            []
        )

    # ========================================================
    # ITEM PATTERN
    # ========================================================

    @staticmethod
    def _item_pattern(
        item: dict[str, Any]
    ) -> tuple[float, list[str]]:

        evidence = []

        name = str(
            item.get(
                "name",
                ""
            )
        ).strip()

        price = item.get(
            "price"
        )

        if name:

            alpha_count = sum(
                char.isalpha()
                for char in name
            )

            if alpha_count >= 2:

                name_score = 1.0

                evidence.append(
                    "item_name_valid"
                )

            else:

                name_score = 0.30

                evidence.append(
                    "item_name_suspicious"
                )

        else:

            name_score = 0.0

            evidence.append(
                "item_name_missing"
            )

        try:

            price_value = float(
                price
            )

            if (
                np.isfinite(
                    price_value
                )
                and price_value >= 0
            ):

                price_score = 1.0

                evidence.append(
                    "item_price_valid"
                )

            else:

                price_score = 0.20

                evidence.append(
                    "item_price_invalid"
                )

        except (
            TypeError,
            ValueError
        ):

            price_score = 0.0

            evidence.append(
                "item_price_not_numeric"
            )

        return (
            0.5 * name_score
            + 0.5 * price_score,
            evidence
        )

    # ========================================================
    # CONTEXT FOR VENDOR
    # ========================================================

    @staticmethod
    def _vendor_context(
        candidate: dict[str, Any] | None
    ) -> tuple[float, list[str]]:

        if not candidate:

            return (
                0.0,
                [
                    "vendor_candidate_missing"
                ]
            )

        method = str(
            candidate.get(
                "extraction_method",
                ""
            )
        )

        evidence = list(
            candidate.get(
                "evidence",
                []
            )
            or []
        )

        if method == "vendor_label":

            return (
                1.0,
                [
                    "explicit_vendor_label"
                ]
                + evidence
            )

        if method == "top_line_heuristic":

            region_index = candidate.get(
                "region_index"
            )

            if region_index == 0:

                return (
                    0.95,
                    [
                        "first_receipt_line"
                    ]
                    + evidence
                )

            return (
                0.80,
                [
                    "upper_receipt_region"
                ]
                + evidence
            )

        return (
            0.60,
            [
                "weak_vendor_context"
            ]
            + evidence
        )

    # ========================================================
    # CONTEXT FOR DATE
    # ========================================================

    @staticmethod
    def _date_context(
        candidate: dict[str, Any] | None
    ) -> tuple[float, list[str]]:

        if not candidate:

            return (
                0.0,
                [
                    "date_candidate_missing"
                ]
            )

        method = str(
            candidate.get(
                "extraction_method",
                ""
            )
        )

        evidence = list(
            candidate.get(
                "evidence",
                []
            )
            or []
        )

        if method == "date_label_regex":

            return (
                1.0,
                [
                    "explicit_date_label"
                ]
                + evidence
            )

        if method == "date_regex":

            return (
                0.90,
                [
                    "date_pattern_detected"
                ]
                + evidence
            )

        return (
            0.60,
            [
                "weak_date_context"
            ]
            + evidence
        )

    # ========================================================
    # CONTEXT FOR TOTAL
    # ========================================================

    @staticmethod
    def _total_context(
        candidate: dict[str, Any] | None
    ) -> tuple[float, list[str]]:

        if not candidate:

            return (
                0.0,
                [
                    "total_candidate_missing"
                ]
            )

        method = str(
            candidate.get(
                "extraction_method",
                ""
            )
        )

        evidence = list(
            candidate.get(
                "evidence",
                []
            )
            or []
        )

        source_text = str(
            candidate.get(
                "source_text",
                ""
            )
        ).upper()

        keyword_detected = any(
            keyword in source_text
            for keyword
            in ConfidenceEngine.TOTAL_KEYWORDS
        )

        if (
            method == "total_keyword_amount"
            and keyword_detected
        ):

            return (
                1.0,
                [
                    "total_keyword_present",
                    "amount_context_present"
                ]
                + evidence
            )

        if method == "total_keyword_next_line":

            return (
                0.90,
                [
                    "total_keyword_adjacent_to_amount"
                ]
                + evidence
            )

        if method == "last_lines_amount_fallback":

            return (
                0.55,
                [
                    "total_fallback_context"
                ]
                + evidence
            )

        return (
            0.60,
            [
                "weak_total_context"
            ]
            + evidence
        )

    # ========================================================
    # ITEMS SOLD COUNT
    # ========================================================

    def _expected_item_count(
        self,
        extraction: dict[str, Any],
        cleaned_ocr: dict[str, Any] | None
    ) -> tuple[int | None, list[str]]:

        warnings = []

        explicit = extraction.get(
            "expected_item_count"
        )

        if explicit is not None:

            try:

                explicit = int(
                    explicit
                )

                if explicit >= 0:
                    return (
                        explicit,
                        [
                            "extraction_expected_item_count"
                        ]
                    )

            except (
                TypeError,
                ValueError
            ):
                pass

        # ----------------------------------------------------
        # Try cleaned OCR text.
        # ----------------------------------------------------

        text = ""

        if cleaned_ocr:

            text = str(
                cleaned_ocr.get(
                    "cleaned_full_text",
                    ""
                )
            )

        if not text:

            text = str(
                extraction.get(
                    "cleaned_text",
                    ""
                )
            )

        match = self.ITEM_COUNT_PATTERN.search(
            text
        )

        if match:

            try:

                count = int(
                    match.group(1)
                )

                if count >= 0:
                    return (
                        count,
                        [
                            "ocr_reported_items_sold"
                        ]
                    )

            except ValueError:
                pass

        warnings.append(
            "expected_item_count_unavailable"
        )

        return (
            None,
            warnings
        )

    # ========================================================
    # ITEM COUNT CONSISTENCY
    # ========================================================

    def _item_count_consistency(
        self,
        expected_count: int | None,
        actual_count: int
    ) -> tuple[
        float,
        bool,
        list[str]
    ]:

        if expected_count is None:

            return (
                0.50,
                False,
                [
                    "item_count_consistency_unavailable"
                ]
            )

        if expected_count == actual_count:

            return (
                1.0,
                False,
                [
                    "reported_item_count_matches_extraction"
                ]
            )

        difference = abs(
            expected_count
            - actual_count
        )

        if difference == 1:

            return (
                0.55,
                True,
                [
                    "reported_item_count_mismatch"
                ]
            )

        return (
            0.20,
            True,
            [
                "major_item_count_mismatch"
            ]
        )

    # ========================================================
    # FINANCIAL CONSISTENCY
    # ========================================================

    def _financial_consistency(
        self,
        extraction: dict[str, Any],
        expected_count: int | None
    ) -> dict[str, Any]:

        items = extraction.get(
            "items",
            []
        )

        if not isinstance(
            items,
            list
        ):
            items = []

        total = self._amount_or_none(
            extraction.get(
                "total_amount"
            )
        )

        item_prices = []

        for item in items:

            if not isinstance(
                item,
                dict
            ):
                continue

            price = self._amount_or_none(
                item.get(
                    "price"
                )
            )

            if price is not None:
                item_prices.append(
                    price
                )

        item_sum = (
            sum(
                item_prices
            )
            if item_prices
            else None
        )

        # ----------------------------------------------------
        # Item count.
        # ----------------------------------------------------

        actual_count = len(
            items
        )

        count_score, count_conflict, count_evidence = (
            self._item_count_consistency(
                expected_count,
                actual_count
            )
        )

        # ----------------------------------------------------
        # Total consistency.
        # ----------------------------------------------------

        total_score = 0.50

        total_conflict = False

        total_evidence = []

        difference = None

        relative_difference = None

        if (
            total is not None
            and item_sum is not None
        ):

            difference = (
                item_sum
                - total
            )

            absolute_difference = abs(
                difference
            )

            if total > 0:

                relative_difference = (
                    absolute_difference
                    / total
                )

            else:

                relative_difference = (
                    absolute_difference
                )

            if (
                absolute_difference
                <= self.config.absolute_total_tolerance
                or relative_difference
                <= self.config.relative_total_tolerance
            ):

                total_score = 1.0

                total_evidence.append(
                    "item_sum_matches_total"
                )

            elif (
                relative_difference
                <= 0.15
            ):

                total_score = 0.65

                total_evidence.append(
                    "item_sum_close_to_total"
                )

            else:

                total_score = 0.25

                total_conflict = True

                total_evidence.append(
                    "item_sum_differs_from_total"
                )

        else:

            total_evidence.append(
                "financial_consistency_not_fully_available"
            )

        conflicts = []

        if count_conflict:

            conflicts.append(
                "item_count_mismatch"
            )

        if total_conflict:

            conflicts.append(
                "item_sum_total_mismatch"
            )

        return {
            "item_sum": (
                round(
                    float(item_sum),
                    2
                )
                if item_sum is not None
                else None
            ),

            "total_amount": (
                round(
                    float(total),
                    2
                )
                if total is not None
                else None
            ),

            "difference": (
                round(
                    float(difference),
                    2
                )
                if difference is not None
                else None
            ),

            "relative_difference": (
                round(
                    float(relative_difference),
                    4
                )
                if relative_difference is not None
                else None
            ),

            "item_count_expected": expected_count,

            "item_count_extracted": actual_count,

            "item_count_confidence": round(
                count_score,
                self.config.confidence_digits
            ),

            "financial_confidence": round(
                total_score,
                self.config.confidence_digits
            ),

            "conflict_detected": bool(
                conflicts
            ),

            "conflicts": conflicts,

            "evidence": (
                count_evidence
                + total_evidence
            ),
        }

    # ========================================================
    # AMOUNT HELPER
    # ========================================================

    @staticmethod
    def _amount_or_none(
        value: Any
    ) -> float | None:

        try:

            if value is None:
                return None

            number = float(
                value
            )

            if not np.isfinite(
                number
            ):
                return None

            if number < 0:
                return None

            return number

        except (
            TypeError,
            ValueError
        ):

            return None

    # ========================================================
    # COMBINE CONFIDENCE
    # ========================================================

    def _combine(
        self,
        ocr_confidence: float,
        extraction_confidence: float,
        pattern_confidence: float,
        context_confidence: float,
        conflict_detected: bool = False,
        additional_penalty: float = 0.0
    ) -> float:

        # ----------------------------------------------------
        # Extraction confidence is not part of the three
        # assignment evidence weights directly. Instead,
        # combine it conservatively with pattern/context:
        #
        # pattern_component =
        #     0.70 * pattern
        #     0.30 * extraction
        #
        # This keeps the assignment's 0.5/0.3/0.2 structure
        # while incorporating the extractor's evidence quality.
        # ----------------------------------------------------

        effective_pattern = (
            0.70 * pattern_confidence
            + 0.30 * extraction_confidence
        )

        confidence = (
            self.config.ocr_weight
            * ocr_confidence
            + self.config.pattern_weight
            * effective_pattern
            + self.config.context_weight
            * context_confidence
        )

        if conflict_detected:

            confidence *= (
                1.0
                - self.config.conflict_penalty
            )

        if additional_penalty > 0:

            confidence *= (
                1.0
                - additional_penalty
            )

        return self._bound(
            confidence
        )

    # ========================================================
    # RELIABILITY
    # ========================================================

    def _reliability(
        self,
        confidence: float
    ) -> str:

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
    # REVIEW FLAG
    # ========================================================

    def _review_required(
        self,
        confidence: float,
        conflict_detected: bool
    ) -> bool:

        return (
            confidence
            < self.config.low_confidence_threshold
            or conflict_detected
        )

    # ========================================================
    # BUILD FIELD
    # ========================================================

    def _build_field(
        self,
        field_name: str,
        value: Any,
        candidate: dict[str, Any] | None,
        pattern_confidence: float,
        pattern_evidence: list[str],
        context_confidence: float,
        context_evidence: list[str],
        cleaned_regions: list[dict[str, Any]],
        conflict_detected: bool = False,
        warnings: list[str] | None = None,
        additional_penalty: float = 0.0
    ) -> FieldConfidence:

        warnings = list(
            warnings
            or []
        )

        extraction_confidence = (
            self._extraction_confidence(
                candidate
            )
        )

        source_text = (
            str(
                candidate.get(
                    "source_text"
                )
            )
            if (
                candidate
                and candidate.get(
                    "source_text"
                ) is not None
            )
            else None
        )

        region_index = (
            candidate.get(
                "region_index"
            )
            if candidate
            else None
        )

        ocr_confidence, region, ocr_match_method = (
            self._find_best_region(
                value=value,
                source_text=source_text,
                regions=cleaned_regions,
                region_index=region_index
            )
        )

        evidence = (
            list(
                pattern_evidence
            )
            + list(
                context_evidence
            )
        )

        evidence.append(
            f"ocr_match:{ocr_match_method}"
        )

        if region is not None:

            evidence.append(
                "field_specific_ocr_region_found"
            )

        else:

            warnings.append(
                "field_specific_ocr_region_not_found"
            )

        confidence = self._combine(
            ocr_confidence=ocr_confidence,
            extraction_confidence=extraction_confidence,
            pattern_confidence=pattern_confidence,
            context_confidence=context_confidence,
            conflict_detected=conflict_detected,
            additional_penalty=additional_penalty
        )

        confidence = round(
            confidence,
            self.config.confidence_digits
        )

        reliability = self._reliability(
            confidence
        )

        review_required = self._review_required(
            confidence,
            conflict_detected
        )

        if review_required:

            warnings.append(
                "review_required"
            )

        return FieldConfidence(
            field_name=field_name,
            value=value,
            confidence=confidence,
            reliability=reliability,
            review_required=review_required,
            ocr_confidence=round(
                self._bound(
                    ocr_confidence
                ),
                self.config.confidence_digits
            ),
            extraction_confidence=round(
                extraction_confidence,
                self.config.confidence_digits
            ),
            pattern_confidence=round(
                self._bound(
                    pattern_confidence
                ),
                self.config.confidence_digits
            ),
            context_confidence=round(
                self._bound(
                    context_confidence
                ),
                self.config.confidence_digits
            ),
            source_text=source_text,
            region_index=region_index,
            conflict_detected=conflict_detected,
            evidence=sorted(
                set(evidence)
            ),
            warnings=sorted(
                set(warnings)
            ),
        )

    # ========================================================
    # BUILD ITEMS
    # ========================================================

    def _build_items(
        self,
        extraction: dict[str, Any],
        cleaned_regions: list[dict[str, Any]],
        financial_consistency: dict[str, Any]
    ) -> list[ItemConfidence]:

        raw_items = extraction.get(
            "items",
            []
        )

        if not isinstance(
            raw_items,
            list
        ):
            return []

        item_count_confidence = (
            financial_consistency.get(
                "item_count_confidence",
                0.50
            )
        )

        output = []

        for item in raw_items:

            if not isinstance(
                item,
                dict
            ):
                continue

            name = str(
                item.get(
                    "name",
                    ""
                )
            ).strip()

            price = self._amount_or_none(
                item.get(
                    "price"
                )
            )

            extraction_confidence = (
                self._extraction_confidence(
                    {
                        "confidence": item.get(
                            "confidence",
                            0.0
                        )
                    }
                )
            )

            pattern_confidence, pattern_evidence = (
                self._item_pattern(
                    item
                )
            )

            source_text = (
                str(
                    item.get(
                        "source_text"
                    )
                )
                if item.get(
                    "source_text"
                ) is not None
                else None
            )

            region_index = item.get(
                "region_index"
            )

            ocr_confidence, _, ocr_method = (
                self._find_best_region(
                    value=name,
                    source_text=source_text,
                    regions=cleaned_regions,
                    region_index=region_index
                )
            )

            # ------------------------------------------------
            # Item context:
            #
            # extraction method + item-count consistency.
            # ------------------------------------------------

            method = str(
                item.get(
                    "extraction_method",
                    ""
                )
            )

            if method:

                method_context = 0.85

                context_evidence = [
                    "item_extraction_method_present"
                ]

            else:

                method_context = 0.55

                context_evidence = [
                    "item_extraction_method_missing"
                ]

            context_confidence = (
                0.70 * method_context
                + 0.30 * item_count_confidence
            )

            context_evidence.append(
                f"ocr_match:{ocr_method}"
            )

            # ------------------------------------------------
            # Item-level confidence.
            # ------------------------------------------------

            confidence = self._combine(
                ocr_confidence=ocr_confidence,
                extraction_confidence=extraction_confidence,
                pattern_confidence=pattern_confidence,
                context_confidence=context_confidence
            )

            confidence = round(
                confidence,
                self.config.confidence_digits
            )

            reliability = self._reliability(
                confidence
            )

            review_required = (
                confidence
                < self.config.low_confidence_threshold
            )

            warnings = []

            if review_required:

                warnings.append(
                    "review_required"
                )

            output.append(
                ItemConfidence(
                    name=name,
                    price=price,
                    confidence=confidence,
                    reliability=reliability,
                    review_required=review_required,
                    ocr_confidence=round(
                        self._bound(
                            ocr_confidence
                        ),
                        self.config.confidence_digits
                    ),
                    extraction_confidence=round(
                        extraction_confidence,
                        self.config.confidence_digits
                    ),
                    pattern_confidence=round(
                        self._bound(
                            pattern_confidence
                        ),
                        self.config.confidence_digits
                    ),
                    context_confidence=round(
                        self._bound(
                            context_confidence
                        ),
                        self.config.confidence_digits
                    ),
                    source_text=source_text,
                    region_index=region_index,
                    evidence=sorted(
                        set(
                            pattern_evidence
                            + context_evidence
                        )
                    ),
                    warnings=sorted(
                        set(warnings)
                    ),
                )
            )

        return output

    # ========================================================
    # PROCESS ONE RECEIPT
    # ========================================================

    def process_file(
        self,
        extraction_path: str | Path
    ) -> ReceiptConfidenceResult:

        start = time.perf_counter()

        extraction_path = Path(
            extraction_path
        )

        receipt_id = (
            extraction_path.stem
        )

        conflicts: list[str] = []

        warnings: list[str] = []

        try:

            extraction = self._load_json(
                extraction_path
            )

            receipt_id = str(
                extraction.get(
                    "receipt_id",
                    receipt_id
                )
            )

            raw_ocr, cleaned_ocr, source_files = (
                self._load_sources(
                    receipt_id
                )
            )

            cleaned_regions = self._regions(
                cleaned_ocr
            )

            # =================================================
            # EXPECTED ITEM COUNT
            # =================================================

            expected_item_count, count_evidence = (
                self._expected_item_count(
                    extraction,
                    cleaned_ocr
                )
            )

            if expected_item_count is None:

                warnings.extend(
                    count_evidence
                )

            # =================================================
            # FINANCIAL CONSISTENCY
            # =================================================

            financial_consistency = (
                self._financial_consistency(
                    extraction,
                    expected_item_count
                )
            )

            conflicts.extend(
                financial_consistency.get(
                    "conflicts",
                    []
                )
            )

            # =================================================
            # VENDOR
            # =================================================

            vendor_value = extraction.get(
                "vendor_name"
            )

            vendor_candidate = extraction.get(
                "vendor_candidate"
            )

            vendor_pattern, vendor_pattern_evidence = (
                self._vendor_pattern(
                    vendor_value
                )
            )

            vendor_context, vendor_context_evidence = (
                self._vendor_context(
                    vendor_candidate
                )
            )

            vendor_field = self._build_field(
                field_name="store_name",
                value=vendor_value,
                candidate=vendor_candidate,
                pattern_confidence=vendor_pattern,
                pattern_evidence=vendor_pattern_evidence,
                context_confidence=vendor_context,
                context_evidence=vendor_context_evidence,
                cleaned_regions=cleaned_regions,
            )

            # =================================================
            # DATE
            # =================================================

            date_value = extraction.get(
                "transaction_date"
            )

            date_candidate = extraction.get(
                "date_candidate"
            )

            date_pattern_score, date_pattern_evidence = (
                self._date_pattern(
                    date_value
                )
            )

            date_context_score, date_context_evidence = (
                self._date_context(
                    date_candidate
                )
            )

            # Detect candidate/value mismatch.
            date_conflict = False

            if (
                isinstance(
                    date_candidate,
                    dict
                )
                and date_candidate.get(
                    "value"
                ) is not None
                and date_value is not None
                and str(
                    date_candidate.get(
                        "value"
                    )
                )
                != str(
                    date_value
                )
            ):

                date_conflict = True

                conflicts.append(
                    "date_candidate_value_mismatch"
                )

            date_field = self._build_field(
                field_name="date",
                value=date_value,
                candidate=date_candidate,
                pattern_confidence=date_pattern_score,
                pattern_evidence=date_pattern_evidence,
                context_confidence=date_context_score,
                context_evidence=date_context_evidence,
                cleaned_regions=cleaned_regions,
                conflict_detected=date_conflict,
            )

            # =================================================
            # TOTAL
            # =================================================

            total_value = extraction.get(
                "total_amount"
            )

            total_candidate = extraction.get(
                "total_candidate"
            )

            total_pattern_score, total_pattern_evidence = (
                self._amount_pattern(
                    total_value
                )
            )

            total_context_score, total_context_evidence = (
                self._total_context(
                    total_candidate
                )
            )

            total_conflict = False

            if (
                isinstance(
                    total_candidate,
                    dict
                )
                and total_candidate.get(
                    "value"
                ) is not None
                and total_value is not None
            ):

                candidate_amount = (
                    self._amount_or_none(
                        total_candidate.get(
                            "value"
                        )
                    )
                )

                actual_amount = (
                    self._amount_or_none(
                        total_value
                    )
                )

                if (
                    candidate_amount is not None
                    and actual_amount is not None
                    and abs(
                        candidate_amount
                        - actual_amount
                    ) > 0.01
                ):

                    total_conflict = True

                    conflicts.append(
                        "total_candidate_value_mismatch"
                    )

            # -------------------------------------------------
            # IMPORTANT:
            # Item-count mismatch affects TOTAL reliability
            # because the total is being interpreted in a receipt
            # whose item extraction is incomplete.
            # -------------------------------------------------

            total_additional_penalty = 0.0

            if (
                expected_item_count is not None
                and expected_item_count
                != len(
                    extraction.get(
                        "items",
                        []
                    )
                    if isinstance(
                        extraction.get(
                            "items",
                            []
                        ),
                        list
                    )
                    else []
                )
            ):

                total_additional_penalty = (
                    self.config.item_count_mismatch_penalty
                    * 0.50
                )

                warnings.append(
                    "Total confidence adjusted because "
                    "reported item count does not match "
                    "extracted item count."
                )

            total_field = self._build_field(
                field_name="total_amount",
                value=total_value,
                candidate=total_candidate,
                pattern_confidence=total_pattern_score,
                pattern_evidence=total_pattern_evidence,
                context_confidence=total_context_score,
                context_evidence=total_context_evidence,
                cleaned_regions=cleaned_regions,
                conflict_detected=total_conflict,
                additional_penalty=total_additional_penalty,
            )

            # =================================================
            # ITEMS
            # =================================================

            confidence_items = (
                self._build_items(
                    extraction=extraction,
                    cleaned_regions=cleaned_regions,
                    financial_consistency=financial_consistency
                )
            )

            # =================================================
            # ITEM COUNT MISMATCH
            # =================================================

            extracted_item_count = len(
                confidence_items
            )

            item_count_conflict = (
                expected_item_count is not None
                and expected_item_count
                != extracted_item_count
            )

            if item_count_conflict:

                warnings.append(
                    "Expected item count differs from "
                    "extracted item count."
                )

                if (
                    "item_count_mismatch"
                    not in conflicts
                ):

                    conflicts.append(
                        "item_count_mismatch"
                    )

            # =================================================
            # MISSING FIELD WARNINGS
            # =================================================

            if vendor_value is None:

                warnings.append(
                    "Vendor name is missing."
                )

            if date_value is None:

                warnings.append(
                    "Transaction date is missing."
                )

            if total_value is None:

                warnings.append(
                    "Total amount is missing."
                )

            if not confidence_items:

                if (
                    expected_item_count is not None
                    and expected_item_count > 0
                ):

                    warnings.append(
                        "Receipt reports items sold, "
                        "but no valid named item was extracted."
                    )

            # =================================================
            # OVERALL CONFIDENCE
            # =================================================

            required_field_scores = [
                vendor_field.confidence,
                date_field.confidence,
                total_field.confidence,
            ]

            item_scores = [
                item.confidence
                for item
                in confidence_items
            ]

            # Required assignment fields carry the highest weight.
            required_score = float(
                np.mean(
                    required_field_scores
                )
            )

            if item_scores:

                item_score = float(
                    np.mean(
                        item_scores
                    )
                )

                overall_confidence = (
                    0.75 * required_score
                    + 0.25 * item_score
                )

            else:

                overall_confidence = (
                    required_score
                )

            # -------------------------------------------------
            # Explicit item-count mismatch penalty.
            # -------------------------------------------------

            if item_count_conflict:

                overall_confidence *= (
                    1.0
                    - self.config.item_count_mismatch_penalty
                )

            # -------------------------------------------------
            # Conflict penalty.
            # -------------------------------------------------

            if conflicts:

                overall_confidence *= (
                    1.0
                    - self.config.conflict_penalty
                )

            overall_confidence = round(
                self._bound(
                    overall_confidence
                ),
                self.config.confidence_digits
            )

            reliability = self._reliability(
                overall_confidence
            )

            field_review_required = any(
                field.review_required
                for field
                in (
                    vendor_field,
                    date_field,
                    total_field
                )
            )

            item_review_required = any(
                item.review_required
                for item
                in confidence_items
            )

            review_required = (
                overall_confidence
                < self.config.low_confidence_threshold
                or bool(conflicts)
                or field_review_required
                or item_review_required
            )

            # =================================================
            # STATUS
            # =================================================

            if (
                not vendor_value
                and not date_value
                and total_value is None
                and not confidence_items
            ):

                status = "failed"

            elif review_required:

                status = "review_required"

            else:

                status = "success"

            # =================================================
            # PROCESSING TIME
            # =================================================

            processing_time_ms = (
                time.perf_counter()
                - start
            ) * 1000.0

            return ReceiptConfidenceResult(
                receipt_id=receipt_id,

                status=status,

                overall_confidence=(
                    overall_confidence
                ),

                reliability=reliability,

                review_required=(
                    review_required
                ),

                expected_item_count=(
                    expected_item_count
                ),

                extracted_item_count=(
                    extracted_item_count
                ),

                fields={
                    "store_name": vendor_field,
                    "date": date_field,
                    "total_amount": total_field,
                },

                items=confidence_items,

                conflicts=sorted(
                    set(
                        conflicts
                    )
                ),

                warnings=sorted(
                    set(
                        warnings
                        + extraction.get(
                            "warnings",
                            []
                        )
                        if isinstance(
                            extraction.get(
                                "warnings",
                                []
                            ),
                            list
                        )
                        else warnings
                    )
                ),

                financial_consistency=(
                    financial_consistency
                ),

                source_files=source_files,

                processing_time_ms=round(
                    processing_time_ms,
                    3
                ),

                timestamp_utc=datetime.now(
                    timezone.utc
                ).isoformat(),
            )

        except Exception as exc:

            logger.exception(
                "Confidence processing failed for %s",
                extraction_path
            )

            processing_time_ms = (
                time.perf_counter()
                - start
            ) * 1000.0

            return ReceiptConfidenceResult(
                receipt_id=receipt_id,

                status="failed",

                overall_confidence=0.0,

                reliability="low",

                review_required=True,

                expected_item_count=None,

                extracted_item_count=0,

                fields={},

                items=[],

                conflicts=[
                    "confidence_engine_exception"
                ],

                warnings=[
                    str(exc)
                ],

                financial_consistency={},

                source_files={
                    "raw_ocr": None,
                    "cleaned_ocr": None,
                },

                processing_time_ms=round(
                    processing_time_ms,
                    3
                ),

                timestamp_utc=datetime.now(
                    timezone.utc
                ).isoformat(),
            )

    # ========================================================
    # SAVE RESULT
    # ========================================================

    def _save_result(
        self,
        result: ReceiptConfidenceResult
    ) -> Path:

        path = (
            self.output_dir
            / f"{result.receipt_id}.json"
        )

        with path.open(
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                asdict(
                    result
                ),
                file,
                indent=2,
                ensure_ascii=False,
                allow_nan=False
            )

        return path

    # ========================================================
    # BATCH
    # ========================================================

    def process_batch(
        self
    ) -> dict[str, Any]:

        start = time.perf_counter()

        logger.info(
            "Starting confidence engine batch."
        )

        files = (
            self._discover_extraction_files()
        )

        results = []

        for index, file in enumerate(
            files,
            start=1
        ):

            result = self.process_file(
                file
            )

            results.append(
                result
            )

            try:

                output_path = self._save_result(
                    result
                )

                logger.info(
                    "Saved confidence result: %s",
                    output_path
                )

            except Exception as exc:

                logger.exception(
                    "Could not save confidence result."
                )

                result.status = "failed"

                result.review_required = True

                result.warnings.append(
                    f"Artifact save failed: {exc}"
                )

            if index % 100 == 0:

                logger.info(
                    "Confidence processed %d/%d receipts.",
                    index,
                    len(files)
                )

        total = len(
            results
        )

        successful = sum(
            result.status == "success"
            for result in results
        )

        review_required = sum(
            result.review_required
            for result in results
        )

        failed = sum(
            result.status == "failed"
            for result in results
        )

        high = sum(
            result.reliability == "high"
            for result in results
        )

        medium = sum(
            result.reliability == "medium"
            for result in results
        )

        low = sum(
            result.reliability == "low"
            for result in results
        )

        conflicts = sum(
            bool(
                result.conflicts
            )
            for result in results
        )

        confidence_values = [
            result.overall_confidence
            for result in results
        ]

        summary = {
            "status": "success",

            "timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat(),

            "total_receipts": total,

            "successful": successful,

            "failed": failed,

            "review_required": review_required,

            "review_rate": (
                review_required / total
                if total
                else 0.0
            ),

            "high_reliability": high,

            "medium_reliability": medium,

            "low_reliability": low,

            "conflict_receipts": conflicts,

            "mean_overall_confidence": (
                round(
                    float(
                        np.mean(
                            confidence_values
                        )
                    ),
                    self.config.confidence_digits
                )
                if confidence_values
                else 0.0
            ),

            "minimum_overall_confidence": (
                round(
                    float(
                        np.min(
                            confidence_values
                        )
                    ),
                    self.config.confidence_digits
                )
                if confidence_values
                else 0.0
            ),

            "maximum_overall_confidence": (
                round(
                    float(
                        np.max(
                            confidence_values
                        )
                    ),
                    self.config.confidence_digits
                )
                if confidence_values
                else 0.0
            ),

            "processing_time_ms": round(
                (
                    time.perf_counter()
                    - start
                ) * 1000.0,
                3
            ),
        }

        if self.config.save_manifest:

            manifest_path = (
                self.output_dir
                / "confidence_manifest.json"
            )

            with manifest_path.open(
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    summary,
                    file,
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=False
                )

            summary[
                "manifest_path"
            ] = str(
                manifest_path
            )

        logger.info(
            "Confidence engine completed."
        )

        return summary

    # ========================================================
    # PUBLIC ENTRY POINT
    # ========================================================

    def run(
        self,
        json_path: str | Path | None = None
    ) -> (
        ReceiptConfidenceResult
        | dict[str, Any]
    ):

        if json_path is not None:

            result = self.process_file(
                json_path
            )

            self._save_result(
                result
            )

            return result

        return self.process_batch()