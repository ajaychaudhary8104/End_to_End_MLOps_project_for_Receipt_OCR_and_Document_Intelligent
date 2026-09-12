from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any
from src.receipt_intelligence.entity.config_entity import FinancialSummaryConfig
from src.receipt_intelligence import logger
import pandas as pd

# ============================================================
# RESULT
# ============================================================

@dataclass
class FinancialSummaryResult:
    """
    Complete financial aggregation result.
    """

    status: str

    generated_at_utc: str

    # --------------------------------------------------------
    # Required assignment outputs
    # --------------------------------------------------------

    total_spend: float

    transaction_count: int

    spend_per_store: dict[str, float]

    # --------------------------------------------------------
    # Trusted dataset metrics
    # --------------------------------------------------------

    receipts_processed: int

    receipts_included: int

    receipts_excluded: int

    # --------------------------------------------------------
    # Operational metrics
    # --------------------------------------------------------

    files_discovered: int

    receipts_loaded: int

    load_failures: int

    # --------------------------------------------------------
    # Data-quality metrics
    # --------------------------------------------------------

    low_confidence_receipts: int

    review_required_receipts: int

    missing_total_receipts: int

    missing_store_receipts: int

    invalid_receipts: int

    duplicate_receipts: int

    # --------------------------------------------------------
    # Excluded financial data
    # --------------------------------------------------------

    excluded_spend: float

    excluded_transactions: int

    # --------------------------------------------------------
    # Descriptive statistics
    # --------------------------------------------------------

    average_transaction_value: float

    median_transaction_value: float

    minimum_transaction_value: float | None

    maximum_transaction_value: float | None

    # --------------------------------------------------------
    # Store-level transaction counts
    # --------------------------------------------------------

    transaction_count_per_store: dict[str, int]

    # --------------------------------------------------------
    # Unknown-store metrics
    # --------------------------------------------------------

    unassigned_store_spend: float

    unassigned_store_transactions: int

    # --------------------------------------------------------
    # Exclusion audit
    # --------------------------------------------------------

    exclusion_reason_counts: dict[str, int]

    exclusion_spend_by_primary_reason: dict[str, float]

    # --------------------------------------------------------
    # Financial reconciliation
    # --------------------------------------------------------

    reconciliation_status: str

    reconciliation_difference: float

    # --------------------------------------------------------
    # Artifact references
    # --------------------------------------------------------

    report_path: str | None

    receipt_table_path: str | None

    audit_table_path: str | None

    # --------------------------------------------------------
    # Operational artifact errors
    # --------------------------------------------------------

    artifact_errors: list[str]

    # --------------------------------------------------------
    # Processing
    # --------------------------------------------------------

    processing_time_ms: float


# ============================================================
# ENGINE
# ============================================================

class FinancialSummaryEngine:
    """
    Production-grade financial aggregation engine.

    PIPELINE
    --------

        Final JSON
             ↓
        Defensive Validation
             ↓
        Receipt ID Resolution
             ↓
        Duplicate Detection
             ↓
        Decimal Money Normalization
             ↓
        Confidence / Review Filtering
             ↓
        Trusted Dataset
             ↓
        ┌─────────────────────────────┐
        │ Total Spend                │
        │ Transaction Count          │
        │ Spend Per Store            │
        │ Transaction Count Per Store│
        └─────────────────────────────┘
             ↓
        Audit Statistics
             ↓
        Reconciliation
             ↓
        JSON + CSV Artifacts

    IMPORTANT
    ---------
    The trusted financial summary is intentionally different from
    the raw extracted financial universe.

    Example:

        Extracted valid totals = 100,000
        Trusted totals         = 82,000
        Excluded spend         = 18,000

    Therefore the report does NOT misleadingly imply that only
    82,000 existed in the extracted dataset.
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

    PRIMARY_EXCLUSION_PRIORITY = (
        "duplicate_receipt_id",
        "validation_failed",
        "validation_status_missing",
        "schema_invalid",
        "missing_total",
        "invalid_total",
        "review_required",
        "invalid_confidence",
        "low_confidence",
    )

    # ========================================================
    # INIT
    # ========================================================

    def __init__(
        self,
        config: FinancialSummaryConfig,
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
        # Configuration validation
        # ----------------------------------------------------

        if not (
            0.0
            <= config.minimum_confidence
            <= 1.0
        ):
            raise ValueError(
                "minimum_confidence must be between 0 and 1."
            )

        if config.currency_decimals < 0:
            raise ValueError(
                "currency_decimals cannot be negative."
            )

        if not config.schema_version.strip():
            raise ValueError(
                "schema_version cannot be empty."
            )

        # ----------------------------------------------------
        # Decimal configuration
        # ----------------------------------------------------

        self._minimum_confidence = Decimal(
            str(
                config.minimum_confidence
            )
        )

        self._currency_quantum = (
            Decimal("1").scaleb(
                -config.currency_decimals
            )
        )

        # ----------------------------------------------------
        # Runtime state
        # ----------------------------------------------------

        self._records: list[
            dict[str, Any]
        ] = []

        self._load_failures: list[
            dict[str, Any]
        ] = []

        self._artifact_errors: list[
            str
        ] = []

    # ========================================================
    # DECIMAL HELPERS
    # ========================================================

    def _quantize_money(
        self,
        value: Decimal,
    ) -> Decimal:
        """
        Financial rounding using ROUND_HALF_UP.
        """

        return value.quantize(
            self._currency_quantum,
            rounding=ROUND_HALF_UP,
        )

    @staticmethod
    def _decimal_from_any(
        value: Any,
    ) -> Decimal | None:
        """
        Convert a value to finite Decimal.

        Returns None for:
            - None
            - NaN
            - Infinity
            - invalid strings
            - bool
        """

        if value is None:
            return None

        if isinstance(
            value,
            bool,
        ):
            return None

        try:

            number = (
                value
                if isinstance(
                    value,
                    Decimal,
                )
                else Decimal(
                    str(value).strip()
                )
            )

            if not number.is_finite():
                return None

            return number

        except (
            InvalidOperation,
            TypeError,
            ValueError,
        ):

            return None

    def _decimal_to_float(
        self,
        value: Decimal | None,
    ) -> float | None:
        """
        Convert Decimal to finite JSON-safe float.
        """

        if value is None:
            return None

        result = float(
            value
        )

        if (
            result != result
            or result in (
                float("inf"),
                float("-inf"),
            )
        ):
            return None

        return round(
            result,
            self.config.currency_decimals,
        )

    # ========================================================
    # FILE DISCOVERY
    # ========================================================

    def _discover_files(
        self,
    ) -> list[Path]:
        """
        Discover final receipt JSON artifacts.

        Batch/report manifests are excluded.
        """

        if not self.input_dir.exists():

            raise FileNotFoundError(
                f"JSON input directory does not exist: "
                f"{self.input_dir}"
            )

        if not self.input_dir.is_dir():

            raise NotADirectoryError(
                f"Expected directory: "
                f"{self.input_dir}"
            )

        excluded_names = {
            "json_manifest.json",
            "financial_summary.json",
        }

        files = sorted(
            path
            for path
            in self.input_dir.rglob(
                "*.json"
            )
            if path.name
            not in excluded_names
        )

        logger.info(
            "Discovered %d receipt JSON files.",
            len(files),
        )

        return files

    # ========================================================
    # JSON LOADING
    # ========================================================

    @staticmethod
    def _load_json(
        path: Path,
    ) -> dict[str, Any]:
        """
        Load JSON and require an object at the root.
        """

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(
                file
            )

        if not isinstance(
            data,
            dict,
        ):

            raise ValueError(
                f"Expected JSON object in {path}"
            )

        return data

    # ========================================================
    # STRING NORMALIZATION
    # ========================================================

    @staticmethod
    def _normalize_store(
        value: Any,
    ) -> str | None:
        """
        Normalize a store display name.

        This does NOT aggressively merge business entities.
        """

        if value is None:
            return None

        text = str(
            value
        ).strip()

        if not text:
            return None

        return " ".join(
            text.split()
        )

    @staticmethod
    def _canonical_store(
        store_name: str | None,
    ) -> str:
        """
        Conservative grouping key.

        Examples:

            "DMART"     -> "DMART"
            " Dmart "   -> "DMART"
            "D-MART"    -> "D MART"

        The display value is preserved separately.
        """

        if store_name is None:
            return "__UNKNOWN_STORE__"

        text = str(
            store_name
        ).upper()

        text = re.sub(
            r"[^A-Z0-9]+",
            " ",
            text,
        )

        text = " ".join(
            text.split()
        )

        return (
            text
            if text
            else "__UNKNOWN_STORE__"
        )

    # ========================================================
    # BOOLEAN NORMALIZATION
    # ========================================================

    @staticmethod
    def _normalize_bool(
        value: Any,
        default: bool = False,
    ) -> bool:
        """
        Avoid Python's dangerous:

            bool("false") == True
        """

        if isinstance(
            value,
            bool,
        ):
            return value

        if (
            isinstance(
                value,
                (int, float),
            )
            and not isinstance(
                value,
                bool,
            )
        ):

            if value == 1:
                return True

            if value == 0:
                return False

        if isinstance(
            value,
            str,
        ):

            normalized = (
                value
                .strip()
                .casefold()
            )

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
    # MONEY NORMALIZATION
    # ========================================================

    def _normalize_amount(
        self,
        value: Any,
    ) -> Decimal | None:
        """
        Normalize currency into Decimal.

        Examples:

            40
            "40.00"
            "₹40.00"
            "Rs. 40.00"
            "Rs 40.00"
            "INR 40.00"
            "1,250.50"

        Negative values are rejected because this engine represents
        purchase spend, not refund/net-ledger accounting.
        """

        if value is None:
            return None

        if isinstance(
            value,
            str,
        ):

            text = value.strip()

            text = re.sub(
                r"(?i)(₹|rs\.?|inr)",
                "",
                text,
            )

            text = text.replace(
                ",",
                "",
            )

            text = text.strip()

            # Accounting-style negative value.
            if (
                text.startswith("(")
                and text.endswith(")")
            ):
                return None

            value = text

        number = self._decimal_from_any(
            value
        )

        if number is None:
            return None

        if number < 0:
            return None

        try:

            return self._quantize_money(
                number
            )

        except (
            InvalidOperation,
            ValueError,
        ):

            return None

    # ========================================================
    # CONFIDENCE
    # ========================================================

    def _get_confidence(
        self,
        data: dict[str, Any],
    ) -> tuple[
        Decimal,
        bool,
        bool,
    ]:
        """
        Return:

            confidence,
            valid,
            missing

        Missing confidence becomes 0.0 and therefore cannot enter
        the trusted financial summary.
        """

        if (
            "overall_confidence"
            not in data
        ):

            return (
                Decimal("0"),
                True,
                True,
            )

        raw_value = data.get(
            "overall_confidence"
        )

        if raw_value is None:

            return (
                Decimal("0"),
                True,
                True,
            )

        value = self._decimal_from_any(
            raw_value
        )

        if value is None:

            return (
                Decimal("0"),
                False,
                False,
            )

        if not (
            Decimal("0")
            <= value
            <= Decimal("1")
        ):

            return (
                Decimal("0"),
                False,
                False,
            )

        return (
            value,
            True,
            False,
        )

    # ========================================================
    # DATE VALIDATION
    # ========================================================

    @staticmethod
    def _validate_date(
        value: Any,
    ) -> bool:
        """
        Require YYYY-MM-DD and a real calendar date.
        """

        if value is None:
            return True

        text = str(
            value
        ).strip()

        if not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}",
            text,
        ):

            return False

        try:

            datetime.strptime(
                text,
                "%Y-%m-%d",
            )

            return True

        except ValueError:

            return False

    # ========================================================
    # RECEIPT ID
    # ========================================================

    @staticmethod
    def _resolve_receipt_id(
        data: dict[str, Any],
        source_path: Path,
    ) -> str:
        """
        Resolve ID from metadata first, then top-level data,
        then filename.
        """

        metadata = data.get(
            "metadata"
        )

        if isinstance(
            metadata,
            dict,
        ):

            receipt_id = metadata.get(
                "receipt_id"
            )

            if receipt_id is not None:

                text = str(
                    receipt_id
                ).strip()

                if text:
                    return text

        receipt_id = data.get(
            "receipt_id"
        )

        if receipt_id is not None:

            text = str(
                receipt_id
            ).strip()

            if text:
                return text

        return source_path.stem

    # ========================================================
    # FINAL JSON VALIDATION
    # ========================================================

    def _validate_receipt_input(
        self,
        data: dict[str, Any],
    ) -> tuple[
        bool,
        list[str],
        str,
    ]:
        """
        Defense-in-depth validation.

        The JSON Generator already validates output, but financial
        calculations must not blindly trust upstream artifacts.
        """

        errors: list[str] = []

        # ----------------------------------------------------
        # Final JSON generator validation status
        # ----------------------------------------------------

        validation = data.get(
            "_validation"
        )

        validation_status = None

        if isinstance(
            validation,
            dict,
        ):

            validation_status = validation.get(
                "status"
            )

        if self.config.require_validation_passed:

            if validation_status is None:

                errors.append(
                    "validation_status_missing"
                )

            elif validation_status != "passed":

                errors.append(
                    "validation_failed"
                )

        # ----------------------------------------------------
        # Required fields
        # ----------------------------------------------------

        for field_name in (
            self.REQUIRED_TOP_LEVEL_FIELDS
        ):

            if field_name not in data:

                errors.append(
                    f"missing_required_field:{field_name}"
                )

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        if (
            "store_name" in data
            and data["store_name"] is not None
            and not isinstance(
                data["store_name"],
                str,
            )
        ):

            errors.append(
                "schema_invalid"
            )

        # ----------------------------------------------------
        # Date
        # ----------------------------------------------------

        date_value = data.get(
            "date"
        )

        if date_value is not None:

            if not isinstance(
                date_value,
                str,
            ):

                errors.append(
                    "schema_invalid"
                )

            elif not self._validate_date(
                date_value
            ):

                errors.append(
                    "invalid_date"
                )

        # ----------------------------------------------------
        # Items
        # ----------------------------------------------------

        items = data.get(
            "items"
        )

        if not isinstance(
            items,
            list,
        ):

            errors.append(
                "schema_invalid"
            )

        else:

            for item in items:

                if not isinstance(
                    item,
                    dict,
                ):

                    errors.append(
                        "schema_invalid"
                    )

                    continue

                name = item.get(
                    "name"
                )

                if (
                    not isinstance(
                        name,
                        str,
                    )
                    or not name.strip()
                ):

                    errors.append(
                        "schema_invalid"
                    )

                if "price" in item:

                    price = item.get(
                        "price"
                    )

                    if price is not None:

                        normalized_price = (
                            self._normalize_amount(
                                price
                            )
                        )

                        if normalized_price is None:

                            errors.append(
                                "schema_invalid"
                            )

        # ----------------------------------------------------
        # Total
        # ----------------------------------------------------

        total_raw = data.get(
            "total_amount"
        )

        if total_raw is not None:

            if (
                isinstance(
                    total_raw,
                    bool,
                )
                or not isinstance(
                    total_raw,
                    (
                        int,
                        float,
                        str,
                        Decimal,
                    ),
                )
            ):

                errors.append(
                    "schema_invalid"
                )

            elif (
                self._normalize_amount(
                    total_raw
                )
                is None
            ):

                errors.append(
                    "invalid_total"
                )

        # ----------------------------------------------------
        # Overall confidence
        # ----------------------------------------------------

        confidence_raw = data.get(
            "overall_confidence"
        )

        if confidence_raw is not None:

            confidence = self._decimal_from_any(
                confidence_raw
            )

            if (
                confidence is None
                or not (
                    Decimal("0")
                    <= confidence
                    <= Decimal("1")
                )
            ):

                errors.append(
                    "invalid_confidence"
                )

        # ----------------------------------------------------
        # Review flag
        # ----------------------------------------------------

        review_value = data.get(
            "review_required"
        )

        if (
            review_value is not None
            and not isinstance(
                review_value,
                (
                    bool,
                    str,
                    int,
                    float,
                ),
            )
        ):

            errors.append(
                "schema_invalid"
            )

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        unique_errors = sorted(
            set(
                errors
            )
        )

        if (
            "validation_failed"
            in unique_errors
        ):

            classification = (
                "validation_failed"
            )

        elif (
            "validation_status_missing"
            in unique_errors
        ):

            classification = (
                "validation_status_missing"
            )

        elif (
            "schema_invalid"
            in unique_errors
        ):

            classification = (
                "schema_invalid"
            )

        else:

            classification = "valid"

        return (
            not unique_errors,
            unique_errors,
            classification,
        )

    # ========================================================
    # NORMALIZE RECEIPT
    # ========================================================

    def _normalize_receipt(
        self,
        data: dict[str, Any],
        source_path: Path,
    ) -> dict[str, Any]:
        """
        Convert one final JSON artifact into an internal record.
        """

        receipt_id = (
            self._resolve_receipt_id(
                data,
                source_path,
            )
        )

        store_name = (
            self._normalize_store(
                data.get(
                    "store_name"
                )
            )
        )

        date_value = data.get(
            "date"
        )

        if date_value is not None:

            date_value = str(
                date_value
            ).strip()

            if not date_value:

                date_value = None

        raw_total = data.get(
            "total_amount"
        )

        total_amount = (
            self._normalize_amount(
                raw_total
            )
        )

        # ----------------------------------------------------
        # Distinguish missing from invalid totals.
        # ----------------------------------------------------

        if raw_total is None:

            missing_total = True

            invalid_total = False

        elif (
            isinstance(
                raw_total,
                str,
            )
            and not raw_total.strip()
        ):

            missing_total = True

            invalid_total = False

        else:

            missing_total = False

            invalid_total = (
                total_amount is None
            )

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        (
            confidence,
            confidence_valid,
            confidence_missing,
        ) = self._get_confidence(
            data
        )

        # ----------------------------------------------------
        # Review
        # ----------------------------------------------------

        review_required = (
            self._normalize_bool(
                data.get(
                    "review_required"
                ),
                default=False,
            )
        )

        # ----------------------------------------------------
        # Reliability
        # ----------------------------------------------------

        reliability_raw = data.get(
            "reliability"
        )

        reliability = (
            str(
                reliability_raw
            ).strip().casefold()
            if reliability_raw is not None
            else "low"
        )

        reliability_valid = (
            reliability
            in self.ALLOWED_RELIABILITY
        )

        if not reliability_valid:

            reliability = "low"

        # ----------------------------------------------------
        # Date quality
        # ----------------------------------------------------

        date_valid = (
            self._validate_date(
                date_value
            )
        )

        # ----------------------------------------------------
        # Schema validation
        # ----------------------------------------------------

        (
            schema_valid,
            validation_errors,
            validation_classification,
        ) = self._validate_receipt_input(
            data
        )

        return {
            "receipt_id": receipt_id,

            "store_name": store_name,

            "store_key": self._canonical_store(
                store_name
            ),

            "transaction_date": date_value,

            "total_amount": total_amount,

            "raw_total_present": (
                raw_total is not None
            ),

            "missing_total": missing_total,

            "invalid_total": invalid_total,

            "overall_confidence": confidence,

            "confidence_valid": confidence_valid,

            "confidence_missing": confidence_missing,

            "reliability": reliability,

            "reliability_valid": reliability_valid,

            "review_required": review_required,

            "date_valid": date_valid,

            "schema_valid": schema_valid,

            "validation_classification": (
                validation_classification
            ),

            "validation_errors": validation_errors,

            "missing_store": (
                store_name is None
            ),

            "source_file": str(
                source_path
            ),

            "included": False,

            "exclusion_reasons": [],

            "primary_exclusion_reason": None,
        }

    # ========================================================
    # EXCLUSION REASONS
    # ========================================================

    def _determine_exclusion_reasons(
        self,
        record: dict[str, Any],
    ) -> list[str]:
        """
        Produce ALL applicable exclusion reasons.

        Multiple reasons are preserved.

        Example:

            [
                "review_required",
                "low_confidence"
            ]
        """

        reasons: list[str] = []

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        classification = (
            record[
                "validation_classification"
            ]
        )

        if (
            classification
            == "validation_failed"
        ):

            reasons.append(
                "validation_failed"
            )

        elif (
            classification
            == "validation_status_missing"
        ):

            reasons.append(
                "validation_status_missing"
            )

        if (
            not record[
                "schema_valid"
            ]
            and classification
            not in {
                "validation_failed",
                "validation_status_missing",
            }
        ):

            reasons.append(
                "schema_invalid"
            )

        # ----------------------------------------------------
        # Total
        # ----------------------------------------------------

        if record[
            "missing_total"
        ]:

            reasons.append(
                "missing_total"
            )

        elif record[
            "invalid_total"
        ]:

            reasons.append(
                "invalid_total"
            )

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        if not record[
            "confidence_valid"
        ]:

            reasons.append(
                "invalid_confidence"
            )

        elif (
            record[
                "overall_confidence"
            ]
            < self._minimum_confidence
        ):

            reasons.append(
                "low_confidence"
            )

        # ----------------------------------------------------
        # Review
        # ----------------------------------------------------

        if (
            self.config.exclude_review_required
            and record[
                "review_required"
            ]
        ):

            reasons.append(
                "review_required"
            )

        return sorted(
            set(reasons),
                key=lambda reason: (
                    self.PRIMARY_EXCLUSION_PRIORITY.index(
                        reason
                    )
                    if reason
                    in self.PRIMARY_EXCLUSION_PRIORITY
                    else 999
                ),
            )

    # ========================================================
    # PRIMARY EXCLUSION REASON
    # ========================================================

    def _primary_exclusion_reason(
        self,
        reasons: list[str],
    ) -> str | None:
        """
        Select ONE primary reason for mutually-exclusive financial
        exclusion-spend reporting.

        All reasons remain preserved separately.
        """

        if not reasons:
            return None

        for reason in (
            self.PRIMARY_EXCLUSION_PRIORITY
        ):

            if reason in reasons:
                return reason

        return reasons[0]

    # ========================================================
    # DUPLICATE DETECTION
    # ========================================================

    def _apply_duplicate_policy(
        self,
        records: list[dict[str, Any]],
    ) -> int:
        """
        Exclude ALL records belonging to duplicate receipt-ID groups.

        Why?

        A "first wins" strategy can silently make results dependent
        on filename order.

        Excluding the entire duplicate group is safer for financial
        reporting than arbitrarily selecting one record.
        """

        groups: dict[
            str,
            list[dict[str, Any]]
        ] = defaultdict(list)

        for record in records:

            groups[
                record[
                    "receipt_id"
                ]
            ].append(
                record
            )

        duplicate_records = 0

        for receipt_id, group in groups.items():

            if len(group) <= 1:
                continue

            duplicate_records += len(
                group
            )

            logger.warning(
                "Duplicate receipt_id detected: %s (%d records)",
                receipt_id,
                len(group),
            )

            for record in group:

                if (
                    "duplicate_receipt_id"
                    not in record[
                        "exclusion_reasons"
                    ]
                ):

                    record[
                        "exclusion_reasons"
                    ].append(
                        "duplicate_receipt_id"
                    )

        return duplicate_records

    # ========================================================
    # FINAL INCLUSION DECISION
    # ========================================================

    def _finalize_record_decision(
        self,
        record: dict[str, Any],
    ) -> None:
        """
        Finalize trusted/excluded state.
        """

        reasons = (
            self._determine_exclusion_reasons(
                record
            )
        )

        # Duplicate policy is applied before this method.
        if (
            "duplicate_receipt_id"
            in record[
                "exclusion_reasons"
            ]
        ):

            reasons.append(
                "duplicate_receipt_id"
            )

        reasons = sorted(
            set(reasons),
                key=lambda reason: (
                    self.PRIMARY_EXCLUSION_PRIORITY.index(
                        reason
                    )
                    if reason
                    in self.PRIMARY_EXCLUSION_PRIORITY
                    else 999
                ),
            )

        record[
            "exclusion_reasons"
        ] = reasons

        if reasons:

            record[
                "included"
            ] = False

            record[
                "primary_exclusion_reason"
            ] = (
                self._primary_exclusion_reason(
                    reasons
                )
            )

        else:

            record[
                "included"
            ] = True

            record[
                "primary_exclusion_reason"
            ] = None

    # ========================================================
    # DECIMAL AGGREGATION
    # ========================================================

    @staticmethod
    def _sum_decimal(
        values: list[Decimal],
    ) -> Decimal:

        return sum(
            values,
            Decimal("0"),
        )

    @staticmethod
    def _median_decimal(
        values: list[Decimal],
    ) -> Decimal:
        """
        Exact Decimal median.
        """

        if not values:
            return Decimal("0")

        ordered = sorted(
            values
        )

        count = len(
            ordered
        )

        middle = count // 2

        if count % 2 == 1:

            return ordered[
                middle
            ]

        return (
            ordered[
                middle - 1
            ]
            + ordered[
                middle
            ]
        ) / Decimal("2")

    # ========================================================
    # TRUSTED RECORDS
    # ========================================================

    def _trusted_records(
        self,
    ) -> list[dict[str, Any]]:

        return [
            record
            for record
            in self._records
            if record[
                "included"
            ]
        ]

    # ========================================================
    # SPEND PER STORE
    # ========================================================

    def _calculate_spend_per_store(
        self,
        records: list[dict[str, Any]],
    ) -> dict[str, float]:
        """
        Aggregate trusted spending by store.
        """

        totals: dict[
            str,
            Decimal
        ] = defaultdict(
            lambda: Decimal("0")
        )

        display_names: dict[
            str,
            str
        ] = {}

        for record in records:

            total = record[
                "total_amount"
            ]

            if total is None:
                continue

            store_key = record[
                "store_key"
            ]

            totals[
                store_key
            ] += total

            if (
                store_key
                not in display_names
            ):

                display_names[
                    store_key
                ] = (
                    record[
                        "store_name"
                    ]
                    or "Unknown / Unassigned"
                )

        ordered_keys = sorted(
            totals.keys(),
            key=lambda key: (
                -totals[key],
                display_names.get(
                    key,
                    key,
                ).casefold(),
            ),
        )

        result: dict[
            str,
            float
        ] = {}

        for store_key in ordered_keys:

            display_name = (
                display_names.get(
                    store_key,
                    "Unknown / Unassigned",
                )
            )

            amount = self._quantize_money(
                totals[
                    store_key
                ]
            )

            result[
                display_name
            ] = (
                self._decimal_to_float(
                    amount
                )
                or 0.0
            )

        return result

    # ========================================================
    # TRANSACTION COUNT PER STORE
    # ========================================================

    def _calculate_transaction_count_per_store(
        self,
        records: list[dict[str, Any]],
    ) -> dict[str, int]:

        counts: Counter[str] = Counter()

        display_names: dict[
            str,
            str
        ] = {}

        for record in records:

            store_key = record[
                "store_key"
            ]

            counts[
                store_key
            ] += 1

            if (
                store_key
                not in display_names
            ):

                display_names[
                    store_key
                ] = (
                    record[
                        "store_name"
                    ]
                    or "Unknown / Unassigned"
                )

        ordered_keys = sorted(
            counts.keys(),
            key=lambda key: (
                -counts[key],
                display_names.get(
                    key,
                    key,
                ).casefold(),
            ),
        )

        return {
            display_names.get(
                key,
                "Unknown / Unassigned",
            ): int(
                counts[key]
            )
            for key in ordered_keys
        }

    # ========================================================
    # RECONCILIATION
    # ========================================================

    def _reconcile(
        self,
        total_spend: Decimal,
        spend_per_store: dict[str, float],
    ) -> tuple[
        str,
        Decimal,
    ]:
        """
        Verify:

            total_spend
            ==
            sum(spend_per_store)

        after currency normalization.
        """

        store_total = self._sum_decimal(
            [
                Decimal(
                    str(
                        amount
                    )
                )
                for amount
                in spend_per_store.values()
            ]
        )

        store_total = self._quantize_money(
            store_total
        )

        difference = self._quantize_money(
            store_total
            - total_spend
        )

        if (
            difference
            == self._quantize_money(
                Decimal("0")
            )
        ):

            return (
                "passed",
                difference,
            )

        return (
            "failed",
            difference,
        )

    # ========================================================
    # RECEIPT TABLE
    # ========================================================

    def _build_receipt_dataframe(
        self,
        records: list[dict[str, Any]],
    ) -> pd.DataFrame:

        rows: list[
            dict[str, Any]
        ] = []

        for record in records:

            rows.append(
                {
                    "receipt_id": (
                        record[
                            "receipt_id"
                        ]
                    ),

                    "store_name": (
                        record[
                            "store_name"
                        ]
                    ),

                    "transaction_date": (
                        record[
                            "transaction_date"
                        ]
                    ),

                    "total_amount": (
                        self._decimal_to_float(
                            record[
                                "total_amount"
                            ]
                        )
                    ),

                    "overall_confidence": (
                        float(
                            record[
                                "overall_confidence"
                            ]
                        )
                    ),

                    "confidence_valid": (
                        bool(
                            record[
                                "confidence_valid"
                            ]
                        )
                    ),

                    "confidence_missing": (
                        bool(
                            record[
                                "confidence_missing"
                            ]
                        )
                    ),

                    "reliability": (
                        record[
                            "reliability"
                        ]
                    ),

                    "reliability_valid": (
                        bool(
                            record[
                                "reliability_valid"
                            ]
                        )
                    ),

                    "review_required": (
                        bool(
                            record[
                                "review_required"
                            ]
                        )
                    ),

                    "date_valid": (
                        bool(
                            record[
                                "date_valid"
                            ]
                        )
                    ),

                    "schema_valid": (
                        bool(
                            record[
                                "schema_valid"
                            ]
                        )
                    ),

                    "validation_classification": (
                        record[
                            "validation_classification"
                        ]
                    ),

                    "included": (
                        bool(
                            record[
                                "included"
                            ]
                        )
                    ),

                    "primary_exclusion_reason": (
                        record[
                            "primary_exclusion_reason"
                        ]
                    ),

                    "exclusion_reasons": ";".join(
                        record[
                            "exclusion_reasons"
                        ]
                    ),

                    "missing_total": (
                        bool(
                            record[
                                "missing_total"
                            ]
                        )
                    ),

                    "invalid_total": (
                        bool(
                            record[
                                "invalid_total"
                            ]
                        )
                    ),

                    "missing_store": (
                        bool(
                            record[
                                "missing_store"
                            ]
                        )
                    ),

                    "source_file": (
                        record[
                            "source_file"
                        ]
                    ),
                }
            )

        columns = [
            "receipt_id",
            "store_name",
            "transaction_date",
            "total_amount",
            "overall_confidence",
            "confidence_valid",
            "confidence_missing",
            "reliability",
            "reliability_valid",
            "review_required",
            "date_valid",
            "schema_valid",
            "validation_classification",
            "included",
            "primary_exclusion_reason",
            "exclusion_reasons",
            "missing_total",
            "invalid_total",
            "missing_store",
            "source_file",
        ]

        return pd.DataFrame(
            rows,
            columns=columns,
        )

    # ========================================================
    # SAVE RECEIPT TABLE
    # ========================================================

    def _save_receipt_table(
        self,
        dataframe: pd.DataFrame,
    ) -> Path:

        path = (
            self.output_dir
            / "receipt_financial_table.csv"
        )

        dataframe.to_csv(
            path,
            index=False,
            encoding="utf-8",
        )

        return path

    # ========================================================
    # SAVE AUDIT TABLE
    # ========================================================

    def _save_audit_table(
        self,
        dataframe: pd.DataFrame,
    ) -> Path:

        path = (
            self.output_dir
            / "financial_audit_table.csv"
        )

        audit = dataframe[
            ~dataframe[
                "included"
            ]
        ].copy()

        audit.to_csv(
            path,
            index=False,
            encoding="utf-8",
        )

        return path

    # ========================================================
    # SAVE STORE SPEND
    # ========================================================

    def _save_spend_per_store(
        self,
        spend_per_store: dict[str, float],
    ) -> Path:

        path = (
            self.output_dir
            / "spend_per_store.csv"
        )

        rows = [
            {
                "store_name": store,
                "total_spend": amount,
            }
            for store, amount
            in spend_per_store.items()
        ]

        dataframe = pd.DataFrame(
            rows,
            columns=[
                "store_name",
                "total_spend",
            ],
        )

        dataframe.to_csv(
            path,
            index=False,
            encoding="utf-8",
        )

        return path

    # ========================================================
    # SAVE STORE TRANSACTION COUNT
    # ========================================================

    def _save_transaction_count_per_store(
        self,
        transaction_count_per_store: dict[str, int],
    ) -> Path:

        path = (
            self.output_dir
            / "transactions_per_store.csv"
        )

        rows = [
            {
                "store_name": store,
                "transaction_count": count,
            }
            for store, count
            in transaction_count_per_store.items()
        ]

        dataframe = pd.DataFrame(
            rows,
            columns=[
                "store_name",
                "transaction_count",
            ],
        )

        dataframe.to_csv(
            path,
            index=False,
            encoding="utf-8",
        )

        return path

    # ========================================================
    # BUILD REPORT
    # ========================================================

    def _build_report_payload(
        self,
        result: FinancialSummaryResult,
    ) -> dict[str, Any]:

        return {
            "schema_version": (
                self.config.schema_version
            ),

            "status": (
                result.status
            ),

            "generated_at_utc": (
                result.generated_at_utc
            ),

            # =================================================
            # ASSIGNMENT-REQUIRED FINANCIAL OUTPUT
            # =================================================

            "financial_summary": {

                "total_spend": (
                    result.total_spend
                ),

                "transaction_count": (
                    result.transaction_count
                ),

                "spend_per_store": (
                    result.spend_per_store
                ),

                "transaction_count_per_store": (
                    result.transaction_count_per_store
                ),

                "average_transaction_value": (
                    result.average_transaction_value
                ),

                "median_transaction_value": (
                    result.median_transaction_value
                ),

                "minimum_transaction_value": (
                    result.minimum_transaction_value
                ),

                "maximum_transaction_value": (
                    result.maximum_transaction_value
                ),
            },

            # =================================================
            # AUDIT / QUALITY
            # =================================================

            "audit_summary": {

                "files_discovered": (
                    result.files_discovered
                ),

                "receipts_processed": (
                    result.receipts_processed
                ),

                "receipts_loaded": (
                    result.receipts_loaded
                ),

                "load_failures": (
                    result.load_failures
                ),

                "receipts_included": (
                    result.receipts_included
                ),

                "receipts_excluded": (
                    result.receipts_excluded
                ),

                "excluded_transactions": (
                    result.excluded_transactions
                ),

                "excluded_spend": (
                    result.excluded_spend
                ),

                "low_confidence_receipts": (
                    result.low_confidence_receipts
                ),

                "review_required_receipts": (
                    result.review_required_receipts
                ),

                "missing_total_receipts": (
                    result.missing_total_receipts
                ),

                "missing_store_receipts": (
                    result.missing_store_receipts
                ),

                "invalid_receipts": (
                    result.invalid_receipts
                ),

                "duplicate_receipts": (
                    result.duplicate_receipts
                ),

                "unassigned_store_spend": (
                    result.unassigned_store_spend
                ),

                "unassigned_store_transactions": (
                    result.unassigned_store_transactions
                ),

                "exclusion_reason_counts": (
                    result.exclusion_reason_counts
                ),

                "exclusion_spend_by_primary_reason": (
                    result.exclusion_spend_by_primary_reason
                ),
            },

            # =================================================
            # RECONCILIATION
            # =================================================

            "reconciliation": {

                "status": (
                    result.reconciliation_status
                ),

                "trusted_total_spend": (
                    result.total_spend
                ),

                "sum_of_store_spend": round(
                    sum(
                        result.spend_per_store.values()
                    ),
                    self.config.currency_decimals,
                ),

                "difference": (
                    result.reconciliation_difference
                ),
            },

            # =================================================
            # TRUST POLICY
            # =================================================

            "trust_policy": {

                "minimum_confidence": (
                    self.config.minimum_confidence
                ),

                "exclude_review_required": (
                    self.config.exclude_review_required
                ),

                "require_validation_passed": (
                    self.config.require_validation_passed
                ),

                "low_confidence_receipts_retained_for_audit": (
                    self.config.include_low_confidence_receipts
                ),
            },

            # =================================================
            # ARTIFACT REFERENCES
            # =================================================

            "artifacts": {

                "receipt_financial_table": (
                    result.receipt_table_path
                ),

                "financial_audit_table": (
                    result.audit_table_path
                ),
            },

            "artifact_errors": (
                result.artifact_errors
            ),

            "processing_time_ms": (
                result.processing_time_ms
            ),
        }

    # ========================================================
    # SAVE JSON REPORT
    # ========================================================

    def _save_json(
        self,
        payload: dict[str, Any],
    ) -> Path:

        path = (
            self.output_dir
            / "financial_summary.json"
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                payload,
                file,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )

        return path

    # ========================================================
    # BUILD RESULT
    # ========================================================

    def _build_result(
        self,
        files_discovered: int,
        start_time: float,
    ) -> FinancialSummaryResult:
        """
        Calculate trusted and audit metrics.
        """

        trusted_records = (
            self._trusted_records()
        )

        # ----------------------------------------------------
        # Trusted amounts
        # ----------------------------------------------------

        trusted_amounts = [
            record[
                "total_amount"
            ]
            for record
            in trusted_records
            if record[
                "total_amount"
            ] is not None
        ]

        total_spend_decimal = (
            self._quantize_money(
                self._sum_decimal(
                    trusted_amounts
                )
            )
        )

        transaction_count = len(
            trusted_amounts
        )

        if trusted_amounts:

            average_decimal = (
                total_spend_decimal
                / Decimal(
                    transaction_count
                )
            )

            average_decimal = (
                self._quantize_money(
                    average_decimal
                )
            )

            median_decimal = (
                self._quantize_money(
                    self._median_decimal(
                        trusted_amounts
                    )
                )
            )

            minimum_decimal = (
                self._quantize_money(
                    min(
                        trusted_amounts
                    )
                )
            )

            maximum_decimal = (
                self._quantize_money(
                    max(
                        trusted_amounts
                    )
                )
            )

        else:

            average_decimal = Decimal("0")

            median_decimal = Decimal("0")

            minimum_decimal = None

            maximum_decimal = None

        # ----------------------------------------------------
        # Store-level calculations
        # ----------------------------------------------------

        spend_per_store = (
            self._calculate_spend_per_store(
                trusted_records
            )
        )

        transaction_count_per_store = (
            self._calculate_transaction_count_per_store(
                trusted_records
            )
        )

        # ----------------------------------------------------
        # Reconciliation
        # ----------------------------------------------------

        (
            reconciliation_status,
            reconciliation_difference,
        ) = self._reconcile(
            total_spend_decimal,
            spend_per_store,
        )

        # ----------------------------------------------------
        # Audit counts
        # ----------------------------------------------------

        receipts_loaded = len(
            self._records
        )

        receipts_excluded = sum(
            not record[
                "included"
            ]
            for record
            in self._records
        )

        low_confidence_receipts = sum(
            (
                not record[
                    "confidence_missing"
                ]
                and record[
                    "overall_confidence"
                ]
                < self._minimum_confidence
            )
            for record
            in self._records
        )

        review_required_receipts = sum(
            record[
                "review_required"
            ]
            for record
            in self._records
        )

        missing_total_receipts = sum(
            record[
                "missing_total"
            ]
            for record
            in self._records
        )

        missing_store_receipts = sum(
            record[
                "missing_store"
            ]
            for record
            in self._records
        )

        invalid_receipts = sum(
            (
                not record[
                    "schema_valid"
                ]
                or not record[
                    "confidence_valid"
                ]
                or not record[
                    "reliability_valid"
                ]
            )
            for record
            in self._records
        )

        duplicate_receipts = sum(
            (
                "duplicate_receipt_id"
                in record[
                    "exclusion_reasons"
                ]
            )
            for record
            in self._records
        )

        # ----------------------------------------------------
        # Excluded financial universe
        # ----------------------------------------------------

        excluded_amounts = [
            record[
                "total_amount"
            ]
            for record
            in self._records
            if (
                not record[
                    "included"
                ]
                and record[
                    "total_amount"
                ] is not None
            )
        ]

        excluded_spend_decimal = (
            self._quantize_money(
                self._sum_decimal(
                    excluded_amounts
                )
            )
        )

        # ----------------------------------------------------
        # Exclusion reason counts
        # ----------------------------------------------------

        exclusion_reason_counts: Counter[
            str
        ] = Counter()

        exclusion_spend_by_primary_reason_decimal: defaultdict[
            str,
            Decimal
        ] = defaultdict(
            lambda: Decimal("0")
        )

        for record in self._records:

            if record[
                "included"
            ]:

                continue

            for reason in record[
                "exclusion_reasons"
            ]:

                exclusion_reason_counts[
                    reason
                ] += 1

            primary_reason = record[
                "primary_exclusion_reason"
            ]

            amount = record[
                "total_amount"
            ]

            if (
                primary_reason
                and amount is not None
            ):

                exclusion_spend_by_primary_reason_decimal[
                    primary_reason
                ] += amount

        exclusion_spend_by_primary_reason = {
            reason: (
                self._decimal_to_float(
                    self._quantize_money(
                        amount
                    )
                )
                or 0.0
            )
            for reason, amount
            in sorted(
                exclusion_spend_by_primary_reason_decimal.items()
            )
        }

        # ----------------------------------------------------
        # Unknown / unassigned store
        # ----------------------------------------------------

        unassigned_spend = Decimal("0")

        unassigned_transactions = 0

        for record in trusted_records:

            if (
                record[
                    "store_key"
                ]
                == "__UNKNOWN_STORE__"
            ):

                unassigned_transactions += 1

                if record[
                    "total_amount"
                ] is not None:

                    unassigned_spend += record[
                        "total_amount"
                    ]

        unassigned_spend = (
            self._quantize_money(
                unassigned_spend
            )
        )

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        load_failures = len(
            self._load_failures
        )

        if files_discovered == 0:

            status = "empty"

        elif (
            receipts_loaded == 0
            and load_failures > 0
        ):

            status = "failed"

        elif (
            reconciliation_status
            == "failed"
        ):

            status = "partial"

        elif (
            load_failures > 0
            or receipts_excluded > 0
            or self._artifact_errors
        ):

            status = "partial"

        else:

            status = "success"

        # ----------------------------------------------------
        # Processing time
        # ----------------------------------------------------

        processing_time_ms = round(
            (
                time.perf_counter()
                - start_time
            ) * 1000.0,
            3,
        )

        return FinancialSummaryResult(

            status=status,

            generated_at_utc=(
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),

            total_spend=(
                self._decimal_to_float(
                    total_spend_decimal
                )
                or 0.0
            ),

            transaction_count=(
                transaction_count
            ),

            receipts_processed=(
                files_discovered
            ),

            receipts_included=(
                len(
                    trusted_records
                )
            ),

            receipts_excluded=(
                receipts_excluded
            ),

            files_discovered=(
                files_discovered
            ),

            receipts_loaded=(
                receipts_loaded
            ),

            load_failures=(
                load_failures
            ),

            low_confidence_receipts=(
                low_confidence_receipts
            ),

            review_required_receipts=(
                review_required_receipts
            ),

            missing_total_receipts=(
                missing_total_receipts
            ),

            missing_store_receipts=(
                missing_store_receipts
            ),

            invalid_receipts=(
                invalid_receipts
            ),

            duplicate_receipts=(
                duplicate_receipts
            ),

            excluded_spend=(
                self._decimal_to_float(
                    excluded_spend_decimal
                )
                or 0.0
            ),

            excluded_transactions=(
                receipts_excluded
            ),

            average_transaction_value=(
                self._decimal_to_float(
                    average_decimal
                )
                or 0.0
            ),

            median_transaction_value=(
                self._decimal_to_float(
                    median_decimal
                )
                or 0.0
            ),

            minimum_transaction_value=(
                self._decimal_to_float(
                    minimum_decimal
                )
            ),

            maximum_transaction_value=(
                self._decimal_to_float(
                    maximum_decimal
                )
            ),

            spend_per_store=(
                spend_per_store
            ),

            transaction_count_per_store=(
                transaction_count_per_store
            ),

            unassigned_store_spend=(
                self._decimal_to_float(
                    unassigned_spend
                )
                or 0.0
            ),

            unassigned_store_transactions=(
                unassigned_transactions
            ),

            exclusion_reason_counts={
                reason: int(
                    count
                )
                for reason, count
                in sorted(
                    exclusion_reason_counts.items()
                )
            },

            exclusion_spend_by_primary_reason=(
                exclusion_spend_by_primary_reason
            ),

            reconciliation_status=(
                reconciliation_status
            ),

            reconciliation_difference=(
                self._decimal_to_float(
                    reconciliation_difference
                )
                or 0.0
            ),

            report_path=None,

            receipt_table_path=None,

            audit_table_path=None,

            artifact_errors=list(
                self._artifact_errors
            ),

            processing_time_ms=(
                processing_time_ms
            ),
        )

    # ========================================================
    # MAIN GENERATION
    # ========================================================

    def generate_summary(
        self,
    ) -> FinancialSummaryResult:

        start_time = (
            time.perf_counter()
        )

        # ----------------------------------------------------
        # Reset runtime state so repeated run() calls are safe.
        # ----------------------------------------------------

        self._records = []

        self._load_failures = []

        self._artifact_errors = []

        logger.info(
            "Starting production-grade financial summary..."
        )

        # ----------------------------------------------------
        # Discover files
        # ----------------------------------------------------

        files = (
            self._discover_files()
        )

        files_discovered = len(
            files
        )

        # ----------------------------------------------------
        # Load every receipt.
        #
        # A single corrupted JSON must NOT stop the complete batch.
        # ----------------------------------------------------

        for source_path in files:

            try:

                data = (
                    self._load_json(
                        source_path
                    )
                )

                record = (
                    self._normalize_receipt(
                        data,
                        source_path,
                    )
                )

                self._records.append(
                    record
                )

            except Exception as exc:

                logger.exception(
                    "Failed to load receipt JSON: %s",
                    source_path,
                )

                self._load_failures.append(
                    {
                        "source_file": str(
                            source_path
                        ),

                        "error": str(
                            exc
                        ),
                    }
                )

        # ----------------------------------------------------
        # Duplicate detection happens before trusted inclusion.
        # ----------------------------------------------------

        self._apply_duplicate_policy(
            self._records
        )

        # ----------------------------------------------------
        # Final inclusion/exclusion decision.
        # ----------------------------------------------------

        for record in self._records:

            self._finalize_record_decision(
                record
            )

        # ----------------------------------------------------
        # Calculate trusted and audit metrics.
        # ----------------------------------------------------

        result = self._build_result(
            files_discovered=(
                files_discovered
            ),
            start_time=start_time,
        )

        # ----------------------------------------------------
        # Receipt-level dataframe
        # ----------------------------------------------------

        dataframe = (
            self._build_receipt_dataframe(
                self._records
            )
        )

        # ----------------------------------------------------
        # Save full receipt financial table.
        # ----------------------------------------------------

        if self.config.save_receipt_table:

            try:

                receipt_table_path = (
                    self._save_receipt_table(
                        dataframe
                    )
                )

                result.receipt_table_path = (
                    str(
                        receipt_table_path
                    )
                )

            except Exception as exc:

                message = (
                    f"Receipt table save failed: {exc}"
                )

                logger.exception(
                    message
                )

                self._artifact_errors.append(
                    message
                )

        # ----------------------------------------------------
        # Save audit table.
        #
        # Contains excluded records only.
        # ----------------------------------------------------

        if (
            self.config.save_audit_table
            and self.config.include_low_confidence_receipts
        ):

            try:

                audit_table_path = (
                    self._save_audit_table(
                        dataframe
                    )
                )

                result.audit_table_path = (
                    str(
                        audit_table_path
                    )
                )

            except Exception as exc:

                message = (
                    f"Audit table save failed: {exc}"
                )

                logger.exception(
                    message
                )

                self._artifact_errors.append(
                    message
                )

        # ----------------------------------------------------
        # Save CSV summaries.
        # ----------------------------------------------------

        if self.config.save_csv:

            try:

                self._save_spend_per_store(
                    result.spend_per_store
                )

            except Exception as exc:

                message = (
                    f"Spend-per-store CSV save failed: {exc}"
                )

                logger.exception(
                    message
                )

                self._artifact_errors.append(
                    message
                )

            try:

                self._save_transaction_count_per_store(
                    result.transaction_count_per_store
                )

            except Exception as exc:

                message = (
                    "Transaction-count-per-store "
                    f"CSV save failed: {exc}"
                )

                logger.exception(
                    message
                )

                self._artifact_errors.append(
                    message
                )

        # ----------------------------------------------------
        # Update artifact errors before creating final report.
        # ----------------------------------------------------

        result.artifact_errors = list(
            self._artifact_errors
        )

        if self._artifact_errors:

            if result.status == "success":

                result.status = "partial"

        # ----------------------------------------------------
        # Build FINAL report after all earlier artifact paths/errors
        # are known.
        # ----------------------------------------------------

        report_payload = (
            self._build_report_payload(
                result
            )
        )

        # ----------------------------------------------------
        # Save final JSON report LAST.
        # ----------------------------------------------------

        if self.config.save_json:

            try:

                report_path = (
                    self._save_json(
                        report_payload
                    )
                )

                result.report_path = (
                    str(
                        report_path
                    )
                )

            except Exception as exc:

                message = (
                    f"Financial summary JSON save failed: {exc}"
                )

                logger.exception(
                    message
                )

                self._artifact_errors.append(
                    message
                )

                result.artifact_errors = list(
                    self._artifact_errors
                )

                if result.status == "success":

                    result.status = "partial"

        logger.info(
            "Financial summary completed. "
            "Status=%s | Trusted spend=%.2f | "
            "Transactions=%d | Included=%d | Excluded=%d",
            result.status,
            result.total_spend,
            result.transaction_count,
            result.receipts_included,
            result.receipts_excluded,
        )

        return result

    # ========================================================
    # PUBLIC ENTRY POINT
    # ========================================================

    def run(
        self,
    ) -> FinancialSummaryResult:

        return self.generate_summary()