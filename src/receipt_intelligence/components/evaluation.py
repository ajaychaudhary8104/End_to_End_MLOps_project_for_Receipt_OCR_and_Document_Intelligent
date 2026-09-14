# ============================================================
# PRODUCTION-GRADE EVALUATION ENGINE
# Receipt Intelligence / IDP Project
# ============================================================
#
# Evaluates:
#   1. OCR quality
#        - CER
#        - WER
#        - corpus CER/WER
#
#   2. Required field extraction
#        - Store accuracy
#        - Date accuracy
#        - Total amount accuracy
#
#   3. Item extraction
#        - Precision
#        - Recall
#        - F1
#        - Micro + Macro metrics
#
#   4. Prediction coverage
#        - Missing predictions
#        - Extra predictions
#        - Duplicate prediction IDs
#        - Duplicate ground-truth IDs
#        - Load failures
#
#   5. Confidence analysis
#        - Mean confidence
#        - Review rate
#        - Confidence calibration / ECE
#        - Confidence buckets
#
#   6. Production safeguards
#        - Never treats missing == correct
#        - Uses union of prediction/GT IDs
#        - Matches GT by receipt_id, not only filename
#        - Does not treat missing GT text as empty text
#        - Does not silently clip invalid confidence
#        - Explicit boolean normalization
#        - Robust date validation
#        - Decimal-based monetary comparison
#        - Duplicate detection
#        - Atomic artifact writes
#        - Partial / failed / no-ground-truth status
#        - Per-receipt audit trail
#
# ============================================================

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

# Optional dependency.
# Exact one-to-one item matching uses scipy when available.
# A deterministic greedy fallback is provided otherwise.
try:
    from scipy.optimize import linear_sum_assignment

    SCIPY_AVAILABLE = True

except ImportError:
    linear_sum_assignment = None
    SCIPY_AVAILABLE = False

from src.receipt_intelligence.entity.config_entity import EvaluationConfig
from src.receipt_intelligence import logger

# ============================================================
# CLASSIFICATION COUNTS
# ============================================================

@dataclass
class ClassificationCounts:
    """
    TP / FP / FN for item matching.
    """

    tp: int = 0

    fp: int = 0

    fn: int = 0


# ============================================================
# PER-RECEIPT EVALUATION RESULT
# ============================================================

@dataclass
class ReceiptEvaluationResult:
    """
    Detailed evaluation result for one logical receipt ID.
    """

    receipt_id: str

    prediction_file: str | None

    ground_truth_file: str | None

    prediction_available: bool

    ground_truth_available: bool

    prediction_status: str

    ground_truth_status: str

    # --------------------------------------------------------
    # OCR
    # --------------------------------------------------------

    cer: float | None

    wer: float | None

    cer_edit_distance: int | None

    cer_reference_characters: int | None

    wer_edit_distance: int | None

    wer_reference_words: int | None

    # --------------------------------------------------------
    # Required fields
    # --------------------------------------------------------

    store_name_correct: bool | None

    date_correct: bool | None

    total_amount_correct: bool | None

    required_fields_accuracy: float | None

    required_fields_all_correct: bool | None

    # --------------------------------------------------------
    # Items
    # --------------------------------------------------------

    item_precision: float | None

    item_recall: float | None

    item_f1: float | None

    item_tp: int | None

    item_fp: int | None

    item_fn: int | None

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    prediction_confidence: float | None

    review_required: bool | None

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    errors: list[str]

    warnings: list[str]

    processing_time_ms: float


# ============================================================
# TEXT NORMALIZATION
# ============================================================

class TextNormalizer:
    """
    Deterministic text normalization.
    """

    @staticmethod
    def normalize(
        text: Any,
        case_sensitive: bool = False,
        normalize_whitespace: bool = True,
        remove_punctuation: bool = False,
    ) -> str:

        if text is None:
            return ""

        text = str(text)

        if normalize_whitespace:

            text = re.sub(
                r"\s+",
                " ",
                text,
            ).strip()

        if not case_sensitive:

            text = text.casefold()

        if remove_punctuation:

            text = re.sub(
                r"[^\w\s]",
                "",
                text,
                flags=re.UNICODE,
            )

            text = re.sub(
                r"\s+",
                " ",
                text,
            ).strip()

        return text


# ============================================================
# SEQUENCE METRICS
# ============================================================

class SequenceMetrics:
    """
    Levenshtein and OCR sequence metrics.
    """

    @staticmethod
    def levenshtein(
        reference: list[Any] | str,
        hypothesis: list[Any] | str,
    ) -> int:
        """
        Memory-efficient two-row Levenshtein distance.
        """

        if len(reference) < len(hypothesis):

            reference, hypothesis = (
                hypothesis,
                reference,
            )

        if len(hypothesis) == 0:

            return len(reference)

        previous = list(
            range(
                len(hypothesis) + 1
            )
        )

        for i, ref_value in enumerate(
            reference,
            start=1,
        ):

            current = [i]

            for j, hyp_value in enumerate(
                hypothesis,
                start=1,
            ):

                insertion = (
                    current[j - 1]
                    + 1
                )

                deletion = (
                    previous[j]
                    + 1
                )

                substitution = (
                    previous[j - 1]
                    + (
                        0
                        if ref_value == hyp_value
                        else 1
                    )
                )

                current.append(
                    min(
                        insertion,
                        deletion,
                        substitution,
                    )
                )

            previous = current

        return previous[-1]

    @staticmethod
    def cer_stats(
        reference: str,
        hypothesis: str,
    ) -> tuple[float, int, int]:
        """
        Returns:

            CER,
            edit distance,
            reference character count
        """

        distance = SequenceMetrics.levenshtein(
            reference,
            hypothesis,
        )

        reference_length = len(
            reference
        )

        denominator = max(
            1,
            reference_length,
        )

        cer = (
            distance
            / denominator
        )

        return (
            float(cer),
            int(distance),
            int(reference_length),
        )

    @staticmethod
    def wer_stats(
        reference: str,
        hypothesis: str,
    ) -> tuple[float, int, int]:
        """
        Returns:

            WER,
            edit distance,
            reference word count
        """

        reference_words = (
            reference.split()
        )

        hypothesis_words = (
            hypothesis.split()
        )

        distance = SequenceMetrics.levenshtein(
            reference_words,
            hypothesis_words,
        )

        reference_word_count = len(
            reference_words
        )

        denominator = max(
            1,
            reference_word_count,
        )

        wer = (
            distance
            / denominator
        )

        return (
            float(wer),
            int(distance),
            int(reference_word_count),
        )


# ============================================================
# FIELD METRICS
# ============================================================

class FieldMetrics:
    """
    Precision / Recall / F1 utilities.
    """

    @staticmethod
    def precision(
        tp: int,
        fp: int,
    ) -> float:

        denominator = (
            tp + fp
        )

        if denominator == 0:
            return 0.0

        return (
            tp
            / denominator
        )

    @staticmethod
    def recall(
        tp: int,
        fn: int,
    ) -> float:

        denominator = (
            tp + fn
        )

        if denominator == 0:
            return 0.0

        return (
            tp
            / denominator
        )

    @staticmethod
    def f1(
        precision: float,
        recall: float,
    ) -> float:

        denominator = (
            precision + recall
        )

        if denominator == 0:
            return 0.0

        return (
            2.0
            * precision
            * recall
            / denominator
        )


# ============================================================
# EVALUATION ENGINE
# ============================================================

class EvaluationEngine:
    """
    Production-grade receipt evaluation framework.

    IMPORTANT DESIGN RULES
    ----------------------
    1. Missing prediction != correct.
    2. Missing ground truth != incorrect.
       It is simply "not evaluable".
    3. Prediction and ground truth are matched by receipt_id.
    4. Missing prediction IDs are evaluated explicitly.
    5. Duplicate IDs are never silently selected.
    6. OCR metrics use an actual OCR artifact whenever possible.
    7. Invalid confidence is not silently clipped.
    8. Amount comparison uses Decimal.
    9. Date parsing validates real calendar dates.
    10. Item matching is one-to-one.
    """

    EXCLUDED_PREDICTION_FILES = frozenset(
        {
            "json_manifest.json",
            "evaluation_report.json",
            "evaluation_details.json",
        }
    )

    # ========================================================
    # INIT
    # ========================================================

    def __init__(
        self,
        config: EvaluationConfig,
    ):
        self.config = config

        self.prediction_dir = Path(
            config.prediction_dir
        )

        self.ground_truth_dir = Path(
            config.ground_truth_dir
        )

        self.ocr_prediction_dir = Path(
            config.ocr_prediction_dir
        )

        self.output_dir = Path(
            config.output_dir
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._validate_config()

        self._prediction_load_failures: list[
            dict[str, Any]
        ] = []

        self._ground_truth_load_failures: list[
            dict[str, Any]
        ] = []

        self._ocr_load_failures: list[
            dict[str, Any]
        ] = []

        self._duplicate_prediction_ids: set[
            str
        ] = set()

        self._duplicate_ground_truth_ids: set[
            str
        ] = set()

        self._duplicate_ocr_ids: set[
            str
        ] = set()

    # ========================================================
    # CONFIG VALIDATION
    # ========================================================

    def _validate_config(
        self,
    ) -> None:

        if (
            self.config.amount_tolerance
            < 0
        ):

            raise ValueError(
                "amount_tolerance cannot be negative."
            )

        if (
            self.config.item_price_tolerance
            < 0
        ):

            raise ValueError(
                "item_price_tolerance cannot be negative."
            )

        if not (
            0.0
            <= self.config.item_name_similarity_threshold
            <= 1.0
        ):

            raise ValueError(
                "item_name_similarity_threshold must be "
                "between 0 and 1."
            )

        if not (
            0.0
            <= self.config.string_similarity_threshold
            <= 1.0
        ):

            raise ValueError(
                "string_similarity_threshold must be "
                "between 0 and 1."
            )

        if not (
            0
            <= self.config.confidence_bin_count
            <= 100
        ):

            raise ValueError(
                "confidence_bin_count must be "
                "between 1 and 100."
            )

        if self.config.confidence_bin_count == 0:

            raise ValueError(
                "confidence_bin_count must be greater than 0."
            )

        if (
            self.config.date_order.upper()
            not in {
                "DMY",
                "MDY",
                "YMD",
            }
        ):

            raise ValueError(
                "date_order must be one of DMY, MDY, or YMD."
            )

        if (
            self.config.cer_warning_threshold
            < 0
        ):

            raise ValueError(
                "cer_warning_threshold cannot be negative."
            )

        if (
            self.config.wer_warning_threshold
            < 0
        ):

            raise ValueError(
                "wer_warning_threshold cannot be negative."
            )

        if not (
            0.0
            <= self.config.item_f1_warning_threshold
            <= 1.0
        ):

            raise ValueError(
                "item_f1_warning_threshold must be between 0 and 1."
            )

        quality_thresholds = {
            "min_store_accuracy": (
                self.config.min_store_accuracy
            ),
            "min_date_accuracy": (
                self.config.min_date_accuracy
            ),
            "min_total_accuracy": (
                self.config.min_total_accuracy
            ),
            "min_item_f1": (
                self.config.min_item_f1
            ),
            "max_corpus_cer": (
                self.config.max_corpus_cer
            ),
            "max_corpus_wer": (
                self.config.max_corpus_wer
            ),
        }

        for name, value in quality_thresholds.items():

            if value is None:
                continue

            if not (
                0.0
                <= value
                <= 1.0
            ):

                raise ValueError(
                    f"{name} must be between 0 and 1."
                )

    # ========================================================
    # DISCOVER JSON FILES
    # ========================================================

    def _discover_json_files(
        self,
        directory: Path,
        excluded_names: set[str] | frozenset[str],
    ) -> list[Path]:
        """
        Discover JSON files recursively.
        """

        if not directory.exists():
            return []

        if not directory.is_dir():

            raise NotADirectoryError(
                f"Expected directory: {directory}"
            )

        return sorted(
            path
            for path in directory.rglob(
                "*.json"
            )
            if path.name
            not in excluded_names
        )

    def _discover_predictions(
        self,
    ) -> list[Path]:

        if not self.prediction_dir.exists():

            raise FileNotFoundError(
                f"Prediction directory does not exist: "
                f"{self.prediction_dir}"
            )

        files = self._discover_json_files(
            self.prediction_dir,
            self.EXCLUDED_PREDICTION_FILES,
        )

        logger.info(
            "Discovered %d prediction files.",
            len(files),
        )

        return files

    def _discover_ground_truth(
        self,
    ) -> list[Path]:

        if not self.ground_truth_dir.exists():

            return []

        files = self._discover_json_files(
            self.ground_truth_dir,
            frozenset(),
        )

        logger.info(
            "Discovered %d ground-truth files.",
            len(files),
        )

        return files

    def _discover_ocr_files(
        self,
    ) -> list[Path]:

        if not self.ocr_prediction_dir.exists():

            return []

        files = self._discover_json_files(
            self.ocr_prediction_dir,
            frozenset(),
        )

        logger.info(
            "Discovered %d OCR evidence files.",
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
                f"Expected JSON object: {path}"
            )

        return data

    # ========================================================
    # RECEIPT ID EXTRACTION
    # ========================================================

    @staticmethod
    def _receipt_id_from_data(
        data: dict[str, Any],
        path: Path,
    ) -> str:

        # Metadata-first.
        metadata = data.get(
            "metadata"
        )

        if isinstance(
            metadata,
            dict,
        ):

            value = metadata.get(
                "receipt_id"
            )

            if value is not None:

                text = str(
                    value
                ).strip()

                if text:
                    return text

        # Top-level fallback.
        value = data.get(
            "receipt_id"
        )

        if value is not None:

            text = str(
                value
            ).strip()

            if text:
                return text

        # Filename fallback.
        return path.stem

    # ========================================================
    # BUILD FILE INDEX
    # ========================================================

    def _build_file_index(
        self,
        files: list[Path],
        kind: str,
    ) -> tuple[
        dict[str, tuple[Path, dict[str, Any]]],
        set[str],
        list[dict[str, Any]],
    ]:
        """
        Returns:

            unique_index
            duplicate_ids
            load_failures
        """

        index: dict[
            str,
            tuple[Path, dict[str, Any]]
        ] = {}

        duplicate_ids: set[
            str
        ] = set()

        failures: list[
            dict[str, Any]
        ] = []

        # Keep first-seen mapping temporarily.
        seen_paths: dict[
            str,
            Path
        ] = {}

        for path in files:

            try:

                data = self._load_json(
                    path
                )

                receipt_id = (
                    self._receipt_id_from_data(
                        data,
                        path,
                    )
                )

                if receipt_id in index:

                    duplicate_ids.add(
                        receipt_id
                    )

                    # Remove ambiguous ID from unique index.
                    index.pop(
                        receipt_id,
                        None,
                    )

                    logger.error(
                        "Duplicate %s receipt_id detected: %s",
                        kind,
                        receipt_id,
                    )

                    continue

                if receipt_id in duplicate_ids:

                    logger.error(
                        "Additional duplicate %s receipt_id: %s",
                        kind,
                        receipt_id,
                    )

                    continue

                index[
                    receipt_id
                ] = (
                    path,
                    data,
                )

                seen_paths[
                    receipt_id
                ] = path

            except Exception as exc:

                logger.exception(
                    "Failed to load %s file: %s",
                    kind,
                    path,
                )

                failures.append(
                    {
                        "file": str(
                            path
                        ),
                        "error": str(
                            exc
                        ),
                    }
                )

        return (
            index,
            duplicate_ids,
            failures,
        )

    # ========================================================
    # NORMALIZE TEXT
    # ========================================================

    def _normalize_text(
        self,
        value: Any,
        *,
        punctuation: bool | None = None,
    ) -> str:

        if punctuation is None:

            punctuation = (
                self.config.remove_punctuation_for_text_eval
            )

        return TextNormalizer.normalize(
            value,
            case_sensitive=(
                self.config.case_sensitive
            ),
            normalize_whitespace=(
                self.config.normalize_whitespace
            ),
            remove_punctuation=punctuation,
        )

    # ========================================================
    # NORMALIZE DATE
    # ========================================================

    def _normalize_date(
        self,
        value: Any,
    ) -> str | None:

        if value is None:
            return None

        text = str(
            value
        ).strip()

        if not text:
            return None

        # ----------------------------------------------------
        # ISO YYYY-MM-DD
        # ----------------------------------------------------

        if re.fullmatch(
            r"\d{4}-\d{2}-\d{2}",
            text,
        ):

            try:

                parsed = datetime.strptime(
                    text,
                    "%Y-%m-%d",
                )

                return parsed.strftime(
                    "%Y-%m-%d"
                )

            except ValueError:

                return None

        # ----------------------------------------------------
        # Full year first: YYYY/MM/DD, YYYY-MM-DD, YYYY.MM.DD
        # ----------------------------------------------------

        ymd_match = re.fullmatch(
            r"(\d{4})"
            r"[\/\-.]"
            r"(\d{1,2})"
            r"[\/\-.]"
            r"(\d{1,2})",
            text,
        )

        if ymd_match:

            try:

                year = int(
                    ymd_match.group(1)
                )

                month = int(
                    ymd_match.group(2)
                )

                day = int(
                    ymd_match.group(3)
                )

                return datetime(
                    year,
                    month,
                    day,
                ).strftime(
                    "%Y-%m-%d"
                )

            except ValueError:

                return None

        # ----------------------------------------------------
        # DMY / MDY forms
        # ----------------------------------------------------

        match = re.fullmatch(
            r"(\d{1,2})"
            r"[\/\-.]"
            r"(\d{1,2})"
            r"[\/\-.]"
            r"(\d{2,4})",
            text,
        )

        if not match:
            return None

        first = int(
            match.group(1)
        )

        second = int(
            match.group(2)
        )

        year = int(
            match.group(3)
        )

        if year < 100:
            year += 2000

        date_order = (
            self.config.date_order.upper()
        )

        if date_order == "DMY":

            day = first
            month = second

        elif date_order == "MDY":

            month = first
            day = second

        else:

            # YMD with 2-digit first component is not accepted
            # because year ambiguity is unsafe.
            return None

        try:

            parsed = datetime(
                year,
                month,
                day,
            )

            return parsed.strftime(
                "%Y-%m-%d"
            )

        except ValueError:

            return None

    # ========================================================
    # NORMALIZE AMOUNT
    # ========================================================

    @staticmethod
    def _normalize_amount(
        value: Any,
    ) -> Decimal | None:
        """
        Decimal-based monetary normalization.
        """

        if value is None:
            return None

        if isinstance(
            value,
            bool,
        ):
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

            # Parenthesized negatives are rejected for purchase
            # receipt evaluation.
            if (
                text.startswith("(")
                and text.endswith(")")
            ):

                return None

            value = text

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

        except (
            InvalidOperation,
            TypeError,
            ValueError,
        ):

            return None

        if not number.is_finite():
            return None

        if number < 0:
            return None

        try:

            return number.quantize(
                Decimal("0.01")
            )

        except InvalidOperation:

            return None

    # ========================================================
    # AMOUNT MATCH
    # ========================================================

    def _amount_matches(
        self,
        prediction: Any,
        ground_truth: Any,
    ) -> bool | None:
        """
        Tri-state result:

            True  = evaluated and correct
            False = evaluated and incorrect
            None  = not evaluable
        """

        # Missing GT => not evaluable.
        if (
            ground_truth is None
        ):
            return None

        truth = self._normalize_amount(
            ground_truth
        )

        # GT exists but is malformed.
        if truth is None:
            return None

        predicted = self._normalize_amount(
            prediction
        )

        if predicted is None:
            return False

        tolerance = Decimal(
            str(
                self.config.amount_tolerance
            )
        )

        return (
            abs(
                predicted - truth
            )
            <= tolerance
        )

    # ========================================================
    # STRING MATCH
    # ========================================================

    def _string_matches(
        self,
        prediction: Any,
        ground_truth: Any,
    ) -> bool | None:
        """
        Tri-state string comparison.
        """

        if ground_truth is None:
            return None

        truth = self._normalize_text(
            ground_truth
        )

        if not truth:

            # Ground truth explicitly empty.
            prediction_text = self._normalize_text(
                prediction
            )

            if not prediction_text:
                return True

            return False

        predicted = self._normalize_text(
            prediction
        )

        if not predicted:
            return False

        if self.config.strict_string_matching:

            return (
                predicted
                == truth
            )

        similarity = SequenceMatcher(
            None,
            predicted,
            truth,
        ).ratio()

        return (
            similarity
            >= self.config.string_similarity_threshold
        )

    # ========================================================
    # ITEM NAME SIMILARITY
    # ========================================================

    def _item_name_similarity(
        self,
        prediction: str,
        ground_truth: str,
    ) -> float:

        predicted = self._normalize_text(
            prediction
        )

        truth = self._normalize_text(
            ground_truth
        )

        if not predicted or not truth:
            return 0.0

        distance = (
            SequenceMetrics.levenshtein(
                predicted,
                truth,
            )
        )

        denominator = max(
            1,
            len(predicted),
            len(truth),
        )

        return max(
            0.0,
            1.0
            - (
                distance
                / denominator
            ),
        )

    # ========================================================
    # ITEM MATCHING
    # ========================================================

    def _match_items(
        self,
        predicted_items: Any,
        ground_truth_items: Any,
    ) -> ClassificationCounts:
        """
        One-to-one item matching.

        Primary method:
            scipy.optimize.linear_sum_assignment

        Fallback:
            deterministic greedy matching

        A match requires:
            - name similarity >= threshold
            - if GT price exists, prediction price must exist
              and match within configured tolerance
        """

        if not isinstance(
            predicted_items,
            list,
        ):

            predicted_items = []

        if not isinstance(
            ground_truth_items,
            list,
        ):

            ground_truth_items = []

        predictions = [
            item
            for item in predicted_items
            if isinstance(
                item,
                dict,
            )
        ]

        truths = [
            item
            for item in ground_truth_items
            if isinstance(
                item,
                dict,
            )
        ]

        counts = ClassificationCounts()

        if not predictions and not truths:

            return counts

        # ----------------------------------------------------
        # Valid candidate pair scores.
        # ----------------------------------------------------

        candidate_scores: dict[
            tuple[int, int],
            float
        ] = {}

        for pred_index, prediction in enumerate(
            predictions
        ):

            predicted_name = str(
                prediction.get(
                    "name",
                    ""
                )
            ).strip()

            predicted_price = (
                self._normalize_amount(
                    prediction.get(
                        "price"
                    )
                )
            )

            for truth_index, truth in enumerate(
                truths
            ):

                truth_name = str(
                    truth.get(
                        "name",
                        ""
                    )
                ).strip()

                truth_price = (
                    self._normalize_amount(
                        truth.get(
                            "price"
                        )
                    )
                )

                name_similarity = (
                    self._item_name_similarity(
                        predicted_name,
                        truth_name,
                    )
                )

                if (
                    name_similarity
                    < self.config.item_name_similarity_threshold
                ):
                    continue

                # ------------------------------------------------
                # Price rule.
                # ------------------------------------------------

                if truth_price is None:

                    price_match = True

                else:

                    if predicted_price is None:
                        continue

                    tolerance = Decimal(
                        str(
                            self.config.item_price_tolerance
                        )
                    )

                    price_match = (
                        abs(
                            predicted_price
                            - truth_price
                        )
                        <= tolerance
                    )

                    if not price_match:
                        continue

                # Name is primary.
                # Price agreement is a small deterministic tie-breaker.
                score = (
                    float(
                        name_similarity
                    )
                    + (
                        0.01
                        if price_match
                        else 0.0
                    )
                )

                candidate_scores[
                    (
                        pred_index,
                        truth_index,
                    )
                ] = score

        # ----------------------------------------------------
        # No valid candidates.
        # ----------------------------------------------------

        if not candidate_scores:

            counts.fp = len(
                predictions
            )

            counts.fn = len(
                truths
            )

            return counts

        matched_predictions: set[
            int
        ] = set()

        matched_truths: set[
            int
        ] = set()

        # ----------------------------------------------------
        # Exact linear assignment when scipy is available.
        # ----------------------------------------------------

        if SCIPY_AVAILABLE:

            prediction_count = len(
                predictions
            )

            truth_count = len(
                truths
            )

            score_matrix = np.zeros(
                (
                    prediction_count,
                    truth_count,
                ),
                dtype=float,
            )

            for (
                pred_index,
                truth_index,
            ), score in candidate_scores.items():

                score_matrix[
                    pred_index,
                    truth_index
                ] = score

            rows, columns = (
                linear_sum_assignment(
                    -score_matrix
                )
            )

            for (
                pred_index,
                truth_index,
            ) in zip(
                rows,
                columns,
            ):

                score = score_matrix[
                    pred_index,
                    truth_index
                ]

                if score <= 0.0:
                    continue

                if (
                    (
                        pred_index,
                        truth_index,
                    )
                    not in candidate_scores
                ):
                    continue

                matched_predictions.add(
                    int(
                        pred_index
                    )
                )

                matched_truths.add(
                    int(
                        truth_index
                    )
                )

        else:

            # ------------------------------------------------
            # Deterministic greedy fallback.
            # ------------------------------------------------

            ranked_pairs = sorted(
                candidate_scores.items(),
                key=lambda pair: (
                    -pair[1],
                    pair[0][0],
                    pair[0][1],
                ),
            )

            for (
                (
                    pred_index,
                    truth_index,
                ),
                _score,
            ) in ranked_pairs:

                if pred_index in matched_predictions:
                    continue

                if truth_index in matched_truths:
                    continue

                matched_predictions.add(
                    pred_index
                )

                matched_truths.add(
                    truth_index
                )

        counts.tp = len(
            matched_predictions
        )

        counts.fp = max(
            0,
            len(predictions)
            - counts.tp,
        )

        counts.fn = max(
            0,
            len(truths)
            - counts.tp,
        )

        return counts

    # ========================================================
    # ITEM METRICS
    # ========================================================

    @staticmethod
    def _item_metrics(
        counts: ClassificationCounts,
        predicted_count: int,
        truth_count: int,
    ) -> tuple[
        float,
        float,
        float,
    ]:
        """
        Receipt-level item metric policy.

        Both empty:
            P = 1
            R = 1
            F1 = 1

        GT empty, prediction non-empty:
            P = 0
            R = 0
            F1 = 0

        GT non-empty, prediction empty:
            P = 0
            R = 0
            F1 = 0

        Otherwise:
            standard metrics.
        """

        if (
            predicted_count == 0
            and truth_count == 0
        ):

            return (
                1.0,
                1.0,
                1.0,
            )

        if truth_count == 0:

            return (
                0.0,
                0.0,
                0.0,
            )

        if predicted_count == 0:

            return (
                0.0,
                0.0,
                0.0,
            )

        precision = (
            FieldMetrics.precision(
                counts.tp,
                counts.fp,
            )
        )

        recall = (
            FieldMetrics.recall(
                counts.tp,
                counts.fn,
            )
        )

        f1 = (
            FieldMetrics.f1(
                precision,
                recall,
            )
        )

        return (
            float(precision),
            float(recall),
            float(f1),
        )

    # ========================================================
    # PREDICTION CONFIDENCE
    # ========================================================

    @staticmethod
    def _prediction_confidence(
        prediction: dict[str, Any],
    ) -> tuple[
        float | None,
        list[str],
    ]:

        warnings: list[str] = []

        if (
            "overall_confidence"
            not in prediction
        ):

            return (
                None,
                warnings,
            )

        value = prediction.get(
            "overall_confidence"
        )

        if value is None:

            return (
                None,
                warnings,
            )

        try:

            confidence = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):

            warnings.append(
                "invalid_prediction_confidence"
            )

            return (
                None,
                warnings,
            )

        if not np.isfinite(
            confidence
        ):

            warnings.append(
                "invalid_prediction_confidence"
            )

            return (
                None,
                warnings,
            )

        if not (
            0.0
            <= confidence
            <= 1.0
        ):

            warnings.append(
                "prediction_confidence_out_of_range"
            )

            return (
                None,
                warnings,
            )

        return (
            float(confidence),
            warnings,
        )

    # ========================================================
    # BOOLEAN NORMALIZATION
    # ========================================================

    @staticmethod
    def _normalize_bool(
        value: Any,
        default: bool = False,
    ) -> bool:

        if isinstance(
            value,
            bool,
        ):

            return value

        if isinstance(
            value,
            (int, float),
        ) and not isinstance(
            value,
            bool,
        ):

            if value == 1:
                return True

            if value == 0:
                return False

        if isinstance(
            value,
            str,
        ):

            text = (
                value
                .strip()
                .casefold()
            )

            if text in {
                "true",
                "yes",
                "y",
                "1",
            }:

                return True

            if text in {
                "false",
                "no",
                "n",
                "0",
            }:

                return False

        return default

    # ========================================================
    # VALIDATE FINAL PREDICTION SCHEMA
    # ========================================================

    def _prediction_schema_warnings(
        self,
        prediction: dict[str, Any],
    ) -> list[str]:
        """
        Defense-in-depth checks for prediction JSON.
        """

        warnings: list[str] = []

        for field_name in (
            "store_name",
            "date",
            "items",
            "total_amount",
        ):

            if field_name not in prediction:

                warnings.append(
                    f"prediction_missing_field:{field_name}"
                )

        items = prediction.get(
            "items"
        )

        if items is not None:

            if not isinstance(
                items,
                list,
            ):

                warnings.append(
                    "prediction_items_not_list"
                )

            else:

                for index, item in enumerate(
                    items
                ):

                    if not isinstance(
                        item,
                        dict,
                    ):

                        warnings.append(
                            f"prediction_item_{index}_not_object"
                        )

        return warnings

    # ========================================================
    # OCR TEXT EXTRACTION
    # ========================================================

    def _extract_ocr_text(
        self,
        ocr_data: dict[str, Any] | None,
    ) -> tuple[
        str | None,
        str | None,
    ]:
        """
        Returns:

            text,
            source_field
        """

        if not ocr_data:
            return (
                None,
                None,
            )

        candidate_fields = (
            self.config.ocr_prediction_text_field,
            *self.config.ocr_prediction_fallback_fields,
        )

        seen_fields: set[
            str
        ] = set()

        for field_name in candidate_fields:

            if field_name in seen_fields:
                continue

            seen_fields.add(
                field_name
            )

            if field_name not in ocr_data:
                continue

            value = ocr_data.get(
                field_name
            )

            if value is None:
                continue

            text = str(
                value
            )

            return (
                text,
                field_name,
            )

        # ----------------------------------------------------
        # Region fallback.
        # ----------------------------------------------------

        regions = ocr_data.get(
            "regions"
        )

        if isinstance(
            regions,
            list,
        ):

            prefer_original = (
                self.config.ocr_prediction_text_field
                == "original_full_text"
            )

            region_texts: list[str] = []

            for region in regions:

                if not isinstance(
                    region,
                    dict,
                ):
                    continue

                if prefer_original:

                    value = region.get(
                        "original_text",
                        region.get(
                            "text",
                            region.get(
                                "cleaned_text",
                                ""
                            )
                        ),
                    )

                else:

                    value = region.get(
                        "cleaned_text",
                        region.get(
                            "text",
                            region.get(
                                "original_text",
                                ""
                            )
                        ),
                    )

                if value is None:
                    continue

                text = str(
                    value
                ).strip()

                if text:
                    region_texts.append(
                        text
                    )

            if region_texts:

                return (
                    "\n".join(
                        region_texts
                    ),
                    "regions",
                )

        return (
            None,
            None,
        )

    # ========================================================
    # GROUND-TRUTH TEXT
    # ========================================================

    def _ground_truth_text(
        self,
        ground_truth: dict[str, Any],
    ) -> tuple[
        str | None,
        bool,
    ]:
        """
        Returns:

            text,
            field_present

        An explicitly empty string is valid ground truth text.
        A missing field is not evaluable.
        """

        field_name = (
            self.config.text_field_in_ground_truth
        )

        if field_name not in ground_truth:

            return (
                None,
                False,
            )

        value = ground_truth.get(
            field_name
        )

        if value is None:

            return (
                None,
                True,
            )

        return (
            str(value),
            True,
        )

    # ========================================================
    # EVALUATE OCR TEXT
    # ========================================================

    def _evaluate_ocr(
        self,
        prediction_data: dict[str, Any] | None,
        ground_truth_data: dict[str, Any] | None,
        ocr_data: dict[str, Any] | None,
    ) -> tuple[
        float | None,
        float | None,
        int | None,
        int | None,
        int | None,
        int | None,
        list[str],
    ]:

        warnings: list[str] = []

        if ground_truth_data is None:

            return (
                None,
                None,
                None,
                None,
                None,
                None,
                warnings,
            )

        truth_text, truth_present = (
            self._ground_truth_text(
                ground_truth_data
            )
        )

        if not truth_present:

            warnings.append(
                "ground_truth_ocr_text_unavailable"
            )

            return (
                None,
                None,
                None,
                None,
                None,
                None,
                warnings,
            )

        if truth_text is None:

            warnings.append(
                "ground_truth_ocr_text_null"
            )

            return (
                None,
                None,
                None,
                None,
                None,
                None,
                warnings,
            )

        prediction_text = None

        source_name = None

        if ocr_data is not None:

            (
                prediction_text,
                source_name,
            ) = self._extract_ocr_text(
                ocr_data
            )

            if prediction_text is not None:

                logger.debug(
                    "OCR evaluation source: %s",
                    source_name,
                )

        # ----------------------------------------------------
        # Optional fallback to actual text fields in final
        # prediction JSON.
        # ----------------------------------------------------

        if (
            prediction_text is None
            and self.config.allow_prediction_text_fallback
            and prediction_data is not None
        ):

            fallback_fields = (
                "full_text",
                "original_full_text",
                "cleaned_text",
                "cleaned_full_text",
            )

            for field_name in fallback_fields:

                if field_name not in prediction_data:
                    continue

                value = prediction_data.get(
                    field_name
                )

                if value is None:
                    continue

                prediction_text = str(
                    value
                )

                warnings.append(
                    "ocr_prediction_text_fallback_used"
                )

                source_name = (
                    f"prediction:{field_name}"
                )

                break

        if prediction_text is None:

            warnings.append(
                "prediction_ocr_text_unavailable"
            )

            return (
                None,
                None,
                None,
                None,
                None,
                None,
                warnings,
            )

        normalized_truth = self._normalize_text(
            truth_text
        )

        normalized_prediction = self._normalize_text(
            prediction_text
        )

        (
            cer,
            cer_distance,
            reference_characters,
        ) = SequenceMetrics.cer_stats(
            normalized_truth,
            normalized_prediction,
        )

        (
            wer,
            wer_distance,
            reference_words,
        ) = SequenceMetrics.wer_stats(
            normalized_truth,
            normalized_prediction,
        )

        if (
            cer
            > self.config.cer_warning_threshold
        ):

            warnings.append(
                "high_character_error_rate"
            )

        if (
            wer
            > self.config.wer_warning_threshold
        ):

            warnings.append(
                "high_word_error_rate"
            )

        return (
            float(cer),
            float(wer),
            int(cer_distance),
            int(reference_characters),
            int(wer_distance),
            int(reference_words),
            warnings,
        )

    # ========================================================
    # EMPTY / MISSING PREDICTION FIELD RESULTS
    # ========================================================

    def _evaluate_required_fields(
        self,
        prediction_data: dict[str, Any] | None,
        ground_truth_data: dict[str, Any] | None,
    ) -> tuple[
        bool | None,
        bool | None,
        bool | None,
        float | None,
        bool | None,
    ]:
        """
        Returns:

            store_correct,
            date_correct,
            total_correct,
            required_field_accuracy,
            required_fields_all_correct
        """

        if ground_truth_data is None:

            return (
                None,
                None,
                None,
                None,
                None,
            )

        # ----------------------------------------------------
        # Prediction values.
        # ----------------------------------------------------

        if prediction_data is None:

            predicted_store = None
            predicted_date = None
            predicted_total = None

        else:

            predicted_store = prediction_data.get(
                "store_name"
            )

            predicted_date = prediction_data.get(
                "date"
            )

            predicted_total = prediction_data.get(
                "total_amount"
            )

        # ----------------------------------------------------
        # GT values.
        # ----------------------------------------------------

        truth_store = ground_truth_data.get(
            "store_name",
            ground_truth_data.get(
                "vendor_name"
            ),
        )

        truth_date = ground_truth_data.get(
            "date",
            ground_truth_data.get(
                "transaction_date"
            ),
        )

        truth_total = ground_truth_data.get(
            "total_amount"
        )

        # ----------------------------------------------------
        # Store.
        # ----------------------------------------------------

        store_correct = (
            self._string_matches(
                predicted_store,
                truth_store,
            )
        )

        # ----------------------------------------------------
        # Date.
        # ----------------------------------------------------

        if truth_date is None:

            date_correct = None

        else:

            truth_date_normalized = (
                self._normalize_date(
                    truth_date
                )
                if self.config.normalize_dates
                else self._normalize_text(
                    truth_date
                )
            )

            predicted_date_normalized = (
                self._normalize_date(
                    predicted_date
                )
                if self.config.normalize_dates
                else self._normalize_text(
                    predicted_date
                )
            )

            if truth_date_normalized is None:

                date_correct = None

            elif predicted_date_normalized is None:

                date_correct = False

            else:

                date_correct = (
                    predicted_date_normalized
                    == truth_date_normalized
                )

        # ----------------------------------------------------
        # Total.
        # ----------------------------------------------------

        total_correct = (
            self._amount_matches(
                predicted_total,
                truth_total,
            )
        )

        # ----------------------------------------------------
        # Required-field accuracy.
        #
        # Only evaluable GT fields contribute.
        # ----------------------------------------------------

        field_values = [
            store_correct,
            date_correct,
            total_correct,
        ]

        evaluable = [
            value
            for value in field_values
            if value is not None
        ]

        if not evaluable:

            required_field_accuracy = None

            required_fields_all_correct = None

        else:

            required_field_accuracy = (
                float(
                    sum(
                        bool(value)
                        for value
                        in evaluable
                    )
                    / len(
                        evaluable
                    )
                )
            )

            # "All correct" only makes sense when all required
            # fields are actually evaluable.
            if len(
                evaluable
            ) == 3:

                required_fields_all_correct = (
                    all(
                        bool(value)
                        for value
                        in evaluable
                    )
                )

            else:

                required_fields_all_correct = None

        return (
            store_correct,
            date_correct,
            total_correct,
            required_field_accuracy,
            required_fields_all_correct,
        )

    # ========================================================
    # EVALUATE ITEMS
    # ========================================================

    def _evaluate_items(
        self,
        prediction_data: dict[str, Any] | None,
        ground_truth_data: dict[str, Any] | None,
    ) -> tuple[
        float | None,
        float | None,
        float | None,
        int | None,
        int | None,
        int | None,
        list[str],
    ]:

        warnings: list[str] = []

        if ground_truth_data is None:

            return (
                None,
                None,
                None,
                None,
                None,
                None,
                warnings,
            )

        # ----------------------------------------------------
        # A missing GT item key means item evaluation cannot
        # be performed.
        #
        # An explicit [] means zero items and IS evaluable.
        # ----------------------------------------------------

        if "items" not in ground_truth_data:

            warnings.append(
                "ground_truth_items_unavailable"
            )

            return (
                None,
                None,
                None,
                None,
                None,
                None,
                warnings,
            )

        truth_items = ground_truth_data.get(
            "items"
        )

        if not isinstance(
            truth_items,
            list,
        ):

            warnings.append(
                "ground_truth_items_malformed"
            )

            return (
                None,
                None,
                None,
                None,
                None,
                None,
                warnings,
            )

        if prediction_data is None:

            predicted_items = []

        else:

            predicted_items = prediction_data.get(
                "items",
                []
            )

        if not isinstance(
            predicted_items,
            list,
        ):

            predicted_items = []

            warnings.append(
                "prediction_items_malformed"
            )

        valid_truth_count = sum(
            isinstance(
                item,
                dict,
            )
            for item in truth_items
        )

        valid_prediction_count = sum(
            isinstance(
                item,
                dict,
            )
            for item in predicted_items
        )

        counts = self._match_items(
            predicted_items,
            truth_items,
        )

        (
            precision,
            recall,
            f1,
        ) = self._item_metrics(
            counts,
            valid_prediction_count,
            valid_truth_count,
        )

        if (
            f1
            < self.config.item_f1_warning_threshold
        ):

            warnings.append(
                "item_extraction_f1_below_threshold"
            )

        return (
            float(precision),
            float(recall),
            float(f1),
            int(counts.tp),
            int(counts.fp),
            int(counts.fn),
            warnings,
        )

    # ========================================================
    # EVALUATE SINGLE RECEIPT
    # ========================================================

    def evaluate_receipt(
        self,
        receipt_id: str,
        prediction_record: tuple[
            Path,
            dict[str, Any]
        ] | None,
        ground_truth_record: tuple[
            Path,
            dict[str, Any]
        ] | None,
        *,
        prediction_status: str = "available",
        ground_truth_status: str = "available",
    ) -> ReceiptEvaluationResult:

        start = time.perf_counter()

        errors: list[str] = []

        warnings: list[str] = []

        prediction_path = (
            prediction_record[0]
            if prediction_record is not None
            else None
        )

        prediction_data = (
            prediction_record[1]
            if prediction_record is not None
            else None
        )

        ground_truth_path = (
            ground_truth_record[0]
            if ground_truth_record is not None
            else None
        )

        ground_truth_data = (
            ground_truth_record[1]
            if ground_truth_record is not None
            else None
        )

        # ----------------------------------------------------
        # Duplicate / missing status
        # ----------------------------------------------------

        if prediction_status == "duplicate":

            errors.append(
                "duplicate_prediction_receipt_id"
            )

        elif prediction_record is None:

            if ground_truth_record is not None:

                errors.append(
                    "prediction_missing"
                )

        if ground_truth_status == "duplicate":

            errors.append(
                "duplicate_ground_truth_receipt_id"
            )

        elif ground_truth_record is None:

            warnings.append(
                "ground_truth_not_available"
            )

        # ----------------------------------------------------
        # Prediction diagnostics.
        # ----------------------------------------------------

        if prediction_data is not None:

            warnings.extend(
                self._prediction_schema_warnings(
                    prediction_data
                )
            )

        # ----------------------------------------------------
        # Confidence.
        # ----------------------------------------------------

        if prediction_data is not None:

            (
                prediction_confidence,
                confidence_warnings,
            ) = self._prediction_confidence(
                prediction_data
            )

            warnings.extend(
                confidence_warnings
            )

            review_required = (
                self._normalize_bool(
                    prediction_data.get(
                        "review_required"
                    ),
                    default=False,
                )
            )

        else:

            prediction_confidence = None

            review_required = (
                True
                if ground_truth_record is not None
                else None
            )

        # ----------------------------------------------------
        # OCR
        # ----------------------------------------------------

        ocr_data = None

        if ground_truth_record is not None:

            # OCR data is retrieved by receipt ID later via index.
            # The caller may patch this after constructing the
            # result; in normal batch operation evaluate_receipt
            # receives OCR data through the temporary attribute.
            ocr_data = getattr(
                self,
                "_current_ocr_data",
                None,
            )

        (
            cer,
            wer,
            cer_distance,
            cer_reference_characters,
            wer_distance,
            wer_reference_words,
            ocr_warnings,
        ) = self._evaluate_ocr(
            prediction_data,
            ground_truth_data,
            ocr_data,
        )

        warnings.extend(
            ocr_warnings
        )

        # ----------------------------------------------------
        # Required fields.
        # ----------------------------------------------------

        (
            store_correct,
            date_correct,
            total_correct,
            required_field_accuracy,
            required_fields_all_correct,
        ) = self._evaluate_required_fields(
            prediction_data,
            ground_truth_data,
        )

        # ----------------------------------------------------
        # Field mismatch warnings.
        # ----------------------------------------------------

        if (
            store_correct is False
        ):

            warnings.append(
                "store_name_mismatch"
            )

        if (
            date_correct is False
        ):

            warnings.append(
                "date_mismatch"
            )

        if (
            total_correct is False
        ):

            warnings.append(
                "total_amount_mismatch"
            )

        # ----------------------------------------------------
        # Items.
        # ----------------------------------------------------

        (
            item_precision,
            item_recall,
            item_f1,
            item_tp,
            item_fp,
            item_fn,
            item_warnings,
        ) = self._evaluate_items(
            prediction_data,
            ground_truth_data,
        )

        warnings.extend(
            item_warnings
        )

        # ----------------------------------------------------
        # No GT case.
        # ----------------------------------------------------

        if ground_truth_record is None:

            cer = None
            wer = None

            cer_distance = None
            cer_reference_characters = None

            wer_distance = None
            wer_reference_words = None

            store_correct = None
            date_correct = None
            total_correct = None

            required_field_accuracy = None

            required_fields_all_correct = None

            item_precision = None
            item_recall = None
            item_f1 = None

            item_tp = None
            item_fp = None
            item_fn = None

        processing_time_ms = round(
            (
                time.perf_counter()
                - start
            )
            * 1000.0,
            3,
        )

        return ReceiptEvaluationResult(
            receipt_id=receipt_id,

            prediction_file=(
                str(prediction_path)
                if prediction_path is not None
                else None
            ),

            ground_truth_file=(
                str(ground_truth_path)
                if ground_truth_path is not None
                else None
            ),

            prediction_available=(
                prediction_record is not None
                and prediction_status != "duplicate"
            ),

            ground_truth_available=(
                ground_truth_record is not None
                and ground_truth_status != "duplicate"
            ),

            prediction_status=(
                prediction_status
            ),

            ground_truth_status=(
                ground_truth_status
            ),

            cer=(
                float(cer)
                if cer is not None
                else None
            ),

            wer=(
                float(wer)
                if wer is not None
                else None
            ),

            cer_edit_distance=(
                cer_distance
            ),

            cer_reference_characters=(
                cer_reference_characters
            ),

            wer_edit_distance=(
                wer_distance
            ),

            wer_reference_words=(
                wer_reference_words
            ),

            store_name_correct=(
                store_correct
            ),

            date_correct=(
                date_correct
            ),

            total_amount_correct=(
                total_correct
            ),

            required_fields_accuracy=(
                float(
                    required_field_accuracy
                )
                if required_field_accuracy is not None
                else None
            ),

            required_fields_all_correct=(
                required_fields_all_correct
            ),

            item_precision=(
                float(item_precision)
                if item_precision is not None
                else None
            ),

            item_recall=(
                float(item_recall)
                if item_recall is not None
                else None
            ),

            item_f1=(
                float(item_f1)
                if item_f1 is not None
                else None
            ),

            item_tp=item_tp,

            item_fp=item_fp,

            item_fn=item_fn,

            prediction_confidence=(
                prediction_confidence
            ),

            review_required=(
                review_required
            ),

            errors=sorted(
                set(
                    errors
                )
            ),

            warnings=sorted(
                set(
                    warnings
                )
            ),

            processing_time_ms=(
                processing_time_ms
            ),
        )

    # ========================================================
    # HELPER: SAFE FLOAT
    # ========================================================

    @staticmethod
    def _safe_float(
        value: Any,
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

            return float(number)

        except (
            TypeError,
            ValueError,
        ):

            return None

    # ========================================================
    # HELPER: MEAN FINITE
    # ========================================================

    @staticmethod
    def _mean(
        values: list[float],
    ) -> float | None:

        if not values:
            return None

        return float(
            np.mean(
                values
            )
        )

    @staticmethod
    def _median(
        values: list[float],
    ) -> float | None:

        if not values:
            return None

        return float(
            np.median(
                values
            )
        )

    # ========================================================
    # CORPUS OCR METRICS
    # ========================================================

    @staticmethod
    def _corpus_metric(
        results: list[
            ReceiptEvaluationResult
        ],
        distance_field: str,
        denominator_field: str,
    ) -> float | None:

        total_distance = 0

        total_denominator = 0

        evaluable = False

        for result in results:

            distance = getattr(
                result,
                distance_field,
            )

            denominator = getattr(
                result,
                denominator_field,
            )

            if (
                distance is None
                or denominator is None
            ):
                continue

            evaluable = True

            total_distance += int(
                distance
            )

            total_denominator += int(
                denominator
            )

        if not evaluable:
            return None

        return float(
            total_distance
            / max(
                1,
                total_denominator,
            )
        )

    # ========================================================
    # BINARY FIELD ACCURACY
    # ========================================================

    @staticmethod
    def _binary_accuracy(
        values: list[bool | None],
    ) -> tuple[
        float | None,
        int,
        int,
    ]:

        evaluable = [
            value
            for value in values
            if value is not None
        ]

        if not evaluable:

            return (
                None,
                0,
                0,
            )

        correct = sum(
            bool(value)
            for value in evaluable
        )

        return (
            float(
                correct
                / len(
                    evaluable
                )
            ),
            int(
                correct
            ),
            int(
                len(
                    evaluable
                )
            ),
        )

    # ========================================================
    # CONFIDENCE CALIBRATION
    # ========================================================

    def _confidence_calibration(
        self,
        results: list[
            ReceiptEvaluationResult
        ],
    ) -> dict[str, Any]:

        calibration_rows: list[
            tuple[float, float]
        ] = []

        for result in results:

            confidence = (
                result.prediction_confidence
            )

            correctness = (
                result.required_fields_all_correct
            )

            if (
                confidence is None
                or correctness is None
            ):

                continue

            calibration_rows.append(
                (
                    float(
                        confidence
                    ),
                    1.0
                    if correctness
                    else 0.0,
                )
            )

        if not calibration_rows:

            return {
                "evaluated_receipts": 0,
                "mean_prediction_confidence": None,
                "required_field_exact_rate": None,
                "expected_calibration_error": None,
                "buckets": [],
            }

        bin_count = (
            self.config.confidence_bin_count
        )

        buckets = []

        total_count = len(
            calibration_rows
        )

        ece = 0.0

        for bucket_index in range(
            bin_count
        ):

            lower = (
                bucket_index
                / bin_count
            )

            upper = (
                (
                    bucket_index + 1
                )
                / bin_count
            )

            if bucket_index == (
                bin_count - 1
            ):

                members = [
                    row
                    for row
                    in calibration_rows
                    if (
                        lower
                        <= row[0]
                        <= upper
                    )
                ]

            else:

                members = [
                    row
                    for row
                    in calibration_rows
                    if (
                        lower
                        <= row[0]
                        < upper
                    )
                ]

            if members:

                mean_confidence = float(
                    np.mean(
                        [
                            row[0]
                            for row
                            in members
                        ]
                    )
                )

                empirical_accuracy = float(
                    np.mean(
                        [
                            row[1]
                            for row
                            in members
                        ]
                    )
                )

                count = len(
                    members
                )

                ece += (
                    count
                    / total_count
                ) * abs(
                    mean_confidence
                    - empirical_accuracy
                )

            else:

                mean_confidence = None

                empirical_accuracy = None

                count = 0

            buckets.append(
                {
                    "lower": round(
                        lower,
                        4,
                    ),

                    "upper": round(
                        upper,
                        4,
                    ),

                    "count": count,

                    "mean_confidence": (
                        mean_confidence
                    ),

                    "empirical_accuracy": (
                        empirical_accuracy
                    ),
                }
            )

        mean_confidence = float(
            np.mean(
                [
                    row[0]
                    for row
                    in calibration_rows
                ]
            )
        )

        exact_rate = float(
            np.mean(
                [
                    row[1]
                    for row
                    in calibration_rows
                ]
            )
        )

        return {
            "evaluated_receipts": (
                total_count
            ),

            "mean_prediction_confidence": (
                mean_confidence
            ),

            "required_field_exact_rate": (
                exact_rate
            ),

            "expected_calibration_error": (
                float(ece)
            ),

            "buckets": buckets,
        }

    # ========================================================
    # QUALITY GATES
    # ========================================================

    def _quality_gates(
        self,
        aggregate: dict[str, Any],
    ) -> dict[str, Any]:

        if not self.config.quality_gate_enabled:

            return {
                "enabled": False,
                "status": "disabled",
                "checks": {},
            }

        field_extraction = aggregate.get(
            "field_extraction",
            {},
        )

        items = aggregate.get(
            "items",
            {},
        )

        ocr = aggregate.get(
            "ocr",
            {},
        )

        checks: dict[
            str,
            dict[str, Any]
        ] = {}

        def add_check(
            name: str,
            actual: float | None,
            threshold: float | None,
            operator: str,
        ) -> None:

            if threshold is None:

                checks[name] = {
                    "status": "not_configured",
                    "actual": actual,
                    "threshold": None,
                }

                return

            if actual is None:

                checks[name] = {
                    "status": "not_evaluable",
                    "actual": None,
                    "threshold": threshold,
                }

                return

            if operator == "min":

                passed = (
                    actual >= threshold
                )

            else:

                passed = (
                    actual <= threshold
                )

            checks[name] = {
                "status": (
                    "passed"
                    if passed
                    else "failed"
                ),
                "actual": actual,
                "threshold": threshold,
            }

        add_check(
            "corpus_cer",
            ocr.get(
                "corpus_cer"
            ),
            self.config.max_corpus_cer,
            "max",
        )

        add_check(
            "corpus_wer",
            ocr.get(
                "corpus_wer"
            ),
            self.config.max_corpus_wer,
            "max",
        )

        add_check(
            "store_accuracy",
            field_extraction.get(
                "store_name_accuracy"
            ),
            self.config.min_store_accuracy,
            "min",
        )

        add_check(
            "date_accuracy",
            field_extraction.get(
                "date_accuracy"
            ),
            self.config.min_date_accuracy,
            "min",
        )

        add_check(
            "total_accuracy",
            field_extraction.get(
                "total_amount_accuracy"
            ),
            self.config.min_total_accuracy,
            "min",
        )

        add_check(
            "item_f1",
            items.get(
                "micro_f1"
            ),
            self.config.min_item_f1,
            "min",
        )

        actual_checks = [
            check
            for check
            in checks.values()
            if check["status"]
            in {
                "passed",
                "failed",
            }
        ]

        if not actual_checks:

            status = "not_evaluable"

        elif any(
            check["status"] == "failed"
            for check
            in actual_checks
        ):

            status = "failed"

        else:

            status = "passed"

        return {
            "enabled": True,
            "status": status,
            "checks": checks,
        }

    # ========================================================
    # AGGREGATE RESULTS
    # ========================================================

    def _aggregate(
        self,
        results: list[
            ReceiptEvaluationResult
        ],
        *,
        prediction_file_count: int,
        ground_truth_file_count: int,
        prediction_ids: set[str],
        ground_truth_ids: set[str],
        duplicate_prediction_ids: set[str],
        duplicate_ground_truth_ids: set[str],
        prediction_load_failures: list[dict[str, Any]],
        ground_truth_load_failures: list[dict[str, Any]],
    ) -> dict[str, Any]:

        unique_ground_truth_ids = (
            set(
                ground_truth_ids
            )
            - set(
                duplicate_ground_truth_ids
            )
        )

        unique_prediction_ids = (
            set(
                prediction_ids
            )
            - set(
                duplicate_prediction_ids
            )
        )

        missing_prediction_ids = (
            unique_ground_truth_ids
            - prediction_ids
        )

        ambiguous_prediction_ids = (
            unique_ground_truth_ids
            & duplicate_prediction_ids
        )

        extra_prediction_ids = (
            unique_prediction_ids
            - ground_truth_ids
        )

        matched_ids = (
            unique_ground_truth_ids
            & unique_prediction_ids
        )

        evaluated = [
            result
            for result
            in results
            if result.ground_truth_available
        ]

        # ----------------------------------------------------
        # OCR
        # ----------------------------------------------------

        cer_values = [
            float(result.cer)
            for result in evaluated
            if result.cer is not None
        ]

        wer_values = [
            float(result.wer)
            for result in evaluated
            if result.wer is not None
        ]

        corpus_cer = self._corpus_metric(
            results,
            "cer_edit_distance",
            "cer_reference_characters",
        )

        corpus_wer = self._corpus_metric(
            results,
            "wer_edit_distance",
            "wer_reference_words",
        )

        # ----------------------------------------------------
        # Required fields
        # ----------------------------------------------------

        (
            store_accuracy,
            store_correct_count,
            store_evaluable_count,
        ) = self._binary_accuracy(
            [
                result.store_name_correct
                for result
                in evaluated
            ]
        )

        (
            date_accuracy,
            date_correct_count,
            date_evaluable_count,
        ) = self._binary_accuracy(
            [
                result.date_correct
                for result
                in evaluated
            ]
        )

        (
            total_accuracy,
            total_correct_count,
            total_evaluable_count,
        ) = self._binary_accuracy(
            [
                result.total_amount_correct
                for result
                in evaluated
            ]
        )

        required_field_accuracy_values = [
            float(
                result.required_fields_accuracy
            )
            for result
            in evaluated
            if result.required_fields_accuracy
            is not None
        ]

        required_fields_all_correct = [
            result.required_fields_all_correct
            for result
            in evaluated
            if result.required_fields_all_correct
            is not None
        ]

        required_field_exact_rate = (
            float(
                np.mean(
                    [
                        1.0
                        if value
                        else 0.0
                        for value
                        in required_fields_all_correct
                    ]
                )
            )
            if required_fields_all_correct
            else None
        )

        # ----------------------------------------------------
        # Items
        # ----------------------------------------------------

        item_results = [
            result
            for result
            in evaluated
            if result.item_f1 is not None
        ]

        total_tp = sum(
            int(
                result.item_tp or 0
            )
            for result
            in item_results
        )

        total_fp = sum(
            int(
                result.item_fp or 0
            )
            for result
            in item_results
        )

        total_fn = sum(
            int(
                result.item_fn or 0
            )
            for result
            in item_results
        )

        micro_precision = (
            FieldMetrics.precision(
                total_tp,
                total_fp,
            )
            if item_results
            else None
        )

        micro_recall = (
            FieldMetrics.recall(
                total_tp,
                total_fn,
            )
            if item_results
            else None
        )

        micro_f1 = (
            FieldMetrics.f1(
                micro_precision,
                micro_recall,
            )
            if (
                micro_precision is not None
                and micro_recall is not None
            )
            else None
        )

        macro_precision_values = [
            float(
                result.item_precision
            )
            for result
            in item_results
            if result.item_precision
            is not None
        ]

        macro_recall_values = [
            float(
                result.item_recall
            )
            for result
            in item_results
            if result.item_recall
            is not None
        ]

        macro_f1_values = [
            float(
                result.item_f1
            )
            for result
            in item_results
            if result.item_f1
            is not None
        ]

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        prediction_results = [
            result
            for result
            in results
            if result.prediction_available
        ]

        confidence_values = [
            float(
                result.prediction_confidence
            )
            for result
            in prediction_results
            if result.prediction_confidence
            is not None
        ]

        review_values = [
            bool(
                result.review_required
            )
            for result
            in prediction_results
            if result.review_required
            is not None
        ]

        calibration = (
            self._confidence_calibration(
                results
            )
        )

        # ----------------------------------------------------
        # Required-field macro accuracy.
        # ----------------------------------------------------

        macro_required_field_accuracy = (
            self._mean(
                required_field_accuracy_values
            )
        )

        # ----------------------------------------------------
        # Coverage
        # ----------------------------------------------------

        coverage = {
            "prediction_files": (
                prediction_file_count
            ),

            "ground_truth_files": (
                ground_truth_file_count
            ),

            "unique_prediction_ids": (
                len(
                    unique_prediction_ids
                )
            ),

            "unique_ground_truth_ids": (
                len(
                    unique_ground_truth_ids
                )
            ),

            "matched_receipts": (
                len(
                    matched_ids
                )
            ),

            "missing_prediction_receipts": (
                len(
                    missing_prediction_ids
                )
            ),

            "ambiguous_prediction_receipts": (
                len(
                    ambiguous_prediction_ids
                )
            ),

            "extra_prediction_receipts": (
                len(
                    extra_prediction_ids
                )
            ),

            "duplicate_prediction_ids": (
                len(
                    duplicate_prediction_ids
                )
            ),

            "duplicate_ground_truth_ids": (
                len(
                    duplicate_ground_truth_ids
                )
            ),

            "prediction_load_failures": (
                len(
                    prediction_load_failures
                )
            ),

            "ground_truth_load_failures": (
                len(
                    ground_truth_load_failures
                )
            ),
        }

        # ----------------------------------------------------
        # Quality gates.
        # ----------------------------------------------------

        aggregate_pre_gate = {
            "ocr": {
                "corpus_cer": corpus_cer,
                "corpus_wer": corpus_wer,
            },

            "field_extraction": {
                "store_name_accuracy": store_accuracy,
                "date_accuracy": date_accuracy,
                "total_amount_accuracy": total_accuracy,
            },

            "items": {
                "micro_f1": micro_f1,
            },
        }

        quality_gates = (
            self._quality_gates(
                aggregate_pre_gate
            )
        )

        # ----------------------------------------------------
        # Final summary.
        # ----------------------------------------------------

        aggregate = {
            "ground_truth_available": bool(
                unique_ground_truth_ids
            ),

            "evaluated_receipts": len(
                evaluated
            ),

            "coverage": coverage,

            "ocr": {

                "corpus_cer": (
                    corpus_cer
                ),

                "mean_cer": (
                    self._mean(
                        cer_values
                    )
                ),

                "median_cer": (
                    self._median(
                        cer_values
                    )
                ),

                "corpus_wer": (
                    corpus_wer
                ),

                "mean_wer": (
                    self._mean(
                        wer_values
                    )
                ),

                "median_wer": (
                    self._median(
                        wer_values
                    )
                ),

                "cer_evaluable_receipts": (
                    len(
                        cer_values
                    )
                ),

                "wer_evaluable_receipts": (
                    len(
                        wer_values
                    )
                ),
            },

            "field_extraction": {

                "store_name_accuracy": (
                    store_accuracy
                ),

                "store_name_correct": (
                    store_correct_count
                ),

                "store_name_evaluable": (
                    store_evaluable_count
                ),

                "date_accuracy": (
                    date_accuracy
                ),

                "date_correct": (
                    date_correct_count
                ),

                "date_evaluable": (
                    date_evaluable_count
                ),

                "total_amount_accuracy": (
                    total_accuracy
                ),

                "total_amount_correct": (
                    total_correct_count
                ),

                "total_amount_evaluable": (
                    total_evaluable_count
                ),

                "required_fields_micro_accuracy": (
                    (
                        (
                            store_correct_count
                            + date_correct_count
                            + total_correct_count
                        )
                        / (
                            store_evaluable_count
                            + date_evaluable_count
                            + total_evaluable_count
                        )
                    )
                    if (
                        store_evaluable_count
                        + date_evaluable_count
                        + total_evaluable_count
                    )
                    else None
                ),

                "required_fields_macro_accuracy": (
                    macro_required_field_accuracy
                ),

                "required_fields_exact_rate": (
                    required_field_exact_rate
                ),

                "required_fields_exact_evaluable": (
                    len(
                        required_fields_all_correct
                    )
                ),
            },

            "items": {

                "micro_precision": (
                    micro_precision
                ),

                "micro_recall": (
                    micro_recall
                ),

                "micro_f1": (
                    micro_f1
                ),

                "macro_precision": (
                    self._mean(
                        macro_precision_values
                    )
                ),

                "macro_recall": (
                    self._mean(
                        macro_recall_values
                    )
                ),

                "macro_f1": (
                    self._mean(
                        macro_f1_values
                    )
                ),

                "true_positive": (
                    total_tp
                ),

                "false_positive": (
                    total_fp
                ),

                "false_negative": (
                    total_fn
                ),

                "evaluable_receipts": (
                    len(
                        item_results
                    )
                ),
            },

            "confidence": {

                "mean_prediction_confidence": (
                    self._mean(
                        confidence_values
                    )
                ),

                "review_rate": (
                    (
                        sum(
                            review_values
                        )
                        / len(
                            review_values
                        )
                    )
                    if review_values
                    else None
                ),

                "confidence_evaluable_receipts": (
                    len(
                        confidence_values
                    )
                ),

                "calibration": (
                    calibration
                ),
            },

            "quality_gates": quality_gates,

            "load_failures": {

                "prediction": (
                    prediction_load_failures
                ),

                "ground_truth": (
                    ground_truth_load_failures
                ),
            },
        }

        return aggregate

    # ========================================================
    # SAVE CSV
    # ========================================================

    def _atomic_write_csv(
        self,
        dataframe: pd.DataFrame,
        path: Path,
    ) -> None:
        """
        Atomic CSV write.
        """

        temporary = Path(
            str(path)
            + ".tmp"
        )

        dataframe.to_csv(
            temporary,
            index=False,
            encoding="utf-8",
        )

        os.replace(
            temporary,
            path,
        )

    def _save_csv(
        self,
        results: list[
            ReceiptEvaluationResult
        ],
    ) -> Path:

        rows = [
            {
                "receipt_id": result.receipt_id,

                "prediction_file": (
                    result.prediction_file
                ),

                "ground_truth_file": (
                    result.ground_truth_file
                ),

                "prediction_available": (
                    result.prediction_available
                ),

                "ground_truth_available": (
                    result.ground_truth_available
                ),

                "prediction_status": (
                    result.prediction_status
                ),

                "ground_truth_status": (
                    result.ground_truth_status
                ),

                "cer": (
                    result.cer
                ),

                "wer": (
                    result.wer
                ),

                "cer_edit_distance": (
                    result.cer_edit_distance
                ),

                "cer_reference_characters": (
                    result.cer_reference_characters
                ),

                "wer_edit_distance": (
                    result.wer_edit_distance
                ),

                "wer_reference_words": (
                    result.wer_reference_words
                ),

                "store_name_correct": (
                    result.store_name_correct
                ),

                "date_correct": (
                    result.date_correct
                ),

                "total_amount_correct": (
                    result.total_amount_correct
                ),

                "required_fields_accuracy": (
                    result.required_fields_accuracy
                ),

                "required_fields_all_correct": (
                    result.required_fields_all_correct
                ),

                "item_precision": (
                    result.item_precision
                ),

                "item_recall": (
                    result.item_recall
                ),

                "item_f1": (
                    result.item_f1
                ),

                "item_tp": (
                    result.item_tp
                ),

                "item_fp": (
                    result.item_fp
                ),

                "item_fn": (
                    result.item_fn
                ),

                "prediction_confidence": (
                    result.prediction_confidence
                ),

                "review_required": (
                    result.review_required
                ),

                "error_count": (
                    len(
                        result.errors
                    )
                ),

                "warning_count": (
                    len(
                        result.warnings
                    )
                ),

                "errors": (
                    "; ".join(
                        result.errors
                    )
                ),

                "warnings": (
                    "; ".join(
                        result.warnings
                    )
                ),

                "processing_time_ms": (
                    result.processing_time_ms
                ),
            }
            for result
            in results
        ]

        dataframe = pd.DataFrame(
            rows,
        )

        path = (
            self.output_dir
            / "evaluation_results.csv"
        )

        self._atomic_write_csv(
            dataframe,
            path,
        )

        return path

    # ========================================================
    # SAVE JSON ATOMICALLY
    # ========================================================

    @staticmethod
    def _atomic_write_json(
        payload: Any,
        path: Path,
    ) -> None:
        """
        Atomic JSON write with NaN protection.
        """

        temporary = Path(
            str(path)
            + ".tmp"
        )

        with temporary.open(
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

            file.flush()

            os.fsync(
                file.fileno()
            )

        os.replace(
            temporary,
            path,
        )

    def _save_json(
        self,
        aggregate: dict[str, Any],
        results: list[
            ReceiptEvaluationResult
        ],
    ) -> Path:

        payload = {
            "schema_version": "1.0",

            "generated_at_utc": (
                datetime.utcnow().isoformat()
                + "Z"
            ),

            "configuration": (
                asdict(
                    self.config
                )
            ),

            "summary": aggregate,

            "receipts": [
                asdict(
                    result
                )
                for result
                in results
            ],
        }

        path = (
            self.output_dir
            / "evaluation_report.json"
        )

        self._atomic_write_json(
            payload,
            path,
        )

        return path

    # ========================================================
    # SAVE DETAILED RESULTS
    # ========================================================

    def _save_detailed_results(
        self,
        results: list[
            ReceiptEvaluationResult
        ],
    ) -> Path:

        path = (
            self.output_dir
            / "evaluation_details.json"
        )

        payload = [
            asdict(
                result
            )
            for result
            in results
        ]

        self._atomic_write_json(
            payload,
            path,
        )

        return path

    # ========================================================
    # MAIN EVALUATION
    # ========================================================

    def evaluate(
        self,
    ) -> dict[str, Any]:

        start = time.perf_counter()

        logger.info(
            "Starting production-grade evaluation..."
        )

        # ----------------------------------------------------
        # Discover files
        # ----------------------------------------------------

        prediction_files = (
            self._discover_predictions()
        )

        ground_truth_files = (
            self._discover_ground_truth()
        )

        ocr_files = (
            self._discover_ocr_files()
        )

        # ----------------------------------------------------
        # Build prediction index.
        # ----------------------------------------------------

        (
            prediction_index,
            duplicate_prediction_ids,
            prediction_load_failures,
        ) = self._build_file_index(
            prediction_files,
            "prediction",
        )

        # ----------------------------------------------------
        # Build GT index.
        # ----------------------------------------------------

        (
            ground_truth_index,
            duplicate_ground_truth_ids,
            ground_truth_load_failures,
        ) = self._build_file_index(
            ground_truth_files,
            "ground_truth",
        )

        # ----------------------------------------------------
        # Build OCR evidence index.
        # ----------------------------------------------------

        (
            ocr_index,
            duplicate_ocr_ids,
            ocr_load_failures,
        ) = self._build_file_index(
            ocr_files,
            "ocr",
        )

        self._prediction_load_failures = (
            prediction_load_failures
        )

        self._ground_truth_load_failures = (
            ground_truth_load_failures
        )

        self._ocr_load_failures = (
            ocr_load_failures
        )

        self._duplicate_prediction_ids = (
            duplicate_prediction_ids
        )

        self._duplicate_ground_truth_ids = (
            duplicate_ground_truth_ids
        )

        self._duplicate_ocr_ids = (
            duplicate_ocr_ids
        )

        # ----------------------------------------------------
        # All logical receipt IDs.
        #
        # Union prevents missing prediction files from silently
        # disappearing from recall/coverage.
        # ----------------------------------------------------

        prediction_ids = set(
            prediction_index.keys()
        ) | set(
            duplicate_prediction_ids
        )

        ground_truth_ids = set(
            ground_truth_index.keys()
        ) | set(
            duplicate_ground_truth_ids
        )

        logical_receipt_ids = (
            prediction_ids
            | ground_truth_ids
        )

        results: list[
            ReceiptEvaluationResult
        ] = []

        # ----------------------------------------------------
        # Evaluate every logical receipt.
        # ----------------------------------------------------

        for index, receipt_id in enumerate(
            sorted(
                logical_receipt_ids,
                key=lambda value: value.casefold(),
            ),
            start=1,
        ):

            # ------------------------------------------------
            # Prediction status.
            # ------------------------------------------------

            if (
                receipt_id
                in duplicate_prediction_ids
            ):

                prediction_record = None

                prediction_status = (
                    "duplicate"
                )

            elif (
                receipt_id
                in prediction_index
            ):

                prediction_record = (
                    prediction_index[
                        receipt_id
                    ]
                )

                prediction_status = (
                    "available"
                )

            else:

                prediction_record = None

                prediction_status = (
                    "missing"
                )

            # ------------------------------------------------
            # Ground-truth status.
            # ------------------------------------------------

            if (
                receipt_id
                in duplicate_ground_truth_ids
            ):

                ground_truth_record = None

                ground_truth_status = (
                    "duplicate"
                )

            elif (
                receipt_id
                in ground_truth_index
            ):

                ground_truth_record = (
                    ground_truth_index[
                        receipt_id
                    ]
                )

                ground_truth_status = (
                    "available"
                )

            else:

                ground_truth_record = None

                ground_truth_status = (
                    "missing"
                )

            # ------------------------------------------------
            # OCR source for this receipt.
            # ------------------------------------------------

            ocr_record = (
                ocr_index.get(
                    receipt_id
                )
            )

            # Do NOT use ambiguous OCR records.
            if (
                receipt_id
                in duplicate_ocr_ids
            ):

                ocr_data = None

            elif ocr_record is not None:

                ocr_data = (
                    ocr_record[1]
                )

            else:

                ocr_data = None

            self._current_ocr_data = (
                ocr_data
            )

            try:

                result = self.evaluate_receipt(
                    receipt_id=receipt_id,
                    prediction_record=(
                        prediction_record
                        if prediction_status
                        == "available"
                        else None
                    ),
                    ground_truth_record=(
                        ground_truth_record
                        if ground_truth_status
                        == "available"
                        else None
                    ),
                    prediction_status=(
                        prediction_status
                    ),
                    ground_truth_status=(
                        ground_truth_status
                    ),
                )

                # ------------------------------------------------
                # Add OCR-index diagnostics.
                # ------------------------------------------------

                if (
                    ground_truth_record is not None
                    and ocr_record is None
                ):

                    result.warnings.append(
                        "ocr_prediction_artifact_not_found"
                    )

                if (
                    receipt_id
                    in duplicate_ocr_ids
                ):

                    result.warnings.append(
                        "duplicate_ocr_receipt_id"
                    )

                result.warnings = sorted(
                    set(
                        result.warnings
                    )
                )

                results.append(
                    result
                )

            except Exception as exc:

                logger.exception(
                    "Evaluation failed for receipt %s",
                    receipt_id,
                )

                results.append(
                    ReceiptEvaluationResult(
                        receipt_id=receipt_id,

                        prediction_file=(
                            str(
                                prediction_record[0]
                            )
                            if prediction_record is not None
                            else None
                        ),

                        ground_truth_file=(
                            str(
                                ground_truth_record[0]
                            )
                            if ground_truth_record is not None
                            else None
                        ),

                        prediction_available=(
                            prediction_record is not None
                        ),

                        ground_truth_available=(
                            ground_truth_record is not None
                        ),

                        prediction_status=(
                            prediction_status
                        ),

                        ground_truth_status=(
                            ground_truth_status
                        ),

                        cer=None,
                        wer=None,

                        cer_edit_distance=None,
                        cer_reference_characters=None,

                        wer_edit_distance=None,
                        wer_reference_words=None,

                        store_name_correct=None,
                        date_correct=None,
                        total_amount_correct=None,

                        required_fields_accuracy=None,

                        required_fields_all_correct=None,

                        item_precision=None,
                        item_recall=None,
                        item_f1=None,

                        item_tp=None,
                        item_fp=None,
                        item_fn=None,

                        prediction_confidence=None,

                        review_required=True,

                        errors=[
                            f"evaluation_exception:{exc}"
                        ],

                        warnings=[
                            "receipt_evaluation_failed"
                        ],

                        processing_time_ms=round(
                            (
                                time.perf_counter()
                                - start
                            )
                            * 1000.0,
                            3,
                        ),
                    )
                )

            finally:

                self._current_ocr_data = None

            if index % 100 == 0:

                logger.info(
                    "Evaluated %d/%d logical receipts.",
                    index,
                    len(
                        logical_receipt_ids
                    ),
                )

        # ----------------------------------------------------
        # Aggregate.
        # ----------------------------------------------------

        aggregate = self._aggregate(
            results,
            prediction_file_count=len(
                prediction_files
            ),
            ground_truth_file_count=len(
                ground_truth_files
            ),
            prediction_ids=prediction_ids,
            ground_truth_ids=ground_truth_ids,
            duplicate_prediction_ids=(
                duplicate_prediction_ids
            ),
            duplicate_ground_truth_ids=(
                duplicate_ground_truth_ids
            ),
            prediction_load_failures=(
                prediction_load_failures
            ),
            ground_truth_load_failures=(
                ground_truth_load_failures
            ),
        )

        # ----------------------------------------------------
        # Artifact saving.
        # ----------------------------------------------------

        artifact_errors: list[
            str
        ] = []

        csv_path: str | None = None

        json_path: str | None = None

        details_path: str | None = None

        if self.config.save_csv:

            try:

                csv_path = str(
                    self._save_csv(
                        results
                    )
                )

            except Exception as exc:

                message = (
                    f"CSV save failed: {exc}"
                )

                artifact_errors.append(
                    message
                )

                logger.exception(
                    message
                )

        if self.config.save_json:

            try:

                json_path = str(
                    self._save_json(
                        aggregate,
                        results,
                    )
                )

            except Exception as exc:

                message = (
                    f"JSON report save failed: {exc}"
                )

                artifact_errors.append(
                    message
                )

                logger.exception(
                    message
                )

        if self.config.save_detailed_results:

            try:

                details_path = str(
                    self._save_detailed_results(
                        results
                    )
                )

            except Exception as exc:

                message = (
                    f"Detailed results save failed: {exc}"
                )

                artifact_errors.append(
                    message
                )

                logger.exception(
                    message
                )

        # ----------------------------------------------------
        # Status.
        # ----------------------------------------------------

        unique_gt_ids = (
            set(
                ground_truth_index.keys()
            )
        )

        has_gt = bool(
            unique_gt_ids
        )

        receipt_evaluation_errors = sum(
            bool(
                result.errors
            )
            for result
            in results
        )

        if (
            not prediction_files
            and not ground_truth_files
        ):

            status = "empty"

        elif not has_gt:

            status = "no_ground_truth"

        elif receipt_evaluation_errors:

            status = "partial"

        elif (
            prediction_load_failures
            or ground_truth_load_failures
            or duplicate_prediction_ids
            or duplicate_ground_truth_ids
            or artifact_errors
        ):

            status = "partial"

        else:

            status = "success"

        # ----------------------------------------------------
        # Final elapsed time.
        # ----------------------------------------------------

        evaluation_time_ms = round(
            (
                time.perf_counter()
                - start
            )
            * 1000.0,
            3,
        )

        final_result = {
            "schema_version": "1.0",

            "status": status,

            "evaluated_files": len(
                results
            ),

            "evaluation_time_ms": (
                evaluation_time_ms
            ),

            "summary": aggregate,

            "artifacts": {

                "json_path": (
                    json_path
                ),

                "csv_path": (
                    csv_path
                ),

                "details_path": (
                    details_path
                ),
            },

            "artifact_errors": (
                artifact_errors
            ),

            "runtime": {

                "scipy_available": (
                    SCIPY_AVAILABLE
                ),

                "prediction_load_failures": (
                    len(
                        prediction_load_failures
                    )
                ),

                "ground_truth_load_failures": (
                    len(
                        ground_truth_load_failures
                    )
                ),

                "ocr_load_failures": (
                    len(
                        ocr_load_failures
                    )
                ),

                "duplicate_prediction_ids": (
                    len(
                        duplicate_prediction_ids
                    )
                ),

                "duplicate_ground_truth_ids": (
                    len(
                        duplicate_ground_truth_ids
                    )
                ),

                "duplicate_ocr_ids": (
                    len(
                        duplicate_ocr_ids
                    )
                ),
            },
        }

        logger.info(
            "Evaluation completed. "
            "Status=%s | Logical receipts=%d | "
            "GT=%d | Predictions=%d",
            status,
            len(
                logical_receipt_ids
            ),
            len(
                ground_truth_ids
            ),
            len(
                prediction_ids
            ),
        )

        return final_result

    # ========================================================
    # PUBLIC ENTRY POINT
    # ========================================================

    def run(
        self,
    ) -> dict[str, Any]:

        return self.evaluate()