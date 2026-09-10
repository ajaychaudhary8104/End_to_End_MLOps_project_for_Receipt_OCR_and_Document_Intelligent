from __future__ import annotations
import json
import logging
import re
import unicodedata
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
from src.receipt_intelligence.entity.config_entity import TextCleaningConfig
from src.receipt_intelligence import logger


# ============================================================
# CLEANED OCR REGION
# ============================================================

@dataclass
class CleanedOCRRegion:
    """
    Cleaned representation of one OCR text region.
    """

    original_text: str

    cleaned_text: str

    confidence: float

    bbox: list[list[float]]

    x_min: float

    y_min: float

    x_max: float

    y_max: float

    width: float

    height: float

    changes: list[str]


# ============================================================
# CLEANED OCR RESULT
# ============================================================

@dataclass
class CleanedOCRResult:
    """
    Cleaned OCR result for one receipt.
    """

    receipt_id: str

    input_path: str

    status: str

    original_full_text: str

    cleaned_full_text: str

    regions: list[CleanedOCRRegion]

    original_region_count: int

    cleaned_region_count: int

    mean_confidence: float

    processing_time_ms: float

    errors: list[str]

    warnings: list[str]

    timestamp_utc: str


# ============================================================
# TEXT CLEANING CLASS
# ============================================================

class OCRTextCleaner:
    """
    Production-grade OCR post-processing layer.

    Flow:

        OCR JSON
           ↓
        Unicode normalization
           ↓
        Control-character cleanup
           ↓
        Whitespace normalization
           ↓
        OCR artifact cleanup
           ↓
        Currency normalization
           ↓
        Date formatting normalization
           ↓
        Duplicate-line cleanup
           ↓
        Clean text
           ↓
        Extraction layer

    Important:
        This class does NOT try to extract vendor/date/total/items.
        That is handled by the next Field Extraction component.
    """

    # --------------------------------------------------------
    # Common OCR substitutions.
    #
    # These are deliberately limited to contexts where the
    # character is very likely to be an OCR artifact.
    # --------------------------------------------------------

    _common_word_corrections = {
        "T0TAL": "TOTAL",
        "TOTAI": "TOTAL",
        "T0TAl": "TOTAL",
        "SUBT0TAL": "SUBTOTAL",
        "AM0UNT": "AMOUNT",
        "INVCICE": "INVOICE",
        "RECElPT": "RECEIPT",
        "R3CEIPT": "RECEIPT",
    }

    # Common currency symbols / prefixes.
    _currency_pattern = re.compile(
        r"(?i)"
        r"(?:₹|rs\.?|inr|r\s*s\.?)"
        r"\s*"
        r"(?=\d)"
    )

    # Multiple spaces.
    _multiple_spaces = re.compile(
        r"[ \t]+"
    )

    # Excessive punctuation noise.
    _repeated_punctuation = re.compile(
        r"([.,:;!?])\1{2,}"
    )

    # Lines made almost entirely of separator characters.
    _separator_line = re.compile(
        r"^[\s\-_=*~.]{4,}$"
    )

    # Date patterns.
    _date_dd_mm_yyyy = re.compile(
        r"\b"
        r"(\d{1,2})"
        r"([./\-])"
        r"(\d{1,2})"
        r"([./\-])"
        r"(\d{2,4})"
        r"\b"
    )

    _date_yyyy_mm_dd = re.compile(
        r"\b"
        r"(\d{4})"
        r"([./\-])"
        r"(\d{1,2})"
        r"([./\-])"
        r"(\d{1,2})"
        r"\b"
    )

    def __init__(self, config: TextCleaningConfig):
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

        if config.min_text_length < 0:
            raise ValueError(
                "min_text_length cannot be negative."
            )

    # ========================================================
    # NUMERIC SAFETY
    # ========================================================

    @staticmethod
    def _safe_float(
        value: Any
    ) -> float:
        """
        Safely convert a value to finite float.
        """

        try:

            value = float(value)

            if not np.isfinite(value):
                return 0.0

            return value

        except (
            TypeError,
            ValueError
        ):

            return 0.0

    # ========================================================
    # UNICODE NORMALIZATION
    # ========================================================

    def _normalize_unicode(
        self,
        text: str,
        changes: list[str]
    ) -> str:
        """
        Normalize visually equivalent Unicode characters.
        """

        if not self.config.normalize_unicode:
            return text

        normalized = unicodedata.normalize(
            "NFKC",
            text
        )

        if normalized != text:
            changes.append(
                "unicode_normalization"
            )

        return normalized

    # ========================================================
    # CONTROL CHARACTER CLEANUP
    # ========================================================

    def _remove_control_characters(
        self,
        text: str,
        changes: list[str]
    ) -> str:
        """
        Remove invisible control characters while preserving
        normal whitespace/newline semantics.
        """

        if not self.config.remove_control_characters:
            return text

        cleaned_chars = []

        changed = False

        for char in text:

            if char in ("\n", "\t", "\r"):

                cleaned_chars.append(char)
                continue

            category = unicodedata.category(
                char
            )

            if category.startswith("C"):

                changed = True
                continue

            cleaned_chars.append(char)

        cleaned = "".join(
            cleaned_chars
        )

        if changed:
            changes.append(
                "control_character_removal"
            )

        return cleaned

    # ========================================================
    # WHITESPACE NORMALIZATION
    # ========================================================

    def _normalize_whitespace(
        self,
        text: str,
        changes: list[str]
    ) -> str:
        """
        Normalize spaces while retaining line structure.
        """

        if not self.config.normalize_whitespace:
            return text

        original = text

        lines = text.splitlines()

        cleaned_lines = []

        for line in lines:

            line = line.strip()

            line = self._multiple_spaces.sub(
                " ",
                line
            )

            if line:
                cleaned_lines.append(
                    line
                )

        cleaned = "\n".join(
            cleaned_lines
        )

        if cleaned != original:
            changes.append(
                "whitespace_normalization"
            )

        return cleaned

    # ========================================================
    # COMMON OCR ERROR CORRECTION
    # ========================================================

    def _correct_common_ocr_errors(
        self,
        text: str,
        changes: list[str]
    ) -> str:
        """
        Correct high-confidence, receipt-specific OCR errors.

        This deliberately does not run unrestricted fuzzy
        spelling correction, because vendor/item names can be
        legitimate unusual words.
        """

        if not self.config.normalize_common_ocr_errors:
            return text

        original = text

        lines = []

        for line in text.splitlines():

            stripped = line.strip()

            # Exact high-confidence replacements.
            replacement = self._common_word_corrections.get(
                stripped
            )

            if replacement is not None:

                line = line.replace(
                    stripped,
                    replacement
                )

            # TOTAL-like OCR errors when followed by an amount.
            line = re.sub(
                r"(?i)\bT[O0]T[A4]L\b(?=\s*[:\-]?\s*[$₹]?\s*\d)",
                "TOTAL",
                line
            )

            # GRAND TOTAL variants.
            line = re.sub(
                r"(?i)\bGRAND\s+T[O0]T[A4]L\b",
                "GRAND TOTAL",
                line
            )

            # AMOUNT PAYABLE variants.
            line = re.sub(
                r"(?i)\bAM[O0]UNT\s+PAYABLE\b",
                "AMOUNT PAYABLE",
                line
            )

            lines.append(line)

        cleaned = "\n".join(lines)

        if cleaned != original:
            changes.append(
                "common_ocr_correction"
            )

        return cleaned

    # ========================================================
    # PUNCTUATION NORMALIZATION
    # ========================================================

    def _normalize_punctuation(
        self,
        text: str,
        changes: list[str]
    ) -> str:
        """
        Remove obvious repeated punctuation noise.
        """

        original = text

        cleaned = self._repeated_punctuation.sub(
            r"\1",
            text
        )

        if cleaned != original:
            changes.append(
                "punctuation_normalization"
            )

        return cleaned

    # ========================================================
    # CURRENCY NORMALIZATION
    # ========================================================

    def _normalize_currency(
        self,
        text: str,
        changes: list[str]
    ) -> str:
        """
        Normalize common INR currency prefixes.

        Example:

            ₹100
            Rs. 100
            INR 100
            Rs 100

        become:

            100
        """

        if not self.config.normalize_currency:
            return text

        original = text

        cleaned = self._currency_pattern.sub(
            "",
            text
        )

        # Normalize common currency OCR variants.
        cleaned = re.sub(
            r"(?i)\bI\s*N\s*R\b",
            "",
            cleaned
        )

        cleaned = re.sub(
            r"(?i)\bR\s*S\b(?=\s*\d)",
            "",
            cleaned
        )

        if cleaned != original:
            changes.append(
                "currency_normalization"
            )

        return cleaned

    # ========================================================
    # DATE NORMALIZATION
    # ========================================================

    def _normalize_dates(
        self,
        text: str,
        changes: list[str]
    ) -> str:
        """
        Normalize common numeric receipt dates.

        Output convention:

            YYYY-MM-DD

        Only transforms syntactically valid-looking numeric
        dates. Ambiguous day/month semantics are preserved by
        choosing the conventional DD/MM interpretation.
        """

        if not self.config.normalize_dates:
            return text

        original = text

        def dd_mm_replacer(
            match: re.Match
        ) -> str:

            day = int(
                match.group(1)
            )

            month = int(
                match.group(3)
            )

            year = int(
                match.group(5)
            )

            if year < 100:
                year += 2000

            if not (
                1 <= day <= 31
                and 1 <= month <= 12
            ):
                return match.group(0)

            return (
                f"{year:04d}-"
                f"{month:02d}-"
                f"{day:02d}"
            )

        text = self._date_dd_mm_yyyy.sub(
            dd_mm_replacer,
            text
        )

        def yyyy_mm_dd_replacer(
            match: re.Match
        ) -> str:

            year = int(
                match.group(1)
            )

            month = int(
                match.group(3)
            )

            day = int(
                match.group(5)
            )

            if not (
                1 <= day <= 31
                and 1 <= month <= 12
            ):
                return match.group(0)

            return (
                f"{year:04d}-"
                f"{month:02d}-"
                f"{day:02d}"
            )

        text = self._date_yyyy_mm_dd.sub(
            yyyy_mm_dd_replacer,
            text
        )

        if text != original:
            changes.append(
                "date_normalization"
            )

        return text

    # ========================================================
    # SEPARATOR / NOISE LINE FILTER
    # ========================================================

    def _remove_noise_lines(
        self,
        text: str,
        changes: list[str]
    ) -> str:
        """
        Remove lines that contain no meaningful text.
        """

        if not self.config.remove_empty_lines:
            return text

        original_lines = text.splitlines()

        cleaned_lines = []

        changed = False

        for line in original_lines:

            stripped = line.strip()

            if not stripped:

                changed = True
                continue

            if self._separator_line.match(
                stripped
            ):

                changed = True
                continue

            cleaned_lines.append(
                stripped
            )

        cleaned = "\n".join(
            cleaned_lines
        )

        if changed:
            changes.append(
                "noise_line_removal"
            )

        return cleaned

    # ========================================================
    # ADJACENT DUPLICATE LINES
    # ========================================================

    def _deduplicate_adjacent_lines(
        self,
        text: str,
        changes: list[str]
    ) -> str:
        """
        Remove immediately repeated OCR lines.

        Example:

            Milk 40
            Milk 40

        becomes:

            Milk 40

        Only adjacent exact duplicates are removed.
        """

        if not self.config.deduplicate_adjacent_lines:
            return text

        lines = text.splitlines()

        if not lines:
            return text

        deduplicated = [
            lines[0]
        ]

        changed = False

        for line in lines[1:]:

            if (
                line.strip()
                == deduplicated[-1].strip()
            ):

                changed = True
                continue

            deduplicated.append(
                line
            )

        cleaned = "\n".join(
            deduplicated
        )

        if changed:
            changes.append(
                "adjacent_duplicate_removal"
            )

        return cleaned

    # ========================================================
    # SINGLE TEXT CLEANING
    # ========================================================

    def clean_text(
        self,
        text: str
    ) -> tuple[str, list[str]]:
        """
        Clean one OCR text string.
        """

        if text is None:
            return "", []

        text = str(text)

        changes: list[str] = []

        text = self._normalize_unicode(
            text,
            changes
        )

        text = self._remove_control_characters(
            text,
            changes
        )

        text = self._normalize_whitespace(
            text,
            changes
        )

        text = self._correct_common_ocr_errors(
            text,
            changes
        )

        text = self._normalize_punctuation(
            text,
            changes
        )

        text = self._normalize_currency(
            text,
            changes
        )

        text = self._normalize_dates(
            text,
            changes
        )

        text = self._remove_noise_lines(
            text,
            changes
        )

        text = self._deduplicate_adjacent_lines(
            text,
            changes
        )

        return text.strip(), changes

    # ========================================================
    # RECEIPT ID
    # ========================================================

    @staticmethod
    def _receipt_id_from_file(
        path: Path
    ) -> str:
        return path.stem

    # ========================================================
    # DISCOVER OCR JSON FILES
    # ========================================================

    def _discover_files(self) -> list[Path]:
        """
        Discover OCR JSON result files.
        """

        if not self.input_dir.exists():
            raise FileNotFoundError(
                f"OCR input directory does not exist: "
                f"{self.input_dir}"
            )

        if not self.input_dir.is_dir():
            raise NotADirectoryError(
                f"Expected directory: "
                f"{self.input_dir}"
            )

        files = sorted(
            self.input_dir.rglob(
                "*.json"
            )
        )

        # Never treat the batch manifest as a receipt.
        files = [
            path
            for path in files
            if path.name != "ocr_manifest.json"
        ]

        logger.info(
            "Discovered %d OCR result files.",
            len(files)
        )

        return files

    # ========================================================
    # LOAD JSON
    # ========================================================

    @staticmethod
    def _load_json(
        path: Path
    ) -> dict[str, Any]:
        """
        Load a single OCR result JSON file.
        """

        with path.open(
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        if not isinstance(
            data,
            dict
        ):

            raise ValueError(
                f"Expected JSON object in {path}"
            )

        return data

    # ========================================================
    # REGION CLEANING
    # ========================================================

    def _clean_region(
        self,
        region: dict[str, Any]
    ) -> CleanedOCRRegion | None:
        """
        Clean one OCR region while preserving geometry and
        confidence metadata.
        """

        original_text = str(
            region.get(
                "text",
                ""
            )
        )

        cleaned_text, changes = (
            self.clean_text(
                original_text
            )
        )

        # Remove useless tiny text results.
        if len(cleaned_text) < self.config.min_text_length:
            return None

        confidence = round(
            self._safe_float(
                region.get(
                    "confidence",
                    0.0
                )
            ),
            self.config.confidence_round_digits
        )

        raw_bbox = region.get(
            "bbox",
            []
        )

        if not isinstance(
            raw_bbox,
            list
        ):
            raw_bbox = []

        # Convert geometry safely.
        bbox: list[list[float]] = []

        for point in raw_bbox:

            if not isinstance(
                point,
                (list, tuple)
            ):
                continue

            if len(point) < 2:
                continue

            try:

                x = float(
                    point[0]
                )

                y = float(
                    point[1]
                )

                if np.isfinite(x) and np.isfinite(y):

                    bbox.append(
                        [
                            x,
                            y
                        ]
                    )

            except (
                TypeError,
                ValueError
            ):
                continue

        def safe_dimension(
            key: str
        ) -> float:

            return self._safe_float(
                region.get(
                    key,
                    0.0
                )
            )

        x_min = safe_dimension(
            "x_min"
        )

        y_min = safe_dimension(
            "y_min"
        )

        x_max = safe_dimension(
            "x_max"
        )

        y_max = safe_dimension(
            "y_max"
        )

        width = safe_dimension(
            "width"
        )

        height = safe_dimension(
            "height"
        )

        # Recalculate from bbox if available.
        if bbox:

            xs = [
                point[0]
                for point in bbox
            ]

            ys = [
                point[1]
                for point in bbox
            ]

            x_min = min(xs)
            y_min = min(ys)
            x_max = max(xs)
            y_max = max(ys)

            width = max(
                0.0,
                x_max - x_min
            )

            height = max(
                0.0,
                y_max - y_min
            )

        return CleanedOCRRegion(
            original_text=original_text,
            cleaned_text=cleaned_text,
            confidence=confidence,
            bbox=bbox,
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
            width=width,
            height=height,
            changes=changes,
        )

    # ========================================================
    # PROCESS SINGLE OCR FILE
    # ========================================================

    def process_file(
        self,
        json_path: str | Path
    ) -> CleanedOCRResult:
        """
        Clean one OCR JSON result.
        """

        start = datetime.now(
            timezone.utc
        )

        json_path = Path(
            json_path
        )

        errors: list[str] = []
        warnings: list[str] = []

        receipt_id = (
            self._receipt_id_from_file(
                json_path
            )
        )

        try:

            data = self._load_json(
                json_path
            )

            receipt_id = str(
                data.get(
                    "receipt_id",
                    receipt_id
                )
            )

            original_full_text = str(
                data.get(
                    "full_text",
                    ""
                )
            )

            # ------------------------------------------------
            # Clean full text separately.
            # ------------------------------------------------

            cleaned_full_text, full_text_changes = (
                self.clean_text(
                    original_full_text
                )
            )

            # ------------------------------------------------
            # Clean regions.
            # ------------------------------------------------

            raw_regions = data.get(
                "regions",
                []
            )

            if not isinstance(
                raw_regions,
                list
            ):
                raw_regions = []

                warnings.append(
                    "OCR regions was not a list."
                )

            cleaned_regions = []

            for raw_region in raw_regions:

                if not isinstance(
                    raw_region,
                    dict
                ):
                    warnings.append(
                        "Skipped malformed OCR region."
                    )
                    continue

                cleaned_region = (
                    self._clean_region(
                        raw_region
                    )
                )

                if cleaned_region is not None:

                    cleaned_regions.append(
                        cleaned_region
                    )

            # ------------------------------------------------
            # Rebuild full text from cleaned regions where
            # possible. This preserves reading order.
            # ------------------------------------------------

            region_text = "\n".join(
                region.cleaned_text
                for region in cleaned_regions
                if region.cleaned_text
            ).strip()

            if region_text:

                cleaned_full_text = region_text

            elif (
                not cleaned_full_text
                and original_full_text
            ):

                warnings.append(
                    "Cleaned OCR output contains no usable text."
                )

            # ------------------------------------------------
            # Confidence
            # ------------------------------------------------

            confidences = [
                region.confidence
                for region in cleaned_regions
            ]

            mean_confidence = round(
                float(
                    np.mean(confidences)
                )
                if confidences
                else 0.0,
                self.config.confidence_round_digits
            )

            # ------------------------------------------------
            # Status
            # ------------------------------------------------

            if cleaned_regions:

                status = "success"

            elif cleaned_full_text:

                status = "partial"

                warnings.append(
                    "Full text exists but no structured OCR "
                    "regions survived cleaning."
                )

            else:

                status = "failed"

                warnings.append(
                    "No usable OCR text remains after cleaning."
                )

        except Exception as exc:

            logger.exception(
                "Text cleaning failed for %s",
                json_path
            )

            errors.append(
                str(exc)
            )

            original_full_text = ""
            cleaned_full_text = ""
            cleaned_regions = []
            mean_confidence = 0.0
            status = "failed"
            full_text_changes = []

        end = datetime.now(
            timezone.utc
        )

        processing_time_ms = (
            end - start
        ).total_seconds() * 1000.0

        result = CleanedOCRResult(
            receipt_id=receipt_id,
            input_path=str(
                json_path
            ),
            status=status,
            original_full_text=original_full_text,
            cleaned_full_text=cleaned_full_text,
            regions=cleaned_regions,
            original_region_count=len(
                data.get("regions", [])
            )
            if "data" in locals()
            and isinstance(
                data.get("regions"),
                list
            )
            else 0,
            cleaned_region_count=len(
                cleaned_regions
            ),
            mean_confidence=mean_confidence,
            processing_time_ms=round(
                processing_time_ms,
                3
            ),
            errors=errors,
            warnings=warnings,
            timestamp_utc=end.isoformat(),
        )

        # Add information at the artifact level through
        # warnings rather than changing the public schema.
        if full_text_changes:
            result.warnings.append(
                "Full-text cleaning applied: "
                + ", ".join(
                    sorted(
                        set(full_text_changes)
                    )
                )
            )

        return result

    # ========================================================
    # SAVE RESULT
    # ========================================================

    def _save_result(
        self,
        result: CleanedOCRResult
    ) -> Path:
        """
        Save cleaned OCR result.
        """

        output_path = (
            self.output_dir
            / f"{result.receipt_id}.json"
        )

        with output_path.open(
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                asdict(result),
                file,
                indent=2,
                ensure_ascii=False
            )

        return output_path

    # ========================================================
    # BATCH PROCESSING
    # ========================================================

    def process_batch(
        self
    ) -> dict[str, Any]:
        """
        Process all OCR JSON files.
        """

        logger.info(
            "Starting OCR text cleaning..."
        )

        files = self._discover_files()

        results: list[CleanedOCRResult] = []

        for index, json_path in enumerate(
            files,
            start=1
        ):

            result = self.process_file(
                json_path
            )

            results.append(
                result
            )

            try:

                output_path = self._save_result(
                    result
                )

                logger.info(
                    "Cleaned OCR saved: %s",
                    output_path
                )

            except Exception as exc:

                logger.exception(
                    "Could not save cleaned OCR for %s",
                    json_path
                )

                result.status = "failed"

                result.errors.append(
                    f"Artifact save failed: {exc}"
                )

            if index % 100 == 0:

                logger.info(
                    "Cleaned %d/%d OCR files.",
                    index,
                    len(files)
                )

        successful = sum(
            result.status == "success"
            for result in results
        )

        partial = sum(
            result.status == "partial"
            for result in results
        )

        failed = sum(
            result.status == "failed"
            for result in results
        )

        region_counts = [
            result.cleaned_region_count
            for result in results
        ]

        confidences = [
            result.mean_confidence
            for result in results
            if result.cleaned_region_count > 0
        ]

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

            "total_files": len(results),

            "successful": successful,

            "partial": partial,

            "failed": failed,

            "success_rate": (
                successful / len(results)
                if results
                else 0.0
            ),

            "partial_rate": (
                partial / len(results)
                if results
                else 0.0
            ),

            "mean_cleaned_regions": (
                float(
                    np.mean(region_counts)
                )
                if region_counts
                else 0.0
            ),

            "mean_ocr_confidence": (
                float(
                    np.mean(confidences)
                )
                if confidences
                else 0.0
            ),

            "average_processing_time_ms": (
                float(
                    np.mean(processing_times)
                )
                if processing_times
                else 0.0
            ),

            "results": [
                asdict(result)
                for result in results
            ],
        }

        manifest_path = (
            self.output_dir
            / "text_cleaning_manifest.json"
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

        summary["manifest_path"] = str(
            manifest_path
        )

        logger.info(
            "OCR text cleaning completed."
        )

        return summary

    # ========================================================
    # PUBLIC ENTRY POINT
    # ========================================================

    def run(self, json_path: str | Path | None = None) -> CleanedOCRResult | dict[str, Any]:
        """
        Process a single OCR JSON file or the complete batch.
        """

        if json_path is not None:

            result = self.process_file(
                json_path
            )

            self._save_result(
                result
            )

            return result

        return self.process_batch()