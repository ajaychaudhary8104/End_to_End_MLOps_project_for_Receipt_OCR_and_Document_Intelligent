from __future__ import annotations
import json
import logging
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Any
import numpy as np
from src.receipt_intelligence.entity.config_entity import FieldExtractionConfig
from src.receipt_intelligence import logger


# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class ExtractionCandidate:
    value: Any
    source_text: str
    confidence: float
    extraction_method: str
    region_index: int | None
    evidence: list[str]

@dataclass
class ExtractedItem:
    name: str
    price: float
    source_text: str
    confidence: float
    extraction_method: str
    region_index: int | None
    evidence: list[str]

@dataclass
class ReceiptExtractionResult:
    receipt_id: str
    input_path: str
    status: str

    vendor_name: str | None
    transaction_date: str | None

    items: list[ExtractedItem]
    total_amount: float | None

    vendor_candidate: ExtractionCandidate | None
    date_candidate: ExtractionCandidate | None
    total_candidate: ExtractionCandidate | None

    item_count: int
    expected_item_count: int | None

    raw_text: str
    cleaned_text: str

    errors: list[str]
    warnings: list[str]

    processing_time_ms: float
    timestamp_utc: str

# ============================================================
# FIELD EXTRACTOR
# ============================================================

class FieldExtractor:
    """
    Production-oriented hybrid receipt OCR field extractor.

    Extracts:
        - vendor
        - transaction date
        - total
        - valid named line items

    Design goals:
        - Python-compatible regexes
        - no fragile inline regex flags
        - resistant to OCR noise
        - avoids treating times as amounts
        - prioritizes final TOTAL over SUBTOTAL
        - never fabricates item names
        - preserves evidence
        - batch-safe
    """

    # ========================================================
    # DATE PATTERNS
    # ========================================================

    DATE_PATTERNS = (
        re.compile(
            r"\b(\d{1,2})[\/\-.](\d{1,2})[\/\-.](\d{2,4})\b"
        ),
        re.compile(
            r"\b(\d{4})[\/\-.](\d{1,2})[\/\-.](\d{1,2})\b"
        ),
    )

    # ========================================================
    # TIME PATTERN
    # ========================================================

    TIME_PATTERN = re.compile(
        r"\b\d{1,2}:\d{2}(?::\d{2})?\b"
    )

    # ========================================================
    # AMOUNT PATTERN
    #
    # Important:
    # This intentionally does NOT match colon-separated times.
    # ========================================================

    AMOUNT_PATTERN = re.compile(
        r"""
        (?<![\w:])
        (?:
            ₹\s*
            |
            Rs\.?\s*
            |
            INR\s*
        )?
        (
            \d{1,3}(?:,\d{3})+(?:\.\d{1,2})?
            |
            \d+(?:\.\d{1,2})?
        )
        (?![\w:])
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    # ========================================================
    # ITEM PATTERNS
    # ========================================================

    ITEM_PRICE_AT_END = re.compile(
        r"""
        ^\s*
        (?P<name>.*?)
        \s+
        (?:
            ₹\s*
            |
            Rs\.?\s*
            |
            INR\s*
        )?
        (?P<price>
            \d{1,3}(?:,\d{3})+(?:\.\d{1,2})?
            |
            \d+(?:\.\d{1,2})?
        )
        \s*$
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    ITEM_PRICE_AT_START = re.compile(
        r"""
        ^\s*
        (?:
            ₹\s*
            |
            Rs\.?\s*
            |
            INR\s*
        )?
        (?P<price>
            \d{1,3}(?:,\d{3})+(?:\.\d{1,2})?
            |
            \d+(?:\.\d{1,2})?
        )
        \s+
        (?P<name>.+?)
        \s*$
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    ITEM_QTY_PRICE = re.compile(
        r"""
        ^\s*
        (?P<name>.+?)
        \s+
        (?P<qty>\d+(?:\.\d+)?)
        \s*
        [xX*×]
        \s*
        (?:
            ₹\s*
            |
            Rs\.?\s*
            |
            INR\s*
        )?
        (?P<price>
            \d{1,3}(?:,\d{3})+(?:\.\d{1,2})?
            |
            \d+(?:\.\d{1,2})?
        )
        \s*$
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    # ========================================================
    # LABEL PATTERNS
    #
    # IMPORTANT:
    # No (?ix), (?i), (?x) inside the regex.
    # ========================================================

    DATE_LABEL_PATTERN = re.compile(
        r"""
        \b
        (?:
            DATE
            |
            TXN\s+DATE
            |
            TRANSACTION\s+DATE
            |
            BILL\s+DATE
            |
            PURCHASE\s+DATE
        )
        \s*[:\-]?\s*
        (?P<value>.+)$
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    VENDOR_LABEL_PATTERN = re.compile(
        r"""
        \b
        (?:
            STORE
            |
            SHOP
            |
            VENDOR
            |
            MERCHANT
            |
            SELLER
            |
            STORE\s+NAME
            |
            MERCHANT\s+NAME
        )
        \s*[:\-]\s*
        (?P<value>.+)$
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    # ========================================================
    # ITEMS SOLD
    #
    # FIXED VERSION OF YOUR FAILING REGEX
    # ========================================================

    ITEMS_SOLD_PATTERN = re.compile(
    r"\b(?:\#\s*)?ITEMS\s+SOLD\s*[:#]?\s*(?P<count>\d+)\b",
    re.IGNORECASE,
)

    # ========================================================
    # RECEIPT STRUCTURAL PATTERNS
    # ========================================================

    PERCENT_PATTERN = re.compile(
        r"\b\d+(?:\.\d+)?\s*%"
    )

    ACCOUNT_PATTERN = re.compile(
        r"\bACCOUNT\s*#?",
        re.IGNORECASE,
    )

    REF_PATTERN = re.compile(
        r"\bREF\s*#?",
        re.IGNORECASE,
    )

    TERMINAL_PATTERN = re.compile(
        r"\bTERMINAL\b",
        re.IGNORECASE,
    )

    APPROVAL_PATTERN = re.compile(
        r"\b(?:APPR|APPROVAL)\b",
        re.IGNORECASE,
    )

    # ========================================================
    # REJECTION TOKENS
    # ========================================================

    ITEM_NAME_REJECT_TOKENS = (
        "TOTAL",
        "SUBTOTAL",
        "TAX",
        "VAT",
        "GST",
        "CGST",
        "SGST",
        "IGST",
        "DISCOUNT",
        "CHANGE",
        "PAYMENT",
        "CARD",
        "DEBIT",
        "CREDIT",
        "REF",
        "ACCOUNT",
        "NETWORK",
        "TERMINAL",
        "APPROVAL",
        "APPR",
        "CODE",
        "STORE HOURS",
        "ITEMS SOLD",
        "SAVE MONEY",
        "THANK YOU",
        "NEW STORE HOURS",
        "CASHIER",
        "MANAGER",
        "EFT",
        "TEND",
        "BALANCE",
        "PAY FROM",
        "PURCHASE",
    )

    FOOTER_TOKENS = (
        "SAVE MONEY",
        "THANK YOU",
        "NEW STORE HOURS",
        "SCAN WITH WALMART APP",
        "SAVINGS CATCHER",
    )

    PAYMENT_TOKENS = (
        "CASH",
        "CARD",
        "CREDIT",
        "DEBIT",
        "PAYMENT",
        "CHANGE",
        "BALANCE",
        "EFT",
        "TEND",
    )

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(self, config: FieldExtractionConfig | None = None):
        self.config = config or FieldExtractionConfig()

        self.input_dir = Path(self.config.input_dir)
        self.output_dir = Path(self.config.output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.total_keywords = tuple(
            keyword.upper()
            for keyword in self.config.total_keywords
        )

        self.reject_keywords = tuple(
            keyword.upper()
            for keyword in self.config.reject_keywords
        )

    # ========================================================
    # FILE HANDLING
    # ========================================================

    def _discover_files(self) -> list[Path]:
        if not self.input_dir.exists():
            raise FileNotFoundError(
                f"Text-cleaning directory does not exist: "
                f"{self.input_dir}"
            )

        if not self.input_dir.is_dir():
            raise NotADirectoryError(
                f"Expected directory: {self.input_dir}"
            )

        files = sorted(
            path
            for path in self.input_dir.rglob("*.json")
            if path.name != "text_cleaning_manifest.json"
        )

        logger.info(
            "Discovered %d cleaned OCR files.",
            len(files),
        )

        return files

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError(
                f"Expected JSON object: {path}"
            )

        return data

    # ========================================================
    # NUMERIC HELPERS
    # ========================================================

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        try:
            number = float(value)

            if not np.isfinite(number):
                return None

            return number

        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_amount(value: Any) -> float | None:
        if value is None:
            return None

        cleaned = str(value).strip()

        cleaned = re.sub(
            r"(?i)(?:₹|rs\.?|inr)",
            "",
            cleaned,
        )

        cleaned = cleaned.replace(",", "")
        cleaned = cleaned.strip()

        try:
            amount = float(cleaned)

            if not np.isfinite(amount):
                return None

            if amount < 0:
                return None

            return amount

        except (TypeError, ValueError):
            return None

    # ========================================================
    # DATE HELPERS
    # ========================================================

    @staticmethod
    def _normalize_date_from_match(
        match: re.Match,
        pattern_index: int,
    ) -> str | None:

        try:
            values = [
                int(group)
                for group in match.groups()
            ]

            if pattern_index == 0:
                day, month, year = values
            else:
                year, month, day = values

            if year < 100:
                year += 2000

            if not (
                1900 <= year <= 2100
                and 1 <= month <= 12
                and 1 <= day <= 31
            ):
                return None

            date(year, month, day)

            return (
                f"{year:04d}-"
                f"{month:02d}-"
                f"{day:02d}"
            )

        except (
            TypeError,
            ValueError,
        ):
            return None

    def _parse_date_string(
        self,
        value: str,
    ) -> str | None:

        if not value:
            return None

        for pattern_index, pattern in enumerate(
            self.DATE_PATTERNS
        ):
            match = pattern.search(value)

            if not match:
                continue

            parsed = self._normalize_date_from_match(
                match,
                pattern_index,
            )

            if parsed:
                return parsed

        return None

    def _looks_like_date(
        self,
        line: str,
    ) -> bool:

        for pattern_index, pattern in enumerate(
            self.DATE_PATTERNS
        ):
            match = pattern.search(line)

            if not match:
                continue

            if (
                self._normalize_date_from_match(
                    match,
                    pattern_index,
                )
                is not None
            ):
                return True

        return False

    # ========================================================
    # TIME HELPERS
    # ========================================================

    @classmethod
    def _contains_time(cls, text: str) -> bool:
        return bool(
            cls.TIME_PATTERN.search(text)
        )

    @classmethod
    def _looks_like_time_only(cls, line: str) -> bool:
        stripped = line.strip()

        if not stripped:
            return False

        return bool(
            cls.TIME_PATTERN.fullmatch(stripped)
        )

    @staticmethod
    def _normalize_ocr_spacing(line: str) -> str:
        return re.sub(
            r"(?<=\d)(?=[A-Za-z])",
            " ",
            line,
        )
    
    # ========================================================
    # AMOUNT EXTRACTION
    # ========================================================

    def _extract_amounts(
        self,
        line: str,
    ) -> list[float]:

        if not line:
            return []
        
        line = self._normalize_ocr_spacing(line)

        # Never interpret a standalone time as money.
        if self._looks_like_time_only(line):
            return []

        amounts: list[float] = []

        for match in self.AMOUNT_PATTERN.finditer(line):
            raw_value = match.group(1)

            value = self._parse_amount(
                raw_value
            )

            if value is None:
                continue

            if not (
                self.config.min_item_price
                <= value
                <= self.config.max_item_price
            ):
                continue

            amounts.append(value)

        return amounts

    # ========================================================
    # TEXT CLASSIFICATION
    # ========================================================

    @staticmethod
    def _clean_vendor_value(
        value: str,
    ) -> str:

        value = str(value).strip()

        value = re.sub(
            r"^[\s:,\-]+|[\s:,\-]+$",
            "",
            value,
        )

        value = re.sub(
            r"\s{2,}",
            " ",
            value,
        )

        return value.strip()

    @staticmethod
    def _clean_item_name(
        name: str,
    ) -> str:

        name = str(name).strip()

        name = re.sub(
            r"^[\s:|,\-]+|[\s:|,\-]+$",
            "",
            name,
        )

        name = re.sub(
            r"\s{2,}",
            " ",
            name,
        )

        return name.strip()

    @staticmethod
    def _looks_like_vendor(
        line: str,
    ) -> bool:

        line = line.strip()

        if not line:
            return False

        if len(line) < 2:
            return False

        if len(line) > 100:
            return False

        alpha_count = sum(
            ch.isalpha()
            for ch in line
        )

        digit_count = sum(
            ch.isdigit()
            for ch in line
        )

        if alpha_count < 2:
            return False

        if digit_count > alpha_count:
            return False

        return True

    @staticmethod
    def _contains_too_many_digits(
        text: str,
    ) -> bool:

        if not text:
            return False

        digits = sum(
            ch.isdigit()
            for ch in text
        )

        letters = sum(
            ch.isalpha()
            for ch in text
        )

        return digits > max(
            3,
            int(letters * 0.35),
        )

    # ========================================================
    # ADDRESS / META DETECTION
    # ========================================================

    @staticmethod
    def _looks_like_address_or_postcode(
        line: str,
    ) -> bool:

        text = line.strip()

        if not text:
            return False

        # Correct \b usage.
        if re.search(
            r"\b\d{5,}\b",
            text,
        ):
            return True

        if re.search(
            r"\b\d+\s+[A-Z][A-Z0-9\- ]{2,}\b",
            text,
            flags=re.IGNORECASE,
        ):
            return True

        if re.search(
            r"""
            \b
            (?:
                ST
                |
                STREET
                |
                ROAD
                |
                RD
                |
                AVE
                |
                AVENUE
                |
                DR
                |
                DRIVE
                |
                BLVD
                |
                LN
                |
                LANE
                |
                CT
                |
                COURT
                |
                CIR
                |
                HIGHWAY
                |
                HWY
            )
            \b
            """,
            text,
            flags=re.IGNORECASE | re.VERBOSE,
        ):
            return True

        return False

    @staticmethod
    def _looks_like_transaction_meta(
        line: str,
    ) -> bool:

        text = line.strip()

        if not text:
            return False

        if re.search(
            r"""
            (?:
                ACCOUNT
                |
                REF
                |
                NETWORK
                |
                TERMINAL
                |
                TC\s*#
                |
                APPR
                |
                APPROVAL
                |
                ITEMS\s+SOLD
                |
                STORE\s+HOURS
            )
            """,
            text,
            flags=re.IGNORECASE | re.VERBOSE,
        ):
            return True

        if re.search(
            r"\b[A-Z]{2,}\s*#\s*\d+\b",
            text,
            flags=re.IGNORECASE,
        ):
            return True

        # Payment-card-like numeric sequence.
        if re.search(
            r"\b\d{4}(?:\s+\d{4}){2,}\b",
            text,
        ):
            return True

        if "@" in text:
            return True

        return False

    @classmethod
    def _looks_like_receipt_footer(
        cls,
        line: str,
    ) -> bool:

        upper = line.upper().strip()

        return any(
            token in upper
            for token in cls.FOOTER_TOKENS
        )

    # ========================================================
    # VENDOR
    # ========================================================

    def _extract_vendor(
        self,
        lines: list[str],
    ) -> ExtractionCandidate | None:

        candidates: list[
            ExtractionCandidate
        ] = []

        # ----------------------------------------------------
        # 1. Explicit label
        # ----------------------------------------------------

        for index, line in enumerate(
            lines[: self.config.vendor_search_lines]
        ):

            match = self.VENDOR_LABEL_PATTERN.search(
                line
            )

            if not match:
                continue

            value = self._clean_vendor_value(
                match.group("value")
            )

            if not value:
                continue

            if self._looks_like_address_or_postcode(
                value
            ):
                continue

            candidates.append(
                ExtractionCandidate(
                    value=value,
                    source_text=line,
                    confidence=0.97,
                    extraction_method="vendor_label",
                    region_index=index,
                    evidence=[
                        "explicit_vendor_label"
                    ],
                )
            )

        if candidates:
            return candidates[0]

        # ----------------------------------------------------
        # 2. Top-line heuristic
        # ----------------------------------------------------

        excluded = (
            "DATE",
            "TIME",
            "RECEIPT",
            "INVOICE",
            "GST",
            "TAX",
            "TOTAL",
            "PHONE",
            "TEL",
            "ADDRESS",
            "BILL",
            "SAVE MONEY",
        )

        for index, line in enumerate(
            lines[: self.config.vendor_search_lines]
        ):

            upper = line.upper()

            if any(
                word in upper
                for word in excluded
            ):
                continue

            if self._looks_like_amount_only(line):
                continue

            if self._looks_like_date(line):
                continue

            if self._looks_like_time_only(line):
                continue

            if self._looks_like_address_or_postcode(
                line
            ):
                continue

            if self._looks_like_transaction_meta(
                line
            ):
                continue

            if self._looks_like_vendor(line):
                candidates.append(
                    ExtractionCandidate(
                        value=self._clean_vendor_value(
                            line
                        ),
                        source_text=line,
                        confidence=(
                            0.82
                            if index == 0
                            else 0.70
                        ),
                        extraction_method=(
                            "top_line_heuristic"
                        ),
                        region_index=index,
                        evidence=[
                            "top_of_receipt",
                            "non_numeric_text",
                            "vendor_like_text",
                        ],
                    )
                )

        if candidates:
            return candidates[0]

        return None

    # ========================================================
    # DATE
    # ========================================================

    def _extract_date(
        self,
        lines: list[str],
    ) -> ExtractionCandidate | None:

        search_lines = lines

        # ----------------------------------------------------
        # 1. Explicit date label
        # ----------------------------------------------------

        for index, line in enumerate(
            search_lines
        ):

            label_match = (
                self.DATE_LABEL_PATTERN.search(
                    line
                )
            )

            if not label_match:
                continue

            parsed = self._parse_date_string(
                label_match.group("value")
            )

            if parsed:
                return ExtractionCandidate(
                    value=parsed,
                    source_text=line,
                    confidence=0.99,
                    extraction_method=(
                        "date_label_regex"
                    ),
                    region_index=index,
                    evidence=[
                        "explicit_date_label",
                        "valid_calendar_date",
                    ],
                )

        # ----------------------------------------------------
        # 2. Generic date scan
        # ----------------------------------------------------

        date_candidates: list[
            ExtractionCandidate
        ] = []

        for index, line in enumerate(
            search_lines
        ):

            if self._looks_like_receipt_footer(
                line
            ):
                # Do not blindly skip footer dates.
                # Some receipts repeat the transaction date
                # near the footer, so we continue scanning.
                pass

            for pattern_index, pattern in enumerate(
                self.DATE_PATTERNS
            ):

                match = pattern.search(line)

                if not match:
                    continue

                parsed = (
                    self._normalize_date_from_match(
                        match,
                        pattern_index,
                    )
                )

                if not parsed:
                    continue

                confidence = 0.92

                # Date near transaction metadata is still
                # legitimate, but slightly less strong.
                if self._looks_like_transaction_meta(
                    line
                ):
                    confidence = 0.88

                date_candidates.append(
                    ExtractionCandidate(
                        value=parsed,
                        source_text=line,
                        confidence=confidence,
                        extraction_method=(
                            "date_regex"
                        ),
                        region_index=index,
                        evidence=[
                            "date_regex_match",
                            "valid_calendar_date",
                        ],
                    )
                )

        if not date_candidates:
            return None

        # Prefer earliest valid transaction date.
        date_candidates.sort(
            key=lambda candidate: (
                -candidate.confidence,
                candidate.region_index
                if candidate.region_index is not None
                else 999999,
            )
        )

        return date_candidates[0]

    # ========================================================
    # TOTAL
    # ========================================================

    def _extract_total(
        self,
        lines: list[str],
    ) -> ExtractionCandidate | None:

        candidates: list[
            ExtractionCandidate
        ] = []

        for index, line in enumerate(lines):

            upper_line = line.upper()

            # ------------------------------------------------
            # Strongest total labels first.
            # ------------------------------------------------

            matching_keyword = None

            for keyword in self.total_keywords:
                if keyword in upper_line:
                    matching_keyword = keyword
                    break

            if matching_keyword is None:
                continue

            # ------------------------------------------------
            # NEVER use SUBTOTAL as total.
            # ------------------------------------------------

            if "SUBTOTAL" in upper_line:
                continue

            # ------------------------------------------------
            # Payment line protection.
            # ------------------------------------------------

            if self._is_likely_payment_line(line):
                # "TOTAL" itself can appear with payment text,
                # so only skip when payment wording dominates.
                if not re.search(
                    r"\bTOTAL\b",
                    upper_line,
                ):
                    continue

            amounts = self._extract_amounts(line)

            if amounts:

                amount = amounts[-1]

                if matching_keyword in (
                    "GRAND TOTAL",
                    "TOTAL PURCHASE",
                    "AMOUNT PAYABLE",
                    "TOTAL AMOUNT",
                    "NET AMOUNT",
                ):
                    confidence = 0.98
                else:
                    confidence = 0.94

                candidates.append(
                    ExtractionCandidate(
                        value=amount,
                        source_text=line,
                        confidence=confidence,
                        extraction_method=(
                            "total_keyword_amount"
                        ),
                        region_index=index,
                        evidence=[
                            f"keyword:{matching_keyword}",
                            "amount_on_same_line",
                        ],
                    )
                )

                continue

            # ------------------------------------------------
            # Label on one line, amount on next line.
            # ------------------------------------------------

            if index + 1 < len(lines):

                next_line = lines[index + 1]

                next_amounts = (
                    self._extract_amounts(
                        next_line
                    )
                )

                if next_amounts:

                    amount = next_amounts[-1]

                    if matching_keyword in (
                        "GRAND TOTAL",
                        "TOTAL PURCHASE",
                        "AMOUNT PAYABLE",
                        "TOTAL AMOUNT",
                        "NET AMOUNT",
                    ):
                        confidence = 0.97
                    else:
                        confidence = 0.94

                    candidates.append(
                        ExtractionCandidate(
                            value=amount,
                            source_text=(
                                f"{line} "
                                f"{next_line}"
                            ),
                            confidence=confidence,
                            extraction_method=(
                                "total_keyword_next_line"
                            ),
                            region_index=index,
                            evidence=[
                                f"keyword:{matching_keyword}",
                                "amount_on_next_line",
                            ],
                        )
                    )

        # ----------------------------------------------------
        # IMPORTANT:
        # If multiple totals exist, choose the strongest
        # semantic label, not simply the largest amount.
        # ----------------------------------------------------

        if candidates:

            priority = {
                "GRAND TOTAL": 100,
                "TOTAL PURCHASE": 95,
                "AMOUNT PAYABLE": 95,
                "TOTAL AMOUNT": 95,
                "NET AMOUNT": 90,
                "TOTAL": 80,
            }

            def score(
                candidate: ExtractionCandidate,
            ):
                keyword_score = 0

                for keyword, value in priority.items():
                    if (
                        f"keyword:{keyword}"
                        in candidate.evidence
                    ):
                        keyword_score = value
                        break

                region = (
                    candidate.region_index
                    if candidate.region_index
                    is not None
                    else 999999
                )

                return (
                    keyword_score,
                    candidate.confidence,
                    -region,
                )

            candidates.sort(
                key=score,
                reverse=True,
            )

            return candidates[0]

        # ----------------------------------------------------
        # Fallback:
        # Search final receipt section, but NEVER accept
        # standalone time values.
        # ----------------------------------------------------

        fallback_candidates: list[
            ExtractionCandidate
        ] = []

        start_index = max(
            0,
            len(lines) - 15,
        )

        for index in range(
            start_index,
            len(lines),
        ):

            line = lines[index]

            if self._looks_like_time_only(line):
                continue

            if self._is_likely_payment_line(
                line
            ):
                continue

            if self._looks_like_transaction_meta(
                line
            ):
                continue

            amounts = self._extract_amounts(
                line
            )

            if not amounts:
                continue

            amount = amounts[-1]

            fallback_candidates.append(
                ExtractionCandidate(
                    value=amount,
                    source_text=line,
                    confidence=0.60,
                    extraction_method=(
                        "last_lines_amount_fallback"
                    ),
                    region_index=index,
                    evidence=[
                        "amount_near_receipt_end",
                        "fallback_extraction",
                        "non_payment_line",
                    ],
                )
            )

        if fallback_candidates:

            # Prefer the last valid amount, but not a time.
            return fallback_candidates[-1]

        return None

    # ========================================================
    # PAYMENT / STRUCTURAL HELPERS
    # ========================================================

    @classmethod
    def _is_likely_payment_line(
        cls,
        line: str,
    ) -> bool:

        upper = line.upper()

        return any(
            keyword in upper
            for keyword in cls.PAYMENT_TOKENS
        )

    # ========================================================
    # AMOUNT-ONLY
    # ========================================================

    def _looks_like_amount_only(
        self,
        line: str,
    ) -> bool:

        stripped = line.strip()

        if not stripped:
            return False

        if self._looks_like_time_only(
            stripped
        ):
            return False

        amounts = self._extract_amounts(
            stripped
        )

        if not amounts:
            return False

        remainder = self.AMOUNT_PATTERN.sub(
            "",
            stripped,
        )

        remainder = re.sub(
            r"[\s:₹$,.RsINR\-]+",
            "",
            remainder,
            flags=re.IGNORECASE,
        )

        return remainder == ""

    # ========================================================
    # ITEM VALIDATION
    # ========================================================

    def _valid_item_name(
        self,
        name: str,
    ) -> bool:

        if not name:
            return False

        clean = self._clean_item_name(
            name
        )

        if not clean:
            return False

        if len(clean) < (
            self.config.min_item_name_length
        ):
            return False

        if len(clean) > (
            self.config.max_item_name_length
        ):
            return False

        upper = clean.upper()

        if any(
            token in upper
            for token in self.ITEM_NAME_REJECT_TOKENS
        ):
            return False

        if self._looks_like_address_or_postcode(
            clean
        ):
            return False

        if self._looks_like_transaction_meta(
            clean
        ):
            return False

        if self._looks_like_date(clean):
            return False

        if self._looks_like_time_only(clean):
            return False

        if self._contains_too_many_digits(clean):
            return False

        if "#" in clean or "@" in clean:
            return False

        alpha_count = sum(
            ch.isalpha()
            for ch in clean
        )

        if alpha_count < 2:
            return False

        return True

    # ========================================================
    # ITEM LINE REJECTION
    # ========================================================

    def _is_rejected_item_line(
        self,
        line: str,
    ) -> bool:

        upper = line.upper().strip()

        if not upper:
            return True

        # Dates are never item names.
        if self._looks_like_date(line):
            return True

        # Times are never item names.
        if self._looks_like_time_only(line):
            return True

        # Amount-only lines.
        if self._looks_like_amount_only(line):
            return True

        # Percent-only / tax-rate lines.
        if self.PERCENT_PATTERN.fullmatch(
            line.strip()
        ):
            return True

        # Structural receipt metadata.
        if self._looks_like_address_or_postcode(
            line
        ):
            return True

        if self._looks_like_transaction_meta(
            line
        ):
            return True

        if self._looks_like_receipt_footer(
            line
        ):
            return True

        # Explicit rejected keywords.
        for keyword in self.reject_keywords:

            if keyword not in upper:
                continue

            # These are definitely not products.
            if keyword in {
                "SUBTOTAL",
                "TAX",
                "VAT",
                "GST",
                "CGST",
                "SGST",
                "IGST",
                "DISCOUNT",
                "CHANGE",
                "PAYMENT",
                "INVOICE",
                "ACCOUNT",
                "REF",
                "NETWORK",
                "TERMINAL",
                "APPROVAL",
                "APPR",
                "ITEMS SOLD",
                "STORE HOURS",
            }:
                return True

        if line.startswith("#") and len(line) < 50:
            return True

        return False

    # ========================================================
    # BUILD ITEM
    # ========================================================

    def _build_item_candidate(
        self,
        name: str,
        price: float | None,
        source_text: str,
        index: int,
        extraction_method: str,
        default_confidence: float,
    ) -> ExtractedItem | None:

        if not self._valid_item_name(name):
            return None

        if price is None:
            return None

        if not np.isfinite(price):
            return None

        if not (
            self.config.min_item_price
            <= price
            <= self.config.max_item_price
        ):
            return None

        return ExtractedItem(
            name=self._clean_item_name(name),
            price=float(price),
            source_text=source_text,
            confidence=round(
                default_confidence,
                self.config.confidence_round_digits,
            ),
            extraction_method=extraction_method,
            region_index=index,
            evidence=[
                "validated_item_candidate",
                "named_item",
                "valid_price",
            ],
        )

    # ========================================================
    # ITEMS
    # ========================================================

    def _extract_items(
        self,
        lines: list[str],
        total_candidate: ExtractionCandidate | None,
    ) -> list[ExtractedItem]:

        items: list[ExtractedItem] = []

        total_value = (
            float(total_candidate.value)
            if total_candidate is not None
            else None
        )

        for index, line in enumerate(lines):

            clean_line = line.strip()

            if not clean_line:
                continue

            if self._is_rejected_item_line(
                clean_line
            ):
                continue

            # ------------------------------------------------
            # A line containing a tax rate should never be
            # treated as an item.
            # ------------------------------------------------

            if self.PERCENT_PATTERN.search(
                clean_line
            ):
                continue

            # ------------------------------------------------
            # Quantity x price
            # ------------------------------------------------

            qty_match = self.ITEM_QTY_PRICE.match(
                clean_line
            )

            if qty_match:

                name = self._clean_item_name(
                    qty_match.group("name")
                )

                price = self._parse_amount(
                    qty_match.group("price")
                )

                candidate = (
                    self._build_item_candidate(
                        name=name,
                        price=price,
                        source_text=clean_line,
                        index=index,
                        extraction_method=(
                            "quantity_price_regex"
                        ),
                        default_confidence=0.92,
                    )
                )

                if candidate is not None:
                    items.append(candidate)

                continue

            # ------------------------------------------------
            # Name + price
            # ------------------------------------------------

            end_match = self.ITEM_PRICE_AT_END.match(
                clean_line
            )

            if end_match:

                name = self._clean_item_name(
                    end_match.group("name")
                )

                price = self._parse_amount(
                    end_match.group("price")
                )

                confidence = 0.84

                # If the only candidate looks exactly like
                # the total, downgrade it heavily.
                if (
                    total_value is not None
                    and abs(
                        price - total_value
                    ) < 1e-9
                ):
                    confidence = 0.55

                candidate = (
                    self._build_item_candidate(
                        name=name,
                        price=price,
                        source_text=clean_line,
                        index=index,
                        extraction_method=(
                            "name_price_regex"
                        ),
                        default_confidence=confidence,
                    )
                )

                if candidate is not None:
                    items.append(candidate)

                continue

            # ------------------------------------------------
            # Price + name
            # ------------------------------------------------

            start_match = self.ITEM_PRICE_AT_START.match(
                clean_line
            )

            if start_match:

                price = self._parse_amount(
                    start_match.group("price")
                )

                name = self._clean_item_name(
                    start_match.group("name")
                )

                candidate = (
                    self._build_item_candidate(
                        name=name,
                        price=price,
                        source_text=clean_line,
                        index=index,
                        extraction_method=(
                            "price_name_regex"
                        ),
                        default_confidence=0.78,
                    )
                )

                if candidate is not None:
                    items.append(candidate)

        return self._deduplicate_items(
            items
        )

    # ========================================================
    # DEDUPLICATION
    # ========================================================

    @staticmethod
    def _deduplicate_items(
        items: list[ExtractedItem],
    ) -> list[ExtractedItem]:

        seen: set[
            tuple[str, float]
        ] = set()

        unique: list[
            ExtractedItem
        ] = []

        for item in items:

            key = (
                item.name.casefold(),
                round(item.price, 2),
            )

            if key in seen:
                continue

            seen.add(key)
            unique.append(item)

        return unique

    # ========================================================
    # ITEMS SOLD COUNT
    # ========================================================

    def _extract_expected_item_count(
        self,
        lines: list[str],
    ) -> int | None:

        for line in lines:

            match = (
                self.ITEMS_SOLD_PATTERN.search(
                    line
                )
            )

            if not match:
                continue

            try:
                count = int(
                    match.group("count")
                )

                if 0 <= count <= 10000:
                    return count

            except (
                TypeError,
                ValueError,
            ):
                pass

        return None

    # ========================================================
    # BUSINESS WARNINGS
    # ========================================================

    @staticmethod
    def _add_business_warnings(
        result: ReceiptExtractionResult,
    ) -> None:

        if result.total_amount is None:
            result.warnings.append(
                "Total amount could not be extracted."
            )

        if result.vendor_name is None:
            result.warnings.append(
                "Vendor name could not be extracted."
            )

        if result.transaction_date is None:
            result.warnings.append(
                "Transaction date could not be extracted."
            )

        if not result.items:
            if result.expected_item_count:
                result.warnings.append(
                    "Receipt indicates "
                    f"{result.expected_item_count} item(s), "
                    "but no valid named line items were "
                    "present in the OCR text."
                )
            else:
                result.warnings.append(
                    "No valid named line items were extracted."
                )

        # ----------------------------------------------------
        # Total vs item sum.
        #
        # Do NOT warn if there are no items because the OCR
        # may simply have lost the item-description lines.
        # ----------------------------------------------------

        if (
            result.total_amount is not None
            and result.items
        ):

            item_sum = sum(
                item.price
                for item in result.items
            )

            total = result.total_amount

            tolerance = max(
                1.00,
                0.05 * total,
            )

            if abs(
                item_sum - total
            ) > tolerance:

                result.warnings.append(
                    "Sum of extracted item prices "
                    "differs materially from extracted total."
                )

        # ----------------------------------------------------
        # Expected count vs extracted count.
        # ----------------------------------------------------

        if (
            result.expected_item_count is not None
            and result.item_count
            != result.expected_item_count
        ):

            result.warnings.append(
                "Extracted item count does not match "
                "the receipt's reported ITEMS SOLD count."
            )

    # ========================================================
    # PROCESS ONE FILE
    # ========================================================

    def process_file(
        self,
        json_path: str | Path,
    ) -> ReceiptExtractionResult:

        start = time.perf_counter()

        json_path = Path(json_path)

        receipt_id = json_path.stem

        errors: list[str] = []
        warnings: list[str] = []

        try:

            data = self._load_json(
                json_path
            )

            receipt_id = str(
                data.get(
                    "receipt_id",
                    receipt_id,
                )
            )

            raw_text = str(
                data.get(
                    "original_full_text",
                    "",
                )
            )

            cleaned_text = str(
                data.get(
                    "cleaned_full_text",
                    "",
                )
            )

            lines = self._get_lines(
                data
            )

            vendor_candidate = (
                self._extract_vendor(lines)
            )

            date_candidate = (
                self._extract_date(lines)
            )

            total_candidate = (
                self._extract_total(lines)
            )

            items = self._extract_items(
                lines,
                total_candidate,
            )

            expected_item_count = (
                self._extract_expected_item_count(
                    lines
                )
            )

            result = ReceiptExtractionResult(
                receipt_id=receipt_id,
                input_path=str(json_path),
                status="success",

                vendor_name=(
                    str(
                        vendor_candidate.value
                    )
                    if vendor_candidate
                    else None
                ),

                transaction_date=(
                    str(
                        date_candidate.value
                    )
                    if date_candidate
                    else None
                ),

                items=items,

                total_amount=(
                    float(
                        total_candidate.value
                    )
                    if total_candidate
                    else None
                ),

                vendor_candidate=vendor_candidate,
                date_candidate=date_candidate,
                total_candidate=total_candidate,

                item_count=len(items),
                expected_item_count=expected_item_count,

                raw_text=raw_text,
                cleaned_text=cleaned_text,

                errors=errors,
                warnings=warnings,

                processing_time_ms=0.0,

                timestamp_utc=(
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                ),
            )

            self._add_business_warnings(
                result
            )

            available_fields = sum(
                value is not None
                for value in (
                    result.vendor_name,
                    result.transaction_date,
                    result.total_amount,
                )
            )

            if (
                available_fields == 0
                and not result.items
            ):

                result.status = "failed"

                result.errors.append(
                    "No required receipt fields "
                    "could be extracted."
                )

            elif (
                available_fields < 3
                or not result.items
            ):

                result.status = "partial"

        except Exception as exc:

            logger.exception(
                "Field extraction failed for %s",
                json_path,
            )

            result = ReceiptExtractionResult(
                receipt_id=receipt_id,
                input_path=str(json_path),
                status="failed",

                vendor_name=None,
                transaction_date=None,
                items=[],
                total_amount=None,

                vendor_candidate=None,
                date_candidate=None,
                total_candidate=None,

                item_count=0,
                expected_item_count=None,

                raw_text="",
                cleaned_text="",

                errors=[str(exc)],
                warnings=[],

                processing_time_ms=0.0,

                timestamp_utc=(
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                ),
            )

        result.processing_time_ms = round(
            (
                time.perf_counter()
                - start
            ) * 1000.0,
            3,
        )

        return result

    # ========================================================
    # GET LINES
    # ========================================================

    @staticmethod
    def _get_lines(
        data: dict[str, Any],
    ) -> list[str]:

        regions = data.get(
            "regions",
            [],
        )

        if isinstance(
            regions,
            list,
        ):

            lines: list[str] = []

            for region in regions:

                if not isinstance(
                    region,
                    dict,
                ):
                    continue

                text = str(
                    region.get(
                        "cleaned_text",
                        "",
                    )
                ).strip()

                if text:
                    lines.append(text)

            if lines:
                return lines

        cleaned_text = str(
            data.get(
                "cleaned_full_text",
                "",
            )
        )

        return [
            line.strip()
            for line in cleaned_text.splitlines()
            if line.strip()
        ]

    # ========================================================
    # REGIONS
    # ========================================================

    @staticmethod
    def _get_regions(
        data: dict[str, Any],
    ) -> list[dict[str, Any]]:

        regions = data.get(
            "regions",
            [],
        )

        if not isinstance(
            regions,
            list,
        ):
            return []

        return [
            region
            for region in regions
            if isinstance(
                region,
                dict,
            )
        ]

    # ========================================================
    # SAVE
    # ========================================================

    def _save_result(
        self,
        result: ReceiptExtractionResult,
    ) -> Path:

        output_path = (
            self.output_dir
            / f"{result.receipt_id}.json"
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                asdict(result),
                file,
                indent=2,
                ensure_ascii=False,
            )

        return output_path

    # ========================================================
    # BATCH
    # ========================================================

    def process_batch(
        self,
    ) -> dict[str, Any]:

        logger.info(
            "Starting receipt field extraction..."
        )

        files = self._discover_files()

        results: list[
            ReceiptExtractionResult
        ] = []

        for index, json_path in enumerate(
            files,
            start=1,
        ):

            result = self.process_file(
                json_path
            )

            results.append(result)

            try:

                output_path = (
                    self._save_result(
                        result
                    )
                )

                logger.info(
                    "Extraction saved: %s",
                    output_path,
                )

            except Exception as exc:

                result.status = "failed"

                result.errors.append(
                    f"Output save failed: {exc}"
                )

                logger.exception(
                    "Failed to save extraction result."
                )

            if index % 100 == 0:

                logger.info(
                    "Extracted %d/%d receipts.",
                    index,
                    len(files),
                )

        total_receipts = len(results)

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

        vendor_extracted = sum(
            result.vendor_name is not None
            for result in results
        )

        date_extracted = sum(
            result.transaction_date is not None
            for result in results
        )

        total_extracted = sum(
            result.total_amount is not None
            for result in results
        )

        receipts_with_items = sum(
            bool(result.items)
            for result in results
        )

        summary = {
            "timestamp_utc": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),

            "input_directory": str(
                self.input_dir.resolve()
            ),

            "output_directory": str(
                self.output_dir.resolve()
            ),

            "total_receipts": total_receipts,

            "successful": successful,
            "partial": partial,
            "failed": failed,

            "success_rate": (
                successful / total_receipts
                if total_receipts
                else 0.0
            ),

            "vendor_extraction_rate": (
                vendor_extracted
                / total_receipts
                if total_receipts
                else 0.0
            ),

            "date_extraction_rate": (
                date_extracted
                / total_receipts
                if total_receipts
                else 0.0
            ),

            "total_extraction_rate": (
                total_extracted
                / total_receipts
                if total_receipts
                else 0.0
            ),

            "item_extraction_rate": (
                receipts_with_items
                / total_receipts
                if total_receipts
                else 0.0
            ),

            "total_items_extracted": sum(
                result.item_count
                for result in results
            ),

            "average_items_per_receipt": (
                float(
                    np.mean(
                        [
                            result.item_count
                            for result in results
                        ]
                    )
                )
                if results
                else 0.0
            ),

            "average_processing_time_ms": (
                float(
                    np.mean(
                        [
                            result.processing_time_ms
                            for result in results
                        ]
                    )
                )
                if results
                else 0.0
            ),

            "results": [
                asdict(result)
                for result in results
            ],
        }

        manifest_path = (
            self.output_dir
            / "extraction_manifest.json"
        )

        with manifest_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                summary,
                file,
                indent=2,
                ensure_ascii=False,
            )

        summary["manifest_path"] = str(
            manifest_path
        )

        logger.info(
            "Field extraction completed."
        )

        return summary

    # ========================================================
    # PUBLIC RUN API
    # ========================================================

    def run(
        self,
        json_path: str | Path | None = None,
    ) -> (
        ReceiptExtractionResult
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