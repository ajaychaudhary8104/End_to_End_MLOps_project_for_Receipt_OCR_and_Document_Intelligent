from src.receipt_intelligence import logger
import shutil
import zipfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import gdown
from src.receipt_intelligence.entity.config_entity import DataIngestionConfig

class DataIngestion:
    """
    Production-grade data ingestion pipeline for a Google Drive ZIP file.

    Pipeline:

        Google Drive file
              ↓
        Download ZIP
              ↓
        Validate ZIP
              ↓
        Secure extraction
              ↓
        Return extracted dataset directory

    Expected config attributes:
        source_URL
        local_data_file
        unzip_dir
    """

    def __init__(self, config: DataIngestionConfig):
        self.config = config

        self.source_url = str(config.source_URL).strip()
        self.download_path = Path(config.local_data_file)
        self.extract_path = Path(config.unzip_dir)

        if not self.source_url:
            raise ValueError("Google Drive source_URL cannot be empty.")

        if not self.download_path:
            raise ValueError("local_data_file cannot be empty.")

        if not self.extract_path:
            raise ValueError("unzip_dir cannot be empty.")

    # ==========================================================
    # GOOGLE DRIVE URL VALIDATION
    # ==========================================================

    @staticmethod
    def _extract_google_drive_file_id(url: str) -> str:
        """
        Extract Google Drive file ID from common Google Drive URLs.

        Supported examples:

            https://drive.google.com/file/d/<FILE_ID>/view

            https://drive.google.com/uc?id=<FILE_ID>

            https://drive.google.com/open?id=<FILE_ID>
        """

        parsed = urlparse(url)

        # ------------------------------------------------------
        # /file/d/<ID>/view
        # ------------------------------------------------------
        if "/file/d/" in parsed.path:

            parts = parsed.path.split("/file/d/")

            if len(parts) == 2:

                file_id = parts[1].split("/")[0].strip()

                if file_id:
                    return file_id

        # ------------------------------------------------------
        # ?id=<ID>
        # ------------------------------------------------------
        query_params = parse_qs(parsed.query)

        file_id = query_params.get("id")

        if file_id and file_id[0].strip():
            return file_id[0].strip()

        raise ValueError(
            f"Unable to extract Google Drive file ID from URL: {url}"
        )

    # ==========================================================
    # DOWNLOAD GOOGLE DRIVE ZIP FILE
    # ==========================================================

    def download_file(self) -> str:
        """
        Download a ZIP file from Google Drive.

        Returns:
            str:
                Path to the downloaded ZIP file.
        """

        try:

            logger.info(
                "Starting Google Drive file download..."
            )

            logger.info(
                "Source URL: %s",
                self.source_url
            )

            logger.info(
                "Download path: %s",
                self.download_path
            )

            # --------------------------------------------------
            # Create parent directory
            # --------------------------------------------------

            self.download_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            # --------------------------------------------------
            # Extract Google Drive file ID
            # --------------------------------------------------

            file_id = self._extract_google_drive_file_id(
                self.source_url
            )

            logger.info(
                "Google Drive file ID detected: %s",
                file_id
            )

            # --------------------------------------------------
            # Convert to direct download URL
            # --------------------------------------------------

            download_url = (
                f"https://drive.google.com/uc?id={file_id}"
            )

            logger.info(
                "Using Google Drive direct download endpoint."
            )

            # --------------------------------------------------
            # Remove stale file if it exists
            # --------------------------------------------------

            if self.download_path.exists():

                logger.warning(
                    "Existing download found. Removing: %s",
                    self.download_path
                )

                if self.download_path.is_dir():
                    shutil.rmtree(self.download_path)

                else:
                    self.download_path.unlink()

            # --------------------------------------------------
            # Download
            # --------------------------------------------------

            downloaded_path = gdown.download(
                url=download_url,
                output=str(self.download_path),
                quiet=False
            )

            if downloaded_path is None:
                raise RuntimeError(
                    "gdown failed to download the Google Drive file."
                )

            downloaded_path = Path(downloaded_path)

            # --------------------------------------------------
            # Validate file exists
            # --------------------------------------------------

            if not downloaded_path.exists():

                raise FileNotFoundError(
                    "Downloaded file does not exist: "
                    f"{downloaded_path}"
                )

            if not downloaded_path.is_file():

                raise RuntimeError(
                    "Downloaded path is not a regular file: "
                    f"{downloaded_path}"
                )

            # --------------------------------------------------
            # Validate file size
            # --------------------------------------------------

            file_size = downloaded_path.stat().st_size

            if file_size <= 0:

                raise RuntimeError(
                    "Downloaded file is empty."
                )

            logger.info(
                "Downloaded file size: %.2f MB",
                file_size / (1024 * 1024)
            )

            # --------------------------------------------------
            # Validate ZIP
            # --------------------------------------------------

            if not zipfile.is_zipfile(downloaded_path):

                # Read first bytes to detect common HTML errors.
                try:

                    with open(
                        downloaded_path,
                        "rb"
                    ) as f:

                        header = f.read(500)

                except Exception:

                    header = b""

                header_lower = header.lower()

                if (
                    b"<html" in header_lower
                    or b"<!doctype" in header_lower
                    or b"google drive" in header_lower
                ):

                    raise RuntimeError(
                        "Google Drive returned an HTML page instead "
                        "of the expected ZIP file. "
                        "Check that the file is shared as "
                        "'Anyone with the link'."
                    )

                raise RuntimeError(
                    f"Downloaded file is not a valid ZIP archive: "
                    f"{downloaded_path}"
                )

            logger.info(
                "ZIP validation successful."
            )

            logger.info(
                "Google Drive file downloaded successfully: %s",
                downloaded_path
            )

            return str(downloaded_path)

        except Exception as e:

            logger.exception(
                "Failed to download Google Drive ZIP file."
            )

            raise RuntimeError(
                "Google Drive ZIP download failed."
            ) from e

    # ==========================================================
    # SECURE ZIP EXTRACTION
    # ==========================================================

    @staticmethod
    def _validate_zip_members(
        zip_ref: zipfile.ZipFile,
        extraction_path: Path
    ) -> None:
        """
        Validate ZIP members against path traversal attacks.
        """

        extraction_root = extraction_path.resolve()

        for member in zip_ref.infolist():

            member_path = (
                extraction_root / member.filename
            ).resolve()

            try:

                member_path.relative_to(
                    extraction_root
                )

            except ValueError:

                raise RuntimeError(
                    "Unsafe ZIP archive detected. "
                    f"Path traversal attempt: {member.filename}"
                )

    # ==========================================================
    # EXTRACT ZIP FILE
    # ==========================================================

    def extract_zip_file(self) -> str:
        """
        Extract the downloaded ZIP file securely.

        Returns:
            str:
                Extraction directory.
        """

        try:

            zip_path = Path(
                self.download_path
            )

            # --------------------------------------------------
            # Validate ZIP exists
            # --------------------------------------------------

            if not zip_path.exists():

                raise FileNotFoundError(
                    f"ZIP file not found: {zip_path}"
                )

            if not zip_path.is_file():

                raise RuntimeError(
                    f"ZIP path is not a file: {zip_path}"
                )

            # --------------------------------------------------
            # Validate ZIP format
            # --------------------------------------------------

            if not zipfile.is_zipfile(zip_path):

                raise RuntimeError(
                    f"Invalid ZIP archive: {zip_path}"
                )

            # --------------------------------------------------
            # Create extraction directory
            # --------------------------------------------------

            self.extract_path.mkdir(
                parents=True,
                exist_ok=True
            )

            logger.info(
                "Extraction directory: %s",
                self.extract_path
            )

            # --------------------------------------------------
            # Open ZIP
            # --------------------------------------------------

            with zipfile.ZipFile(
                zip_path,
                mode="r"
            ) as zip_ref:

                members = zip_ref.infolist()

                if not members:

                    raise RuntimeError(
                        "ZIP archive is empty."
                    )

                logger.info(
                    "ZIP contains %d entries.",
                    len(members)
                )

                # --------------------------------------------------
                # Security validation
                # --------------------------------------------------

                self._validate_zip_members(
                    zip_ref,
                    self.extract_path
                )

                # --------------------------------------------------
                # Extract
                # --------------------------------------------------

                for member in members:

                    logger.debug(
                        "Extracting: %s",
                        member.filename
                    )

                    zip_ref.extract(
                        member,
                        path=self.extract_path
                    )

            # --------------------------------------------------
            # Validate extraction
            # --------------------------------------------------

            extracted_files = [
                p
                for p in self.extract_path.rglob("*")
                if p.is_file()
            ]

            if not extracted_files:

                raise RuntimeError(
                    "ZIP extraction completed but no files "
                    "were found."
                )

            logger.info(
                "Successfully extracted %d files.",
                len(extracted_files)
            )

            for file_path in extracted_files:

                logger.debug(
                    "Extracted: %s",
                    file_path
                )

            logger.info(
                "ZIP extraction completed successfully."
            )

            return str(self.extract_path)

        except Exception as e:

            logger.exception(
                "Failed to extract ZIP file."
            )

            raise RuntimeError(
                "ZIP extraction failed."
            ) from e

    # ==========================================================
    # COMPLETE INGESTION PIPELINE
    # ==========================================================

    def run(self) -> str:
        """
        Execute complete data ingestion pipeline.

        Pipeline:

            Google Drive ZIP
                    ↓
            Download
                    ↓
            Validate
                    ↓
            Secure extraction
                    ↓
            Return extracted directory

        Returns:
            str:
                Path to extracted dataset.
        """

        logger.info(
            "Starting data ingestion..."
        )

        # ------------------------------------------------------
        # Step 1: Download
        # ------------------------------------------------------

        zip_path = self.download_file()

        logger.info(
            "Download completed: %s",
            zip_path
        )

        # ------------------------------------------------------
        # Step 2: Extract
        # ------------------------------------------------------

        extracted_path = self.extract_zip_file()

        logger.info(
            "Extraction completed: %s",
            extracted_path
        )

        # ------------------------------------------------------
        # Final validation
        # ------------------------------------------------------

        extracted_files = [
            p
            for p in Path(extracted_path).rglob("*")
            if p.is_file()
        ]

        if not extracted_files:

            raise RuntimeError(
                "Data ingestion completed but the extracted "
                "directory contains no files."
            )

        logger.info(
            "Final dataset validation successful."
        )

        logger.info(
            "Total extracted files: %d",
            len(extracted_files)
        )

        logger.info(
            "Data ingestion completed successfully."
        )

        return str(extracted_path)