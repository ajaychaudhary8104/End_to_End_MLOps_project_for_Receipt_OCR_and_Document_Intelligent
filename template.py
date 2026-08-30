from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s]: %(message)s:"
)

project_name = "receipt_intelligence"

list_of_files = [
    # Source
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/utils/__init__.py",

    # Data
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "data/output/.gitkeep",

    # Notebooks
    "notebooks/01_eda.ipynb",
    "notebooks/02_ocr.ipynb",
    "notebooks/03_extraction.ipynb",

    # Config / MLOps
    "config/config.yaml",
    "params.yaml",
    "dvc.yaml",

    # Tests
    "tests/__init__.py",

    # Deployment
    "Dockerfile",

    # Root files
    "app.py",
    "requirements.txt",
    "setup.py",
    "pyproject.toml",
    "README.md",
    ".gitignore",

    # CI/CD
    ".github/workflows/.gitkeep",
]


for filepath in list_of_files:

    filepath = Path(filepath)
    filedir = filepath.parent

    if filedir != Path("."):
        filedir.mkdir(parents=True, exist_ok=True)
        logging.info(
            f"Creating directory: {filedir} "
            f"for file: {filepath.name}"
        )

    if not filepath.exists() or filepath.stat().st_size == 0:
        filepath.touch()
        logging.info(f"Creating file: {filepath}")
    else:
        logging.info(f"{filepath.name} already exists")
