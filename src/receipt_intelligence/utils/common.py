"""
common.py
Utility functions for the Carbon Crunch Receipt Information Extraction project.

"""

import os
import re
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
import yaml
import joblib
from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations

from src.receipt_intelligence import logger


# --------------------------------------------------------------------------- #
# Generic I/O utilities
# --------------------------------------------------------------------------- #

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads a yaml file and returns its content as a ConfigBox.

    Args:
        path_to_yaml (Path): path to the yaml file

    Raises:
        ValueError: if the yaml file is empty
        e: any other exception raised while reading

    Returns:
        ConfigBox: parsed yaml content
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """Creates a list of directories.

    Args:
        path_to_directories (list): list of directory paths to create
        verbose (bool, optional): log each directory creation. Defaults to True.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """Saves a dict as a json file.

    Args:
        path (Path): path to the json file
        data (dict): data to save
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"json file saved at: {path}")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """Loads a json file's content.

    Args:
        path (Path): path to the json file

    Returns:
        ConfigBox: content as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)
    logger.info(f"json file loaded successfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: object, path: Path):
    """Saves data as a binary file using joblib.

    Args:
        data (object): data to save
        path (Path): path to the binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> object:
    """Loads a binary file using joblib.

    Args:
        path (Path): path to the binary file

    Returns:
        object: the loaded object
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data


@ensure_annotations
def get_size(path: Path) -> str:
    """Gets file size in KB.

    Args:
        path (Path): path to the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"


# --------------------------------------------------------------------------- #
# Image preprocessing utilities
# --------------------------------------------------------------------------- #

def load_image(path: Path) -> np.ndarray:
    """Loads an image from disk in BGR format.

    Raises:
        FileNotFoundError: if the image cannot be read (missing/corrupt file)
    """
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image at: {path}")
    return image


def save_image(path: Path, image: np.ndarray):
    """Saves an image to disk."""
    cv2.imwrite(str(path), image)
    logger.info(f"image saved at: {path}")
