import os
from pathlib import Path
from typing import Final

from ._get_dataset import ALL_LANGUAGES, LANGUAGE_TYPES, get_dataset

__version__ = "0.2.0"

FLEURS_DATASETS_CACHE: Final[Path] = Path(
    os.getenv("FLEURS_DATASETS_CACHE", "~/.cache/huggingface/datasets/fleurs")
).expanduser()

__all__ = [
    "ALL_LANGUAGES",
    "get_dataset",
    "LANGUAGE_TYPES",
]
