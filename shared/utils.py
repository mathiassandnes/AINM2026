"""Common utilities for competition tasks."""

import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent


def load_json(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_json(data: dict, path: str | Path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def timer(func):
    """Simple timing decorator."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} took {elapsed:.2f}s")
        return result
    return wrapper
