import json
import os
from loguru import logger
from typing import Dict

INDEX_PATH = os.path.join("data", "index.json")
os.makedirs("data", exist_ok=True)

def save_json(path: str, data: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _load_index():
    if not os.path.exists(INDEX_PATH):
        return []
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_index(idx):
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False, indent=2)

def update_index(entry: Dict):
    idx = _load_index()
    # remove existing if same path
    idx = [e for e in idx if e.get("path") != entry.get("path")]
    idx.append(entry)
    _save_index(idx)
    logger.debug("Index updated")
