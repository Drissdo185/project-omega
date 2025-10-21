from rapidfuzz import fuzz
from typing import List, Dict
import os
import json
from app.storage.json_store import _load_index, load_json
from loguru import logger

def _list_files_for_label(label: str):
    idx = _load_index()
    return [e for e in idx if e.get("label") == label]

def find_relevant_sections(query: str, folder: str, top_k: int = 3) -> List[Dict]:
    # Stage 1: get files for the folder
    candidates = []
    idx = _load_index()
    files = [e for e in idx if e.get("label") == folder]
    # If no files found in index, also attempt filesystem scan
    if not files:
        folder_path = os.path.join("data", folder)
        if os.path.exists(folder_path):
            for fn in os.listdir(folder_path):
                if fn.endswith(".json"):
                    p = os.path.join(folder_path, fn)
                    files.append({"path": p})
    # naive search: compute fuzzy ratio between query and section content or title
    for f in files:
        try:
            doc = load_json(f["path"])
            for sec in doc.get("sections", []):
                score = max(
                    fuzz.partial_ratio(query, sec.get("content", "")[:1000]),
                    fuzz.partial_ratio(query, sec.get("title", ""))
                )
                if score > 10:
                    candidates.append({
                        "doc_path": f["path"],
                        "section_id": sec.get("id"),
                        "title": sec.get("title"),
                        "content": sec.get("content"),
                        "score": score
                    })
        except Exception as e:
            logger.error(f"Error reading {f.get('path')}: {e}")
    # rank and return top_k
    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    return candidates[:top_k]
