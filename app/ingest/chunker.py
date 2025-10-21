import os
import json
import uuid
from datetime import datetime
from typing import List, Dict
from loguru import logger

from app.utils.tokenizer import split_to_paragraphs, chunk_by_tokens
from app.storage.json_store import save_json, update_index
from app.classifier.labeler import classify_text_short

DEFAULT_CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE_TOKENS", 800))
DEFAULT_OVERLAP = int(os.environ.get("CHUNK_OVERLAP_TOKENS", 60))
BATCH_SAVE_SIZE = int(os.environ.get("BATCH_SAVE_SIZE", 20))

def chunk_text_and_save(document_title: str, text: str, source_filename: str, out_base_dir: str = "data"):
    # classify document (use first ~2000 tokens)
    preview = text[:8000]
    label, score = classify_text_short(preview)

    paragraphs = split_to_paragraphs(text)
    chunks = chunk_by_tokens(paragraphs, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP)

    # form json sections
    sections = []
    for i, ch in enumerate(chunks):
        sec = {
            "id": str(uuid.uuid4()),
            "title": f"Section {i+1}",
            "content": ch["text"],
            "page_start": ch["start_idx"],
            "page_end": ch["end_idx"],
            "tokens": ch["tokens"]
        }
        sections.append(sec)

    json_obj = {
        "title": document_title,
        "sections": sections,
        "metadata": {
            "source_filename": source_filename,
            "detected_folder": label,
            "detected_score": score,
            "created_at": datetime.utcnow().isoformat()
        }
    }

    # save to folder
    folder = os.path.join(out_base_dir, label)
    os.makedirs(folder, exist_ok=True)
    basename = os.path.splitext(os.path.basename(source_filename))[0]
    out_path = os.path.join(folder, f"{basename}.json")

    save_json(out_path, json_obj)
    update_index({
        "path": out_path,
        "title": document_title,
        "label": label,
        "num_sections": len(sections),
        "created_at": json_obj["metadata"]["created_at"]
    })
    logger.info(f"Saved chunked JSON to {out_path}")
    return out_path, len(sections), label
