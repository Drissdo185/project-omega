# app/api/functions.py
from app.classifier.labeler import classify_text_short
from app.search.searcher import find_relevant_sections as _find_relevant_sections
from app.utils.azure_openai_client import AzureOpenAIClient
from app.chat.manager import process_user_message
from typing import Dict, Any

def get_registered_functions():
    return [
        {
            "name": "classify_document",
            "description": "Classify input text as HR, IT, or Other.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to classify."}
                },
                "required": ["text"],
            },
        },
        {
            "name": "search_folder",
            "description": "Search for related text snippets in the chosen folder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "folder": {"type": "string", "enum": ["HR", "IT", "Other"]},
                },
                "required": ["query", "folder"],
            },
        },
        {
            "name": "answer_from_context",
            "description": "Generate a precise answer based on the provided contexts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "contexts": {"type": "string"},
                },
                "required": ["question", "contexts"],
            },
        },
    ]


def classify_document(text: str):
    label, score = classify_text_short(text)
    return {"label": label, "confidence": score}


def search_folder(query: str, folder: str):
    """
    Returns list of result dicts:
      { doc_path, section_id, title, content, score }
    """
    results = _find_relevant_sections(query, folder, top_k=5)  # increase top_k for more sources
    # _find_relevant_sections should already provide doc_path, section_id, title, content, score
    print("Search results:", results)
    return {"results": results}


def answer_from_context(session_state: Dict, query: str, top_k: int = 3):
    return process_user_message(session_state, query, top_k=top_k)
