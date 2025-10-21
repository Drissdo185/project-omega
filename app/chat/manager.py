# app/chat/manager.py
import json
import os
from datetime import datetime
from loguru import logger

from app.utils.azure_openai_client import AzureOpenAIClient
from app.classifier.labeler import classify_text_short
from app.search.searcher import find_relevant_sections

# -------------------------------------------------
# Strict Answer Prompt
# -------------------------------------------------
ANSWER_PROMPT_TEMPLATE = """
You are a strict assistant that MUST answer using ONLY the provided CONTEXTS.
Each context below is labeled [1], [2], etc. and contains exact text excerpts from documents.

INSTRUCTIONS:
- For each context labeled [n] that contains information directly answering the QUESTION, return exactly a numbered line:
  [n] <exact excerpt from context n that answers the question>
- Omit any context that does not contain an answer.
- Do NOT add any additional explanation, paraphrase, or new content.
- Do NOT invent or add facts.
- If none of the contexts contain an answer, reply EXACTLY:
I don't know

QUESTION:
{question}

CONTEXTS:
{contexts}

Remember: only return the numbered listing lines or exactly "I don't know".
"""


# -------------------------------------------------
# Log helper — persists to /logs/call_log.json
# -------------------------------------------------
def _log_call(session_state, fn_name, args, result_summary):
    """Record a function call in memory and to disk."""
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "function": fn_name,
        "args": args,
        "result": result_summary,
    }

    # in-memory state
    try:
        if session_state is not None:
            session_state.setdefault("call_log", []).append(entry)
    except Exception as e:
        logger.warning(f"Could not append to session_state.call_log: {e}")

    # persist to file
    try:
        os.makedirs("logs", exist_ok=True)
        log_path = os.path.join("logs", "call_log.json")
        existing = []
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                try:
                    existing = json.load(f)
                    if not isinstance(existing, list):
                        existing = []
                except Exception:
                    existing = []
        existing.append(entry)
        existing = existing[-1000:]
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to persist call_log.json: {e}")

    return entry


# -------------------------------------------------
# Helper formatters
# -------------------------------------------------
def _format_sources(sources):
    lines = []
    for idx, s in enumerate(sources, start=1):
        filename = os.path.basename(s.get("doc_path", "")) or "unknown.json"
        title = s.get("title") or s.get("section_id") or ""
        lines.append(f"[{idx}] {filename} - {title}")
    return "\n".join(lines)


def _make_contexts_string(contexts):
    parts = []
    for i, c in enumerate(contexts, start=1):
        text = c.get("content", "").strip()
        parts.append(f"[{i}] {text}")
    return "\n\n".join(parts)


# -------------------------------------------------
# Core message processor with logging
# -------------------------------------------------
def process_user_message(session_state, message: str, top_k: int = 3):
    """
    Process a user message end-to-end with detailed logging:
      1. Classify message (HR / IT / Other)
      2. Search for relevant contexts
      3. Generate strict [1][2][3] answer
      4. Persist logs for each stage
    """
    # --- Step 1: Classification
    label, score = classify_text_short(message)
    folder = label if label in ("HR", "IT") else "Other"
    _log_call(session_state, "classify_text_short", {"message": message}, f"Label={label}, Score={score}")

    # Add user message to chat history
    session_state.setdefault("history", []).append({"role": "user", "text": message, "folder": folder})

    # --- Step 2: Search contexts in the determined folder
    contexts = find_relevant_sections(message, folder, top_k=top_k)
    _log_call(session_state, "find_relevant_sections", {"folder": folder, "top_k": top_k}, f"Found {len(contexts)} sections")

    if not contexts:
        # No context → "I don't know"
        assistant_text = "I don't know"
        _log_call(session_state, "generate_answer", {"folder": folder}, "No context found → I don't know")
        session_state.setdefault("history", []).append({"role": "assistant", "text": assistant_text, "folder": folder})
        session_state.setdefault("sources_history", []).append([])
        return {"answer": assistant_text, "contexts": [], "sources": [], "folder": folder}

    # --- Step 3: Generate strict answer using OpenAI
    client = AzureOpenAIClient()
    contexts_block = _make_contexts_string(contexts)
    prompt = ANSWER_PROMPT_TEMPLATE.format(question=message, contexts=contexts_block)

    _log_call(session_state, "AzureOpenAIClient.call_completion", {"folder": folder, "prompt_len": len(prompt)}, "Invoking model")

    try:
        resp = client.call_completion(prompt=prompt, temperature=0.0, max_tokens=800, system_message="You are a strict extractor.")
        model_text = resp.get("text", "").strip()
    except Exception as e:
        model_text = ""
        _log_call(session_state, "AzureOpenAIClient.call_completion", {"error": str(e)}, "Exception occurred")

    # Validate strict output format
    if not model_text:
        assistant_text = "I don't know"
        _log_call(session_state, "validate_output", {"model_text": model_text}, "Empty model response")
    elif model_text == "I don't know":
        assistant_text = "I don't know"
        _log_call(session_state, "validate_output", {"model_text": model_text}, "Model said I don't know")
    else:
        lines = [ln.strip() for ln in model_text.splitlines() if ln.strip()]
        valid = any(ln.startswith("[") and "]" in ln for ln in lines)
        if not valid:
            assistant_text = "I don't know"
            _log_call(session_state, "validate_output", {"model_text": model_text}, "Invalid output format")
        else:
            assistant_text = "\n".join(lines)
            _log_call(session_state, "validate_output", {"model_text": model_text}, f"Valid output with {len(lines)} lines")

    # --- Step 4: Prepare sources
    indices_returned = []
    if assistant_text != "I don't know":
        for ln in assistant_text.splitlines():
            ln = ln.strip()
            if ln.startswith("["):
                try:
                    idx = int(ln[1:ln.index("]")])
                    indices_returned.append(idx)
                except Exception:
                    continue

    if indices_returned:
        selected_sources = []
        for idx in indices_returned:
            if 1 <= idx <= len(contexts):
                c = contexts[idx - 1]
                selected_sources.append({
                    "doc_path": c.get("doc_path"),
                    "title": c.get("title") or c.get("section_id"),
                    "score": c.get("score"),
                })
    else:
        selected_sources = [{"doc_path": c.get("doc_path"), "title": c.get("title") or c.get("section_id"), "score": c.get("score")} for c in contexts]

    _log_call(session_state, "prepare_sources", {"count": len(selected_sources)}, "Sources ready")

    # --- Step 5: Save final chat message
    session_state.setdefault("history", []).append({"role": "assistant", "text": assistant_text, "folder": folder})
    session_state.setdefault("sources_history", []).append(selected_sources)

    _log_call(session_state, "process_user_message", {"folder": folder}, "Completed pipeline")

    sources_block = _format_sources(selected_sources)
    final_message = assistant_text
    if assistant_text != "I don't know":
        final_message = assistant_text + "\n\nSources:\n" + sources_block

    return {"answer": final_message, "contexts": contexts, "sources": selected_sources, "folder": folder}
