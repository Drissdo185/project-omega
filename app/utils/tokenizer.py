from typing import List
import logging

try:
    import tiktoken
    _ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENCODER = None
    logging.warning("tiktoken not available. Falling back to naive token estimate.")

def count_tokens(text: str) -> int:
    if not text:
        return 0
    if _ENCODER:
        return len(_ENCODER.encode(text))
    # naive fallback
    return max(1, len(text.split()))

def split_to_paragraphs(text: str) -> List[str]:
    # split on double newline or newline
    paras = []
    for block in text.split("\n\n"):
        b = block.strip()
        if b:
            paras.extend([p.strip() for p in b.split("\n") if p.strip()])
    return paras

def chunk_by_tokens(paragraphs: List[str], chunk_size: int, overlap: int):
    """
    Returns list of dicts: { 'text': ..., 'tokens': n, 'paragraph_indices': (start,end) }
    """
    chunks = []
    current = []
    current_tokens = 0
    start_idx = 0

    for i, p in enumerate(paragraphs):
        tcount = count_tokens(p)
        if current_tokens + tcount <= chunk_size or not current:
            current.append(p)
            current_tokens += tcount
        else:
            # flush current chunk
            chunk_text = "\n\n".join(current)
            chunks.append({"text": chunk_text, "tokens": current_tokens, "start_idx": start_idx, "end_idx": i-1})
            # For overlap: compute overlap paragraphs by token count
            if overlap > 0:
                # keep paragraphs from end backward until overlap tokens satisfied
                overlap_paras = []
                overlap_tokens = 0
                for j in range(len(current)-1, -1, -1):
                    para = current[j]
                    overlap_paras.insert(0, para)
                    overlap_tokens += count_tokens(para)
                    if overlap_tokens >= overlap:
                        break
                current = overlap_paras.copy()
                current_tokens = sum(count_tokens(x) for x in current)
                start_idx = i - len(current)
            else:
                current = []
                current_tokens = 0
                start_idx = i
            # add current paragraph
            if p not in current:
                current.append(p)
                current_tokens += tcount

    # flush remaining
    if current:
        chunk_text = "\n\n".join(current)
        chunks.append({"text": chunk_text, "tokens": current_tokens, "start_idx": start_idx, "end_idx": len(paragraphs)-1})
    return chunks
