import pdfplumber
from typing import List
from loguru import logger

def extract_text_pages(pdf_path: str) -> List[str]:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for p in pdf.pages:
            try:
                text = p.extract_text() or ""
            except Exception as e:
                logger.error(f"Error extracting page: {e}")
                text = ""
            pages.append(text)
    return pages

def extract_full_text(pdf_path: str) -> str:
    pages = extract_text_pages(pdf_path)
    return "\n\n".join(pages)
