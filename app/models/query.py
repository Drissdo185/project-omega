# app/models/query.py
from dataclasses import dataclass
from typing import List, Optional, Dict

@dataclass
class QueryResult:
    """Result from vision-based query"""
    query: str
    answer: str
    page_numbers: List[int]  # Pages used
    document_id: str
    sources: List[Dict]  # Source details
    total_cost: float = 0.0