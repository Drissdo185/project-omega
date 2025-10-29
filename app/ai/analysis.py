""" AI analysis documents analysis, labeling, and metadata extraction """

from enum import Enum


class DocumentCategory(str, Enum):
    """Document categories for classification"""
    CV = "cv"
    SCIENCE_PAPER = "science_paper"
    GENERAL = "general"



class ContentType(str, Enum):
    """Types of content found in documents"""
    TEXT = "text"
    TABLE = "table"
    CHART = "chart"
    FORM = "form"
    IMAGE = "image"
    MIXED = "mixed"
    UNKNOWN = "unknown"
