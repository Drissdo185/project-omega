"""
Factory for creating document processors.
"""

import logging
from pathlib import Path
from typing import Dict, Type

from processors.base import BaseProcessor
from processors.pdf import PDFProcessor
from exceptions import ProcessingError

logger = logging.getLogger(__name__)


class ProcessorFactory:
    """
    Factory for creating appropriate document processors based on file type.
    """

    # Registry of processors by file extension
    _processors: Dict[str, Type[BaseProcessor]] = {
        '.pdf': PDFProcessor,
    }

    @classmethod
    def get_processor(cls, file_type: str) -> BaseProcessor:
        """
        Get a processor instance for the specified file type.

        Args:
            file_type: File extension (e.g., '.pdf', 'pdf') or file path.

        Returns:
            BaseProcessor: Processor instance for the file type.

        Raises:
            ProcessingError: If no processor is available for the file type.
        """
        # Normalize file type
        if '.' in file_type and '/' in file_type:
            # It's a file path
            file_type = Path(file_type).suffix
        elif not file_type.startswith('.'):
            # Add leading dot if missing
            file_type = f'.{file_type}'

        file_type = file_type.lower()

        # Get processor class
        processor_class = cls._processors.get(file_type)

        if processor_class is None:
            raise ProcessingError(
                f"No processor available for file type: {file_type}. "
                f"Supported types: {', '.join(cls._processors.keys())}"
            )

        # Create and return processor instance
        processor = processor_class()
        logger.debug(f"Created processor: {processor}")
        return processor

    @classmethod
    def get_processor_for_file(cls, file_path: str) -> BaseProcessor:
        """
        Get a processor instance for a specific file.

        Args:
            file_path: Path to the file.

        Returns:
            BaseProcessor: Processor instance for the file.

        Raises:
            ProcessingError: If no processor is available for the file type.
        """
        return cls.get_processor(file_path)

    @classmethod
    def register_processor(cls, file_type: str, processor_class: Type[BaseProcessor]) -> None:
        """
        Register a new processor for a file type.

        Args:
            file_type: File extension (e.g., '.docx').
            processor_class: Processor class to handle this file type.
        """
        if not file_type.startswith('.'):
            file_type = f'.{file_type}'

        file_type = file_type.lower()
        cls._processors[file_type] = processor_class
        logger.info(f"Registered processor for {file_type}: {processor_class.__name__}")

    @classmethod
    def supported_types(cls) -> list:
        """
        Get list of supported file types.

        Returns:
            list: List of supported file extensions.
        """
        return list(cls._processors.keys())

    @classmethod
    def __repr__(cls) -> str:
        return f"ProcessorFactory(supported={cls.supported_types()})"
