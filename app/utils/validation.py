# app/utils/validation.py
"""
Input validation and sanitization utilities
"""
from pathlib import Path
from typing import Optional, List
from loguru import logger


class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


class FileValidator:
    """Validate file inputs"""
    
    # Maximum file size: 100MB
    MAX_FILE_SIZE = 100 * 1024 * 1024
    
    # Allowed file extensions
    ALLOWED_EXTENSIONS = {'.pdf'}
    
    # Maximum filename length
    MAX_FILENAME_LENGTH = 255
    
    @classmethod
    def validate_pdf_file(cls, file_path: str) -> None:
        """
        Validate PDF file before processing
        
        Args:
            file_path: Path to PDF file
            
        Raises:
            ValidationError: If validation fails
        """
        path = Path(file_path)
        
        # Check file exists
        if not path.exists():
            raise ValidationError(f"File not found: {file_path}")
        
        # Check is file (not directory)
        if not path.is_file():
            raise ValidationError(f"Path is not a file: {file_path}")
        
        # Check file extension
        if path.suffix.lower() not in cls.ALLOWED_EXTENSIONS:
            raise ValidationError(
                f"Invalid file type: {path.suffix}. Only PDF files are allowed."
            )
        
        # Check file size
        file_size = path.stat().st_size
        if file_size == 0:
            raise ValidationError("File is empty")
        
        if file_size > cls.MAX_FILE_SIZE:
            size_mb = file_size / (1024 * 1024)
            max_mb = cls.MAX_FILE_SIZE / (1024 * 1024)
            raise ValidationError(
                f"File too large: {size_mb:.1f}MB. Maximum size is {max_mb:.1f}MB"
            )
        
        # Check filename length
        if len(path.name) > cls.MAX_FILENAME_LENGTH:
            raise ValidationError(
                f"Filename too long: {len(path.name)} characters. "
                f"Maximum is {cls.MAX_FILENAME_LENGTH} characters"
            )
        
        logger.info(f"✅ File validation passed: {path.name} ({file_size/1024:.1f}KB)")
    
    @classmethod
    def validate_image_path(cls, image_path: str) -> None:
        """
        Validate image file path
        
        Args:
            image_path: Path to image file
            
        Raises:
            ValidationError: If validation fails
        """
        path = Path(image_path)
        
        if not path.exists():
            raise ValidationError(f"Image file not found: {image_path}")
        
        if not path.is_file():
            raise ValidationError(f"Image path is not a file: {image_path}")
        
        # Check image extensions
        valid_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
        if path.suffix.lower() not in valid_extensions:
            raise ValidationError(
                f"Invalid image type: {path.suffix}. "
                f"Allowed types: {', '.join(valid_extensions)}"
            )
        
        # Check file size (images should be reasonable)
        file_size = path.stat().st_size
        if file_size == 0:
            raise ValidationError(f"Image file is empty: {image_path}")
        
        if file_size > 10 * 1024 * 1024:  # 10MB max for images
            raise ValidationError(
                f"Image file too large: {file_size/1024/1024:.1f}MB. "
                f"Maximum is 10MB"
            )


class DocumentValidator:
    """Validate document objects"""
    
    @classmethod
    def validate_document_id(cls, doc_id: str) -> None:
        """
        Validate document ID format
        
        Args:
            doc_id: Document ID to validate
            
        Raises:
            ValidationError: If validation fails
        """
        if not doc_id:
            raise ValidationError("Document ID cannot be empty")
        
        if not isinstance(doc_id, str):
            raise ValidationError(f"Document ID must be string, got {type(doc_id)}")
        
        # Check format: doc_[hash]
        if not doc_id.startswith("doc_"):
            raise ValidationError(
                f"Invalid document ID format: {doc_id}. "
                f"Expected format: doc_[hash]"
            )
        
        # Check length (doc_ + 12 char hash)
        if len(doc_id) != 16:
            raise ValidationError(
                f"Invalid document ID length: {len(doc_id)}. "
                f"Expected 16 characters (doc_ + 12 char hash)"
            )
    
    @classmethod
    def validate_page_number(cls, page_num: int, total_pages: int) -> None:
        """
        Validate page number is within document bounds
        
        Args:
            page_num: Page number to validate
            total_pages: Total number of pages in document
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(page_num, int):
            raise ValidationError(f"Page number must be integer, got {type(page_num)}")
        
        if page_num < 1:
            raise ValidationError(f"Page number must be >= 1, got {page_num}")
        
        if page_num > total_pages:
            raise ValidationError(
                f"Page number {page_num} exceeds total pages {total_pages}"
            )
    
    @classmethod
    def validate_partition_id(cls, partition_id: int, total_partitions: int) -> None:
        """
        Validate partition ID
        
        Args:
            partition_id: Partition ID to validate
            total_partitions: Total number of partitions
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(partition_id, int):
            raise ValidationError(
                f"Partition ID must be integer, got {type(partition_id)}"
            )
        
        if partition_id < 1:
            raise ValidationError(f"Partition ID must be >= 1, got {partition_id}")
        
        if partition_id > total_partitions:
            raise ValidationError(
                f"Partition ID {partition_id} exceeds total partitions {total_partitions}"
            )


class APIValidator:
    """Validate API inputs"""
    
    @classmethod
    def validate_api_key(cls, api_key: Optional[str]) -> None:
        """
        Validate OpenAI API key format
        
        Args:
            api_key: API key to validate
            
        Raises:
            ValidationError: If validation fails
        """
        if not api_key:
            raise ValidationError("API key cannot be empty")
        
        if not isinstance(api_key, str):
            raise ValidationError(f"API key must be string, got {type(api_key)}")
        
        # Check minimum length
        if len(api_key) < 10:
            raise ValidationError("API key too short")
        
        # Check format (OpenAI keys typically start with sk-)
        if not api_key.startswith(("sk-", "test-")):
            logger.warning(
                "API key doesn't start with expected prefix (sk- or test-). "
                "This might be a custom endpoint."
            )
    
    @classmethod
    def validate_question(cls, question: str) -> None:
        """
        Validate user question
        
        Args:
            question: User question to validate
            
        Raises:
            ValidationError: If validation fails
        """
        if not question:
            raise ValidationError("Question cannot be empty")
        
        if not isinstance(question, str):
            raise ValidationError(f"Question must be string, got {type(question)}")
        
        # Remove whitespace
        question = question.strip()
        
        if len(question) == 0:
            raise ValidationError("Question cannot be only whitespace")
        
        # Check minimum length
        if len(question) < 3:
            raise ValidationError("Question too short (minimum 3 characters)")
        
        # Check maximum length (to prevent token overflow)
        if len(question) > 1000:
            raise ValidationError(
                f"Question too long: {len(question)} characters. "
                f"Maximum is 1000 characters"
            )
    
    @classmethod
    def validate_page_list(cls, pages: List[int], max_pages: int = 100) -> None:
        """
        Validate list of page numbers
        
        Args:
            pages: List of page numbers
            max_pages: Maximum allowed pages
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(pages, list):
            raise ValidationError(f"Pages must be list, got {type(pages)}")
        
        if len(pages) == 0:
            raise ValidationError("Page list cannot be empty")
        
        if len(pages) > max_pages:
            raise ValidationError(
                f"Too many pages: {len(pages)}. Maximum is {max_pages}"
            )
        
        # Check all are integers
        for i, page_num in enumerate(pages):
            if not isinstance(page_num, int):
                raise ValidationError(
                    f"Page number at index {i} is not integer: {type(page_num)}"
                )
            
            if page_num < 1:
                raise ValidationError(f"Page number must be >= 1, got {page_num}")


class ConfigValidator:
    """Validate configuration parameters"""
    
    @classmethod
    def validate_render_scale(cls, scale: float) -> None:
        """Validate render scale parameter"""
        if not isinstance(scale, (int, float)):
            raise ValidationError(f"Render scale must be numeric, got {type(scale)}")
        
        if scale <= 0:
            raise ValidationError(f"Render scale must be > 0, got {scale}")
        
        if scale > 5.0:
            raise ValidationError(
                f"Render scale too high: {scale}. Maximum recommended is 5.0"
            )
    
    @classmethod
    def validate_jpeg_quality(cls, quality: int) -> None:
        """Validate JPEG quality parameter"""
        if not isinstance(quality, int):
            raise ValidationError(f"JPEG quality must be integer, got {type(quality)}")
        
        if quality < 1 or quality > 100:
            raise ValidationError(
                f"JPEG quality must be 1-100, got {quality}"
            )
    
    @classmethod
    def validate_partition_size(cls, size: int) -> None:
        """Validate partition size parameter"""
        if not isinstance(size, int):
            raise ValidationError(f"Partition size must be integer, got {type(size)}")
        
        if size < 1:
            raise ValidationError(f"Partition size must be >= 1, got {size}")
        
        if size > 100:
            raise ValidationError(
                f"Partition size too large: {size}. Maximum recommended is 100"
            )
    
    @classmethod
    def validate_max_tokens(cls, tokens: int) -> None:
        """Validate max tokens parameter"""
        if not isinstance(tokens, int):
            raise ValidationError(f"Max tokens must be integer, got {type(tokens)}")
        
        if tokens < 1:
            raise ValidationError(f"Max tokens must be >= 1, got {tokens}")
        
        if tokens > 16000:
            logger.warning(
                f"Max tokens very high: {tokens}. This may cause long processing times."
            )
    
    @classmethod
    def validate_temperature(cls, temp: float) -> None:
        """Validate temperature parameter"""
        if not isinstance(temp, (int, float)):
            raise ValidationError(f"Temperature must be numeric, got {type(temp)}")
        
        if temp < 0 or temp > 2.0:
            raise ValidationError(
                f"Temperature must be 0-2.0, got {temp}"
            )
