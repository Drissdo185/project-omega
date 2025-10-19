"""
Custom exceptions for Project Omega.
"""


class DocPixieException(Exception):
    """Base exception for all Project Omega errors."""

    def __init__(self, message: str, *args, **kwargs):
        self.message = message
        super().__init__(message, *args, **kwargs)

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r})"


class DocumentNotFoundError(DocPixieException):
    """Raised when a requested document cannot be found."""

    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        message = f"Document not found: {doc_id}"
        super().__init__(message)

    def __repr__(self) -> str:
        return f"DocumentNotFoundError(doc_id={self.doc_id!r})"


class ProcessingError(DocPixieException):
    """Raised when document processing fails."""

    def __init__(self, message: str, doc_id: str = None, details: str = None):
        self.doc_id = doc_id
        self.details = details
        full_message = message
        if doc_id:
            full_message = f"[{doc_id}] {message}"
        if details:
            full_message = f"{full_message}: {details}"
        super().__init__(full_message)

    def __repr__(self) -> str:
        return f"ProcessingError(message={self.message!r}, doc_id={self.doc_id!r})"


class StorageError(DocPixieException):
    """Raised when storage operations fail."""

    def __init__(self, message: str, operation: str = None, path: str = None):
        self.operation = operation
        self.path = path
        full_message = message
        if operation:
            full_message = f"[{operation}] {message}"
        if path:
            full_message = f"{full_message} (path: {path})"
        super().__init__(full_message)

    def __repr__(self) -> str:
        return f"StorageError(message={self.message!r}, operation={self.operation!r}, path={self.path!r})"


class VisionModelError(DocPixieException):
    """Raised when vision model API calls fail."""

    def __init__(self, message: str, provider: str = None, model: str = None, status_code: int = None):
        self.provider = provider
        self.model = model
        self.status_code = status_code
        full_message = message
        if provider:
            full_message = f"[{provider}] {message}"
        if model:
            full_message = f"{full_message} (model: {model})"
        if status_code:
            full_message = f"{full_message} (status: {status_code})"
        super().__init__(full_message)

    def __repr__(self) -> str:
        return (f"VisionModelError(message={self.message!r}, provider={self.provider!r}, "
                f"model={self.model!r}, status_code={self.status_code!r})")


class ConfigurationError(DocPixieException):
    """Raised when configuration is invalid or missing."""

    def __init__(self, message: str, config_key: str = None):
        self.config_key = config_key
        full_message = message
        if config_key:
            full_message = f"Configuration error for '{config_key}': {message}"
        super().__init__(full_message)

    def __repr__(self) -> str:
        return f"ConfigurationError(message={self.message!r}, config_key={self.config_key!r})"
