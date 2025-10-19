"""
PDF document processor using PyMuPDF (fitz).
"""

import logging
from pathlib import Path
from typing import Optional
import io

try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError("PyMuPDF is required. Install with: pip install pymupdf")

from PIL import Image

from processors.base import BaseProcessor
from models.document import Document, Page
from storage.base import BaseStorage
from exceptions import ProcessingError
from core.config import get_settings

logger = logging.getLogger(__name__)


class PDFProcessor(BaseProcessor):
    """
    PDF processor using PyMuPDF to convert pages to images.
    """

    def __init__(self, dpi: Optional[int] = None, image_quality: Optional[int] = None):
        """
        Initialize PDF processor.

        Args:
            dpi: DPI for image conversion (defaults to config value).
            image_quality: JPEG quality 1-100 (defaults to config value).
        """
        settings = get_settings()
        self.dpi = dpi if dpi is not None else settings.IMAGE_DPI
        self.image_quality = image_quality if image_quality is not None else settings.IMAGE_QUALITY
        logger.info(f"Initialized PDFProcessor (dpi={self.dpi}, quality={self.image_quality})")

    def supports(self, file_path: str) -> bool:
        """
        Check if file is a PDF.

        Args:
            file_path: Path to the file.

        Returns:
            bool: True if file has .pdf extension.
        """
        return Path(file_path).suffix.lower() == '.pdf'

    def process(self, file_path: str, doc_id: str, storage: BaseStorage) -> Document:
        """
        Process a PDF file.

        Converts each page to a JPEG image and stores it.

        Args:
            file_path: Path to the PDF file.
            doc_id: Unique document identifier.
            storage: Storage instance to save images.

        Returns:
            Document: Document object with status="processing".

        Raises:
            ProcessingError: If processing fails.
        """
        file_path_obj = Path(file_path)

        if not file_path_obj.exists():
            raise ProcessingError(
                f"File not found: {file_path}",
                doc_id=doc_id
            )

        if not self.supports(file_path):
            raise ProcessingError(
                f"Unsupported file type: {file_path_obj.suffix}",
                doc_id=doc_id
            )

        try:
            logger.info(f"Processing PDF: {file_path} (doc_id={doc_id})")

            # Open PDF
            pdf_document = fitz.open(file_path)
            page_count = len(pdf_document)

            logger.debug(f"PDF has {page_count} pages")

            # Create Document object
            document = Document(
                id=doc_id,
                name=file_path_obj.name,
                page_count=page_count,
                status="processing",
                pages=[]
            )

            # Process each page
            for page_num in range(1, page_count + 1):
                try:
                    page = self._process_page(
                        pdf_document=pdf_document,
                        page_num=page_num,
                        doc_id=doc_id,
                        storage=storage
                    )
                    document.add_page(page)
                    logger.debug(f"Processed page {page_num}/{page_count}")

                except Exception as e:
                    pdf_document.close()
                    raise ProcessingError(
                        f"Failed to process page {page_num}: {e}",
                        doc_id=doc_id,
                        details=str(e)
                    )

            # Close PDF
            pdf_document.close()

            logger.info(f"Successfully processed PDF: {doc_id} ({page_count} pages)")
            return document

        except ProcessingError:
            raise
        except Exception as e:
            raise ProcessingError(
                f"Failed to process PDF: {e}",
                doc_id=doc_id,
                details=str(e)
            )

    def _process_page(
        self,
        pdf_document: fitz.Document,
        page_num: int,
        doc_id: str,
        storage: BaseStorage
    ) -> Page:
        """
        Process a single PDF page.

        Args:
            pdf_document: Open PyMuPDF document.
            page_num: Page number (1-indexed).
            doc_id: Document identifier.
            storage: Storage instance.

        Returns:
            Page: Page object with image saved to storage.

        Raises:
            Exception: If processing fails.
        """
        # Get page (PyMuPDF uses 0-indexed pages)
        pdf_page = pdf_document[page_num - 1]

        # Calculate zoom factor for desired DPI
        # PDF default is 72 DPI
        zoom = self.dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        # Render page to pixmap
        pixmap = pdf_page.get_pixmap(matrix=matrix)

        # Convert to PIL Image
        img_data = pixmap.tobytes("jpeg")
        image = Image.open(io.BytesIO(img_data))

        # Get dimensions
        width, height = image.size

        # Convert to JPEG bytes with specified quality
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='JPEG', quality=self.image_quality)
        image_bytes = img_buffer.getvalue()

        # Save to storage
        image_path = storage.save_page_image(
            doc_id=doc_id,
            page_num=page_num,
            image_bytes=image_bytes
        )

        # Create Page object
        page = Page(
            page_number=page_num,
            image_path=image_path,
            width=width,
            height=height
        )

        return page

    def __repr__(self) -> str:
        return f"PDFProcessor(dpi={self.dpi}, quality={self.image_quality})"
