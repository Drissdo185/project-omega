# app/processors/factory.py
from .base import BaseProcessor
from .pdf_vision import VisionPDFProcessor

class ProcessorFactory:
    """Factory to create appropriate processor"""
    
    @staticmethod
    def create_processor(file_path: str, **config) -> BaseProcessor:
        """Create processor based on file type"""
        
        if file_path.lower().endswith('.pdf'):
            return VisionPDFProcessor(
                render_scale=config.get('render_scale', 2.0),
                jpeg_quality=config.get('jpeg_quality', 90),
                max_image_size=config.get('max_image_size', (1400, 1400))
            )
        
        raise ValueError(f"No vision processor for file: {file_path}")