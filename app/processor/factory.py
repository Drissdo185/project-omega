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
                render_scale=config.get('render_scale'),
                jpeg_quality=config.get('jpeg_quality'),
                max_image_size=config.get('max_image_size')
            )
        
        raise ValueError(f"No vision processor for file: {file_path}")