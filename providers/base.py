"""
Abstract base class for vision AI providers.
"""

from abc import ABC, abstractmethod
from typing import List


class BaseProvider(ABC):
    """
    Abstract base class for vision AI providers.

    Defines the interface for interacting with vision models.
    """

    @abstractmethod
    def analyze_image(self, image_path: str, prompt: str) -> str:
        """
        Analyze a single image with a prompt.

        Args:
            image_path: Path to the image file.
            prompt: Text prompt for the vision model.

        Returns:
            str: Model's response.

        Raises:
            VisionModelError: If the API call fails.
        """
        pass

    @abstractmethod
    def analyze_images(self, image_paths: List[str], prompt: str) -> str:
        """
        Analyze multiple images with a prompt.

        Args:
            image_paths: List of paths to image files.
            prompt: Text prompt for the vision model.

        Returns:
            str: Model's response.

        Raises:
            VisionModelError: If the API call fails.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
