from abc import ABC, abstractmethod
from typing import Union, BinaryIO
import os


class DataStorageHandler(ABC):
    """
    Abstract base class for data storage handlers.
    Defines the interface for saving and loading files from different storage backends.
    """

    @abstractmethod
    def save_file(self, file_path: str, data: Union[bytes, BinaryIO], 
                  content_type: str = None) -> str:
        """
        Save data to storage.
        
        Args:
            file_path: Path where the file should be saved
            data: Data to save (bytes or file-like object)
            content_type: MIME type of the content (optional)
            
        Returns:
            str: The path/URL where the file was saved
        """
        pass

    @abstractmethod
    def load_file(self, file_path: str) -> bytes:
        """
        Load data from storage.
        
        Args:
            file_path: Path to the file to load
            
        Returns:
            bytes: The file content
        """
        pass

    @abstractmethod
    def file_exists(self, file_path: str) -> bool:
        """
        Check if a file exists in storage.
        
        Args:
            file_path: Path to check
            
        Returns:
            bool: True if file exists, False otherwise
        """
        pass

    @abstractmethod
    def delete_file(self, file_path: str) -> bool:
        """
        Delete a file from storage.
        
        Args:
            file_path: Path to the file to delete
            
        Returns:
            bool: True if file was deleted successfully, False otherwise
        """
        pass

    @abstractmethod
    def create_directory(self, dir_path: str) -> bool:
        """
        Create a directory in storage.
        
        Args:
            dir_path: Path of the directory to create
            
        Returns:
            bool: True if directory was created successfully, False otherwise
        """
        pass

    @abstractmethod
    def get_file_url(self, file_path: str) -> str:
        """
        Get the URL or full path to access a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            str: URL or full path to access the file
        """
        pass

    def _normalize_path(self, file_path: str) -> str:
        """
        Normalize file path by removing leading slashes and ensuring consistent separators.
        
        Args:
            file_path: Raw file path
            
        Returns:
            str: Normalized file path
        """
        # Remove leading slashes and normalize path separators
        normalized = file_path.lstrip('/').replace('\\', '/')
        return normalized
