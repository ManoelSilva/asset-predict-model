import os
import logging
from typing import Union, BinaryIO

from .data_storage_handler import DataStorageHandler


class SystemDataStorageService(DataStorageHandler):
    """
    Local file system implementation of DataStorageHandler.
    """

    def __init__(self, base_path: str = None):
        """
        Initialize system storage service.
        
        Args:
            base_path: Base directory for file operations (defaults to current working directory)
        """
        self.base_path = base_path or os.getcwd()
        logging.info(f"System storage service initialized with base path: {self.base_path}")

    def save_file(self, file_path: str, data: Union[bytes, BinaryIO], 
                  content_type: str = None) -> str:
        """
        Save data to local file system.
        
        Args:
            file_path: Path where the file should be saved
            data: Data to save (bytes or file-like object)
            content_type: MIME type of the content (optional, not used for local storage)
            
        Returns:
            str: Full path where the file was saved
        """
        try:
            # Create full path
            full_path = os.path.join(self.base_path, file_path)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            # Convert file-like object to bytes if needed
            if hasattr(data, 'read'):
                data = data.read()
            
            # Write data to file
            with open(full_path, 'wb') as f:
                f.write(data)
            
            logging.info(f"File saved to local storage: {full_path}")
            return full_path
            
        except Exception as e:
            logging.error(f"Error saving file to local storage: {e}")
            raise

    def load_file(self, file_path: str) -> bytes:
        """
        Load data from local file system.
        
        Args:
            file_path: Path to the file to load
            
        Returns:
            bytes: The file content
        """
        try:
            full_path = os.path.join(self.base_path, file_path)
            
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"File not found: {full_path}")
            
            with open(full_path, 'rb') as f:
                data = f.read()
            
            logging.info(f"File loaded from local storage: {full_path}")
            return data
            
        except Exception as e:
            logging.error(f"Error loading file from local storage: {e}")
            raise

    def file_exists(self, file_path: str) -> bool:
        """
        Check if a file exists in local file system.
        
        Args:
            file_path: Path to check
            
        Returns:
            bool: True if file exists, False otherwise
        """
        full_path = os.path.join(self.base_path, file_path)
        return os.path.exists(full_path)

    def delete_file(self, file_path: str) -> bool:
        """
        Delete a file from local file system.
        
        Args:
            file_path: Path to the file to delete
            
        Returns:
            bool: True if file was deleted successfully, False otherwise
        """
        try:
            full_path = os.path.join(self.base_path, file_path)
            
            if os.path.exists(full_path):
                os.remove(full_path)
                logging.info(f"File deleted from local storage: {full_path}")
                return True
            else:
                logging.warning(f"File not found for deletion: {full_path}")
                return False
                
        except Exception as e:
            logging.error(f"Error deleting file from local storage: {e}")
            return False

    def create_directory(self, dir_path: str) -> bool:
        """
        Create a directory in local file system.
        
        Args:
            dir_path: Path of the directory to create
            
        Returns:
            bool: True if directory was created successfully, False otherwise
        """
        try:
            full_path = os.path.join(self.base_path, dir_path)
            os.makedirs(full_path, exist_ok=True)
            logging.info(f"Directory created in local storage: {full_path}")
            return True
        except Exception as e:
            logging.error(f"Error creating directory in local storage: {e}")
            return False

    def get_file_url(self, file_path: str) -> str:
        """
        Get the full file path for local access.
        
        Args:
            file_path: Path to the file
            
        Returns:
            str: Full path to access the file
        """
        return os.path.join(self.base_path, file_path)
