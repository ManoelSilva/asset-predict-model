import os
import logging
from typing import Optional

from .data_storage_handler import DataStorageHandler
from .s3_data_storage_service import S3DataStorageService
from .system_data_storage_service import SystemDataStorageService


class DataStorageService:
    """
    Factory and coordinator for data storage services.
    Determines which storage backend to use based on environment variables.
    """

    def __init__(self, environment: str = None, s3_bucket: str = None, base_path: str = None):
        """
        Initialize the data storage service.
        
        Args:
            environment: Storage environment ('AWS' for S3, 'SYSTEM' or None for local)
            s3_bucket: S3 bucket name (optional, can be set via S3_BUCKET_NAME env var)
            base_path: Base path for local storage (optional, defaults to current directory)
        """
        self.environment = environment or os.getenv('enviroment', 'SYSTEM').upper()
        self.s3_bucket = s3_bucket
        self.base_path = base_path
        self._storage_handler: Optional[DataStorageHandler] = None
        
        logging.info(f"DataStorageService initialized with environment: {self.environment}")

    def get_storage_handler(self) -> DataStorageHandler:
        """
        Get the appropriate storage handler based on environment configuration.
        
        Returns:
            DataStorageHandler: Configured storage handler
        """
        if self._storage_handler is None:
            if self.environment == 'AWS':
                self._storage_handler = S3DataStorageService(
                    bucket_name=self.s3_bucket
                )
                logging.info("Using S3 storage handler")
            else:  # SYSTEM or any other value
                self._storage_handler = SystemDataStorageService(
                    base_path=self.base_path
                )
                logging.info("Using local file system storage handler")
        
        return self._storage_handler

    def save_file(self, file_path: str, data, content_type: str = None) -> str:
        """
        Save a file using the appropriate storage handler.
        
        Args:
            file_path: Path where the file should be saved
            data: Data to save
            content_type: MIME type of the content (optional)
            
        Returns:
            str: Path/URL where the file was saved
        """
        handler = self.get_storage_handler()
        return handler.save_file(file_path, data, content_type)

    def load_file(self, file_path: str):
        """
        Load a file using the appropriate storage handler.
        
        Args:
            file_path: Path to the file to load
            
        Returns:
            File content (bytes)
        """
        handler = self.get_storage_handler()
        return handler.load_file(file_path)

    def file_exists(self, file_path: str) -> bool:
        """
        Check if a file exists using the appropriate storage handler.
        
        Args:
            file_path: Path to check
            
        Returns:
            bool: True if file exists, False otherwise
        """
        handler = self.get_storage_handler()
        return handler.file_exists(file_path)

    def delete_file(self, file_path: str) -> bool:
        """
        Delete a file using the appropriate storage handler.
        
        Args:
            file_path: Path to the file to delete
            
        Returns:
            bool: True if file was deleted successfully, False otherwise
        """
        handler = self.get_storage_handler()
        return handler.delete_file(file_path)

    def create_directory(self, dir_path: str) -> bool:
        """
        Create a directory using the appropriate storage handler.
        
        Args:
            dir_path: Path of the directory to create
            
        Returns:
            bool: True if directory was created successfully, False otherwise
        """
        handler = self.get_storage_handler()
        return handler.create_directory(dir_path)

    def get_file_url(self, file_path: str) -> str:
        """
        Get the URL or full path to access a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            str: URL or full path to access the file
        """
        handler = self.get_storage_handler()
        return handler.get_file_url(file_path)

    def is_s3_storage(self) -> bool:
        """
        Check if the current storage is S3.
        
        Returns:
            bool: True if using S3 storage, False otherwise
        """
        return self.environment == 'AWS'

    def is_local_storage(self) -> bool:
        """
        Check if the current storage is local file system.
        
        Returns:
            bool: True if using local storage, False otherwise
        """
        return self.environment != 'AWS'
