import logging
import os
from abc import ABC, abstractmethod

from asset_model_data_storage.data_storage_service import DataStorageService


class BasePersistService(ABC):
    """
    Abstract base class for model persistence services.
    Provides common functionality for saving and loading models.
    """

    # Subclasses must define their own DEFAULT_MODEL_NAME
    LSTM_MODEL_FILE: str = None

    def __init__(self, storage_service: DataStorageService = None):
        """
        Initialize the model persistence service.
        
        Args:
            storage_service: Data storage service instance (optional, creates default if not provided)
        """
        self._storage_service = storage_service or DataStorageService()

    @abstractmethod
    def save_model(self, model, model_dir: str = "models", model_name: str = None) -> str:
        """
        Save a trained model using the configured storage service.
        
        Args:
            model: Trained model to save
            model_dir: Directory to save the model
            model_name: Name of the model file (uses DEFAULT_MODEL_NAME if None)
            
        Returns:
            str: Path/URL to the saved model file
        """
        pass

    @abstractmethod
    def load_model(self, model_path: str):
        """
        Load a saved model using the configured storage service.
        
        Args:
            model_path: Path to the saved model file
            
        Returns:
            Loaded model object
        """
        pass

    def model_exists(self, model_dir: str = "models", model_name: str = None) -> bool:
        """
        Check if a model file exists using the configured storage service.
        
        Args:
            model_dir: Directory where the model should be
            model_name: Name of the model file (uses DEFAULT_MODEL_NAME if None)
            
        Returns:
            bool: True if model exists, False otherwise
        """
        model_name = model_name or self.LSTM_MODEL_FILE
        model_path = os.path.join(model_dir, model_name).replace('\\', '/')
        return self._storage_service.file_exists(model_path)

    def get_model_path(self, model_dir: str = "models", model_name: str = None) -> str:
        """
        Get the full path/URL to a model file using the configured storage service.
        
        Args:
            model_dir: Directory where the model is located
            model_name: Name of the model file (uses DEFAULT_MODEL_NAME if None)
            
        Returns:
            str: Full path/URL to the model file
        """
        model_name = model_name or self.LSTM_MODEL_FILE
        model_path = os.path.join(model_dir, model_name).replace('\\', '/')
        return self._storage_service.get_file_url(model_path)

    def get_model_relative_path(self, model_dir: str = "models", model_name: str = None) -> str:
        """
        Get the relative path to a model file for storage operations.
        
        Args:
            model_dir: Directory where the model is located
            model_name: Name of the model file (uses DEFAULT_MODEL_NAME if None)
            
        Returns:
            str: Relative path to the model file for storage operations
        """
        model_name = model_name or self.LSTM_MODEL_FILE
        return os.path.join(model_dir, model_name).replace('\\', '/')

