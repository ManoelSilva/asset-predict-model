import logging
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from io import BytesIO

from b3.service.data.storage.data_storage_service import DataStorageService

# Constants
DEFAULT_MODEL_NAME = "b3_model.joblib"


class B3ModelSavingService:
    """
    Service responsible for saving and loading trained B3 models.
    """

    def __init__(self, storage_service: DataStorageService = None):
        """
        Initialize the model saving service.
        
        Args:
            storage_service: Data storage service instance (optional, creates default if not provided)
        """
        self.storage_service = storage_service or DataStorageService()

    def save_model(self, model: RandomForestClassifier, model_dir: str = "models",
                   model_name: str = DEFAULT_MODEL_NAME) -> str:
        """
        Save a trained model using the configured storage service.
        
        Args:
            model: Trained model to save
            model_dir: Directory to save the model
            model_name: Name of the model file
            
        Returns:
            str: Path/URL to the saved model file
        """
        logging.info(f"Saving model to {model_dir}...")

        # Create model path
        model_path = os.path.join(model_dir, model_name).replace('\\', '/')
        
        # Create directory if using local storage
        if self.storage_service.is_local_storage():
            self.storage_service.create_directory(model_dir)

        # Save model to bytes
        model_bytes = BytesIO()
        joblib.dump(model, model_bytes)
        model_bytes.seek(0)

        # Save using storage service
        saved_path = self.storage_service.save_file(model_path, model_bytes, 'application/octet-stream')
        logging.info(f"Model saved: {saved_path}")

        return saved_path

    def load_model(self, model_path = 'models') -> RandomForestClassifier:
        """
        Load a saved model using the configured storage service.
        
        Args:
            model_path: Path to the saved model file
            
        Returns:
            RandomForestClassifier: Loaded model
        """
        logging.info(f"Loading model from {model_path}...")

        if not self.storage_service.file_exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load model data from storage
        model_data = self.storage_service.load_file(model_path)
        
        # Load model from bytes
        model_bytes = BytesIO(model_data)
        model = joblib.load(model_bytes)
        logging.info("Model loaded successfully")

        return model

    def model_exists(self, model_dir: str = "models", model_name: str = DEFAULT_MODEL_NAME) -> bool:
        """
        Check if a model file exists using the configured storage service.
        
        Args:
            model_dir: Directory where the model should be
            model_name: Name of the model file
            
        Returns:
            bool: True if model exists, False otherwise
        """
        model_path = os.path.join(model_dir, model_name).replace('\\', '/')
        return self.storage_service.file_exists(model_path)

    def get_model_path(self, model_dir: str = "models", model_name: str = DEFAULT_MODEL_NAME) -> str:
        """
        Get the full path/URL to a model file using the configured storage service.
        
        Args:
            model_dir: Directory where the model is located
            model_name: Name of the model file
            
        Returns:
            str: Full path/URL to the model file
        """
        model_path = os.path.join(model_dir, model_name).replace('\\', '/')
        return self.storage_service.get_file_url(model_path)

    def get_model_relative_path(self, model_dir: str = "models", model_name: str = DEFAULT_MODEL_NAME) -> str:
        """
        Get the relative path to a model file for storage operations.
        
        Args:
            model_dir: Directory where the model is located
            model_name: Name of the model file
            
        Returns:
            str: Relative path to the model file for storage operations
        """
        return os.path.join(model_dir, model_name).replace('\\', '/')
