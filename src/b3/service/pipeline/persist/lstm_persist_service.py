import logging
import os
import tempfile
from io import BytesIO

from asset_model_data_storage.data_storage_service import DataStorageService
from tensorflow.keras.models import load_model

DEFAULT_MODEL_NAME = "b3_lstm_mtl.keras"


class LSTMPersistService:
    """
    Service responsible for saving and loading trained LSTM models.
    """

    def __init__(self, storage_service: DataStorageService = None):
        """
        Initialize the LSTM model saving service.
        
        Args:
            storage_service: Data storage service instance (optional, creates default if not provided)
        """
        self.storage_service = storage_service or DataStorageService()

    def save_model(self, keras_model, model_dir: str = "models", model_name: str = DEFAULT_MODEL_NAME) -> str:
        """
        Save a trained Keras model using the configured storage service.
        
        Args:
            keras_model: Trained Keras model to save
            model_dir: Directory to save the model
            model_name: Name of the model file
            
        Returns:
            str: Path/URL to the saved model file
        """
        logging.info(f"Saving Keras pipeline to {model_dir}...")
        model_path = os.path.join(model_dir, model_name).replace('\\', '/')

        if self.storage_service.is_local_storage():
            self.storage_service.create_directory(model_dir)

        with tempfile.NamedTemporaryFile(suffix=".keras", delete=True) as tmp:
            keras_model.save(tmp.name)
            with open(tmp.name, "rb") as f:
                model_bytes = BytesIO(f.read())

        saved_path = self.storage_service.save_file(model_path, model_bytes, 'application/octet-stream')
        logging.info(f"Keras pipeline saved: {saved_path}")
        return saved_path

    def load_model(self, model_path: str):
        """
        Load a saved Keras model using the configured storage service.
        
        Args:
            model_path: Path to the saved model file
            
        Returns:
            Loaded Keras model
        """
        logging.info(f"Loading Keras pipeline from {model_path}...")
        if not self.storage_service.file_exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        data = self.storage_service.load_file(model_path)
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=True) as tmp:
            with open(tmp.name, "wb") as f:
                f.write(data)
            model = load_model(tmp.name)
        logging.info("Keras pipeline loaded successfully")
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

    @staticmethod
    def get_model_relative_path(model_dir: str = "models", model_name: str = DEFAULT_MODEL_NAME) -> str:
        """
        Get the relative path to a model file for storage operations.

        Args:
            model_dir: Directory where the model is located
            model_name: Name of the model file

        Returns:
            str: Relative path to the model file for storage operations
        """
        return os.path.join(model_dir, model_name).replace('\\', '/')
