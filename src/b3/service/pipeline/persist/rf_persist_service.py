import logging
import os
from io import BytesIO

import joblib
from asset_model_data_storage.data_storage_service import DataStorageService
from sklearn.ensemble import RandomForestClassifier

from b3.service.pipeline.persist.base_persist_service import BasePersistService


class RandomForestPersistService(BasePersistService):
    """
    Service responsible for saving and loading trained B3 Random Forest models.
    """

    DEFAULT_MODEL_NAME = "b3_model.joblib"

    def __init__(self, storage_service: DataStorageService = None):
        """
        Initialize the model saving service.
        
        Args:
            storage_service: Data storage service instance (optional, creates default if not provided)
        """
        super().__init__(storage_service)

    def save_model(self, model: RandomForestClassifier, model_dir: str = "models",
                   model_name: str = None) -> str:
        """
        Save a trained model using the configured storage service.
        
        Args:
            model: Trained model to save
            model_dir: Directory to save the model
            model_name: Name of the model file
            
        Returns:
            str: Path/URL to the saved model file
        """
        model_name = model_name or self.DEFAULT_MODEL_NAME
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

    def load_model(self, model_path='models') -> RandomForestClassifier:
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
