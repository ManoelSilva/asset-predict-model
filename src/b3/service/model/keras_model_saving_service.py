import logging
import os
import tempfile
from io import BytesIO

from asset_model_data_storage.data_storage_service import DataStorageService
from tensorflow.keras.models import load_model


DEFAULT_MODEL_NAME = "b3_lstm_mtl.keras"


class B3KerasModelSavingService:
    def __init__(self, storage_service: DataStorageService = None):
        self.storage_service = storage_service or DataStorageService()

    def save_model(self, keras_model, model_dir: str = "models", model_name: str = DEFAULT_MODEL_NAME) -> str:
        logging.info(f"Saving Keras model to {model_dir}...")
        model_path = os.path.join(model_dir, model_name).replace('\\', '/')

        if self.storage_service.is_local_storage():
            self.storage_service.create_directory(model_dir)

        with tempfile.NamedTemporaryFile(suffix=".keras", delete=True) as tmp:
            keras_model.save(tmp.name)
            with open(tmp.name, "rb") as f:
                model_bytes = BytesIO(f.read())

        saved_path = self.storage_service.save_file(model_path, model_bytes, 'application/octet-stream')
        logging.info(f"Keras model saved: {saved_path}")
        return saved_path

    def load_model(self, model_path: str):
        logging.info(f"Loading Keras model from {model_path}...")
        if not self.storage_service.file_exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        data = self.storage_service.load_file(model_path)
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=True) as tmp:
            with open(tmp.name, "wb") as f:
                f.write(data)
            model = load_model(tmp.name)
        logging.info("Keras model loaded successfully")
        return model

    def get_model_path(self, model_dir: str = "models", model_name: str = DEFAULT_MODEL_NAME) -> str:
        model_path = os.path.join(model_dir, model_name).replace('\\', '/')
        return self.storage_service.get_file_url(model_path)


