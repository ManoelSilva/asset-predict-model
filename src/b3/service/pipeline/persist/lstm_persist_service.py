import logging
import os
import tempfile
from io import BytesIO

import torch
from asset_model_data_storage.data_storage_service import DataStorageService

from b3.service.pipeline.persist.base_persist_service import BasePersistService


class LSTMPersistService(BasePersistService):
    """
    Service responsible for saving and loading trained LSTM models (PyTorch).
    """

    DEFAULT_MODEL_NAME = "b3_lstm_mtl.pt"

    def __init__(self, storage_service: DataStorageService = None):
        """
        Initialize the LSTM model saving service.
        
        Args:
            storage_service: Data storage service instance (optional, creates default if not provided)
        """
        super().__init__(storage_service)

    def save_model(self, model_net, model_dir: str = "models", model_name: str = None) -> str:
        """
        Save a trained PyTorch model using the configured storage service.
        
        Args:
            model_net: Trained PyTorch LSTMNet model
            model_dir: Directory to save the model
            model_name: Name of the model file
            
        Returns:
            str: Path/URL to the saved model file
        """
        model_name = model_name or self.DEFAULT_MODEL_NAME
        logging.info(f"Saving PyTorch model to {model_dir}...")
        model_path = os.path.join(model_dir, model_name).replace('\\', '/')

        if self._storage_service.is_local_storage():
            self._storage_service.create_directory(model_dir)

        # Create a temp file
        fd, tmp_path = tempfile.mkstemp(suffix=".pt")
        os.close(fd)

        try:
            # Save state dict and architecture config
            checkpoint = {
                'state_dict': model_net.state_dict(),
                'input_features': model_net.lstm.input_size,
                'hidden_size': model_net.lstm.hidden_size,
                'dropout': model_net.dropout.p
            }
            torch.save(checkpoint, tmp_path)

            with open(tmp_path, "rb") as f:
                model_bytes = BytesIO(f.read())

            saved_path = self._storage_service.save_file(model_path, model_bytes, 'application/octet-stream')
            logging.info(f"PyTorch model saved: {saved_path}")
            return saved_path
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def load_model(self, model_path: str):
        """
        Load a saved PyTorch model checkpoint from storage.
        
        Args:
            model_path: Path to the saved model file
            
        Returns:
            dict: Checkpoint dictionary containing state_dict and config
        """
        logging.info(f"Loading PyTorch model from {model_path}...")
        if not self._storage_service.file_exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        data = self._storage_service.load_file(model_path)

        # Create a temp file
        fd, tmp_path = tempfile.mkstemp(suffix=".pt")
        os.close(fd)

        try:
            with open(tmp_path, "wb") as f:
                f.write(data)

            # Load with weights_only=False if needed, but safest default is usually fine for dicts if trusted
            # Using map_location='cpu' to avoid CUDA errors if loading on CPU machine
            checkpoint = torch.load(tmp_path, map_location=torch.device('cpu'))
            logging.info("PyTorch model checkpoint loaded successfully")
            return checkpoint
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception as e:
                    logging.warning(f"Failed to remove temp file {tmp_path}: {e}")
