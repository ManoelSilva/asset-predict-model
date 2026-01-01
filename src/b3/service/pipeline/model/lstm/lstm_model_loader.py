import logging

from b3.service.pipeline.model.lstm.lstm_config import LSTMConfig
from b3.service.pipeline.model.lstm.pytorch_mtl_model import B3PytorchMTLModel
from b3.service.pipeline.persist.lstm_persist_service import LSTMPersistService


class LSTMModelLoader:
    """Handles loading and reconstruction of LSTM models from checkpoints."""

    def __init__(self, persist_service: LSTMPersistService, default_config: LSTMConfig, device):
        """
        Initialize model loader.
        
        Args:
            persist_service: Service for loading model checkpoints
            default_config: Default LSTM configuration
            device: PyTorch device
        """
        self._persist_service = persist_service
        self._default_config = default_config
        self._device = device

    def load_model(self, model_path: str) -> B3PytorchMTLModel:
        """
        Load LSTM model from storage.
        
        Args:
            model_path: Path to saved model
            
        Returns:
            Loaded B3PytorchMTLModel
        """
        checkpoint = self._persist_service.load_model(model_path)

        # Extract config from checkpoint
        input_features = checkpoint['input_features']
        hidden_size = checkpoint['hidden_size']
        dropout = checkpoint['dropout']
        state_dict = checkpoint['state_dict']

        # Update config with loaded architecture params to ensure weights match
        model_config = LSTMConfig(
            lookback=self._default_config.lookback,
            horizon=self._default_config.horizon,
            units=hidden_size,
            dropout=dropout,
            learning_rate=self._default_config.learning_rate,
            epochs=self._default_config.epochs,
            batch_size=self._default_config.batch_size,
            loss_weight_action=self._default_config.loss_weight_action,
            loss_weight_return=self._default_config.loss_weight_return
        )

        model = B3PytorchMTLModel(
            input_timesteps=self._default_config.lookback,
            input_features=input_features,
            config=model_config,
            device=self._device
        )

        model.model.load_state_dict(state_dict)
        logging.info(f"Successfully loaded LSTM model from {model_path}")

        return model
