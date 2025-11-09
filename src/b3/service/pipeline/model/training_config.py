"""
Configuration dataclass for model training parameters.
Centralizes training parameters to simplify method signatures.
"""
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """
    Configuration for model training.
    Provides default values for all training parameters.
    """
    model_type: str = "rf"
    model_dir: str = "models"
    n_jobs: int = 5
    test_size: float = 0.2
    val_size: float = 0.2

    # LSTM-specific parameters
    lookback: int = 32
    horizon: int = 1
    epochs: int = 25
    batch_size: int = 128
    learning_rate: float = 1e-3
    units: int = 96
    dropout: float = 0.2
    price_col: str = "close"

    def get_lstm_config_dict(self) -> dict:
        """
        Get dictionary of LSTM-specific configuration parameters.
        
        Returns:
            Dictionary containing LSTM configuration parameters
        """
        return {
            'lookback': self.lookback,
            'horizon': self.horizon,
            'units': self.units,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'price_col': self.price_col
        }

    def get_rf_config_dict(self) -> dict:
        """
        Get dictionary of Random Forest-specific configuration parameters.
        
        Returns:
            Dictionary containing RF configuration parameters
        """
        return {
            'n_jobs': self.n_jobs
        }

    def get_all_config_dict(self) -> dict:
        """
        Get dictionary of all configuration parameters.
        
        Returns:
            Dictionary containing all configuration parameters
        """
        result = {
            'n_jobs': self.n_jobs,
            'lookback': self.lookback,
            'horizon': self.horizon,
            'units': self.units,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'price_col': self.price_col
        }
        return result
