from dataclasses import dataclass
from typing import Any, Optional
import numpy as np
import pandas as pd


@dataclass
class SplitDataParams:
    """
    Parameter object for split_data method.
    Encapsulates all parameters needed for data splitting.
    """
    X: Any
    y: Any
    test_size: float = 0.2
    val_size: float = 0.2

    # LSTM-specific parameters (optional)
    yR: Optional[np.ndarray] = None
    p0: Optional[np.ndarray] = None
    pf: Optional[np.ndarray] = None

    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.test_size <= 0 or self.test_size >= 1:
            raise ValueError(f"test_size must be between 0 and 1, got {self.test_size}")
        if self.val_size <= 0 or self.val_size >= 1:
            raise ValueError(f"val_size must be between 0 and 1, got {self.val_size}")

    def has_lstm_params(self) -> bool:
        """Check if this parameter object contains LSTM-specific parameters."""
        return self.yR is not None or self.p0 is not None or self.pf is not None


@dataclass
class TrainModelParams:
    """
    Parameter object for train_model method.
    Encapsulates all parameters needed for model training.
    """
    X_train: Any
    y_train: Any

    # LSTM-specific parameters (optional)
    yR_train: Optional[np.ndarray] = None

    # Validation data (optional)
    X_val: Optional[Any] = None
    y_val: Optional[Any] = None
    yR_val: Optional[np.ndarray] = None

    def has_lstm_params(self) -> bool:
        """Check if this parameter object contains LSTM-specific parameters."""
        return self.yR_train is not None


@dataclass
class PrepareDataParams:
    """
    Parameter object for prepare_data method.
    Encapsulates all parameters needed for data preparation.
    """
    X: pd.DataFrame
    y: pd.Series

    # LSTM-specific parameters (optional)
    df_processed: Optional[pd.DataFrame] = None

    def has_lstm_params(self) -> bool:
        """Check if this parameter object contains LSTM-specific parameters."""
        return self.df_processed is not None
