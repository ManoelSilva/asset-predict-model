import logging
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Optional


class LSTMScaler:
    """
    Handles scaling of LSTM input data (3D arrays).
    Wrapper around StandardScaler to handle reshaping of 3D sequences (N, T, F).
    """

    def __init__(self, scaler: Optional[StandardScaler] = None):
        self._scaler = scaler or StandardScaler()

    @property
    def scaler(self) -> StandardScaler:
        return self._scaler

    def fit(self, X: np.ndarray) -> None:
        """
        Fit scaler on 3D input data.
        
        Args:
            X: Input data of shape (N, T, F)
        """
        if X.ndim != 3:
            raise ValueError(f"Expected 3D input (N, T, F), got shape {X.shape}")

        N, T, F = X.shape
        X_reshaped = X.reshape(N * T, F)
        self._scaler.fit(X_reshaped)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform 3D input data using fitted scaler.
        
        Args:
            X: Input data of shape (N, T, F)
            
        Returns:
            Scaled data of shape (N, T, F)
        """
        if X.ndim != 3:
            raise ValueError(f"Expected 3D input (N, T, F), got shape {X.shape}")

        if not self.is_fitted():
            logging.warning("Scaler initialized but not fitted, using raw input")
            return X

        N, T, F = X.shape
        X_reshaped = X.reshape(N * T, F)
        X_scaled = self._scaler.transform(X_reshaped)
        return X_scaled.reshape(N, T, F)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit to data, then transform it.
        
        Args:
            X: Input data of shape (N, T, F)
            
        Returns:
            Scaled data of shape (N, T, F)
        """
        self.fit(X)
        return self.transform(X)

    def is_fitted(self) -> bool:
        """Check if scaler is fitted."""
        try:
            # Check for mean_ attribute which is set after fitting
            self._scaler.mean_
            return True
        except AttributeError:
            return False

    def reset(self) -> None:
        """Reset the internal scaler."""
        self._scaler = StandardScaler()
