from typing import Tuple

import numpy as np
from pandas import Series
from sklearn.model_selection import train_test_split

from constants import RANDOM_STATE


class LSTMDataSplitter:
    """Handles splitting of LSTM sequences into train/validation/test sets."""

    @staticmethod
    def split_sequences(X: np.ndarray, y_action: Series, y_return: np.ndarray,
                        last_price: np.ndarray, future_price: np.ndarray,
                        test_size: float, val_size: float) -> Tuple[np.ndarray, ...]:
        """
        Split sequences into train/validation/test sets.
        
        Args:
            X: Feature sequences
            y_action: Action target sequences
            y_return: Return target sequences
            last_price: Initial price sequences
            future_price: Future price sequences
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            
        Returns:
            Tuple containing all train/val/test splits
        """
        y_norm = y_action.astype(str)
        X_train, X_temp, yA_train, yA_temp, yR_train, yR_temp, p0_train, p0_temp, pf_train, pf_temp = train_test_split(
            X, y_norm, y_return, last_price, future_price,
            test_size=test_size, random_state=RANDOM_STATE, stratify=y_norm
        )
        X_val, X_test, yA_val, yA_test, yR_val, yR_test, p0_val, p0_test, pf_val, pf_test = train_test_split(
            X_temp, yA_temp, yR_temp, p0_temp, pf_temp,
            test_size=val_size, random_state=RANDOM_STATE, stratify=yA_temp
        )
        return (X_train, X_val, X_test,
                yA_train, yA_val, yA_test,
                yR_train, yR_val, yR_test,
                p0_train, p0_val, p0_test,
                pf_train, pf_val, pf_test)
