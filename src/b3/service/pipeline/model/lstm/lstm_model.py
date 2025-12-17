import logging
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from asset_model_data_storage.data_storage_service import DataStorageService
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split

from b3.service.pipeline.model.lstm.keras_mtl_model import B3KerasMTLModel
from b3.service.pipeline.model.lstm.lstm_config import LSTMConfig
from b3.service.pipeline.model.model import BaseModel
from b3.service.pipeline.model_evaluation_service import B3ModelEvaluationService
from b3.service.pipeline.persist.lstm_persist_service import LSTMPersistService
from constants import RANDOM_STATE


class LSTMModel(BaseModel):
    """
    LSTM Multi-Task Learning implementation for B3 asset prediction.
    """

    def __init__(self, config: Optional[LSTMConfig] = None,
                 storage_service: Optional[DataStorageService] = None):
        """
        Initialize LSTM model.
        
        Args:
            config: LSTM configuration
            storage_service: Data storage service for saving/loading models
        """
        super().__init__(config or LSTMConfig())
        self.storage_service = storage_service or DataStorageService()
        self.saving_service = LSTMPersistService(self.storage_service)
        self.evaluation_service = B3ModelEvaluationService(self.storage_service)

    def prepare_data(self, X: pd.DataFrame, y: pd.Series, df_processed: pd.DataFrame = None, **kwargs) -> Tuple[
        np.ndarray, pd.Series, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM training by building sequences.
        
        Args:
            X: Feature data
            y: Target labels
            df_processed: Full processed dataframe (required for LSTM)
            **kwargs: Additional parameters
            
        Returns:
            Tuple containing (X_seq, y_action_seq, y_return_seq, last_price_seq, future_price_seq)
        """
        if not self.validate_data(X, y):
            raise ValueError("Invalid input data for LSTM")

        if df_processed is None:
            raise ValueError("df_processed is required for LSTM sequence building")

        lookback = kwargs.get('lookback', self.config.lookback)
        horizon = kwargs.get('horizon', self.config.horizon)
        price_col = kwargs.get('price_col', self.config.price_col)

        logging.info(f"Building LSTM sequences with lookback={lookback}, horizon={horizon}")

        X_seq, yA_seq, yR_seq, p0_seq, pf_seq = self._build_sequences(
            X, y, df_processed, price_col=price_col, lookback=lookback, horizon=horizon
        )

        if len(X_seq) == 0:
            raise ValueError("No sequences built for LSTM. Check ordering, ticker column, and lookback/horizon.")

        logging.info(f"Built {len(X_seq)} sequences for LSTM training")
        return X_seq, yA_seq, yR_seq, p0_seq, pf_seq

    @staticmethod
    def _group_keys(df: DataFrame):
        if "ticker" in df.columns:
            return list(df["ticker"].astype(str).unique())
        return [None]

    @staticmethod
    def _df_for_group(df: DataFrame, group: Optional[str]) -> DataFrame:
        gdf = df if group is None else df[df["ticker"].astype(str) == group]
        for col in ["date", "datetime", "timestamp"]:
            if col in gdf.columns:
                return gdf.sort_values(col)
        return gdf

    @staticmethod
    def _build_sequences(X_df: DataFrame, y_action: Series, df_full: DataFrame,
                         price_col: str = "close", lookback: int = 32, horizon: int = 1
                         ) -> Tuple[np.ndarray, Series, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build sequences for multitask learning.

        Returns:
            X_seq: (N, lookback, n_features)
            y_action_seq: Series of action labels aligned to sequences
            y_return_seq: (N,) predicted target returns over horizon
            last_price_seq: (N,) price at prediction time (for price mapping)
            future_price_seq: (N,) actual future price at t+h (for regression eval)
        """
        import gc
        features = X_df.columns.tolist()
        n_features = len(features)

        # Pre-process targets globally
        y_processed = y_action.astype(str).str.replace("_target", "", regex=False)

        # 1. Collect Valid Group Data and Calculate Total Size
        processed_groups = []
        total_sequences = 0

        # Use unique tickers to iterate
        unique_tickers = df_full["ticker"].unique() if "ticker" in df_full.columns else [None]

        for ticker in unique_tickers:
            if ticker is None:
                gdf = df_full
            else:
                gdf = df_full[df_full["ticker"] == ticker]

            # Sort by date
            for col in ["date", "datetime", "timestamp"]:
                if col in gdf.columns:
                    gdf = gdf.sort_values(col)
                    break

            indices = gdf.index

            # Extract Data
            # Optimized: check bounds first
            if len(gdf) <= lookback + horizon:
                continue

            # Intersection check (fast fail)
            if not indices.isin(X_df.index).all():
                common = indices.intersection(X_df.index)
                if len(common) <= lookback + horizon: continue
                gdf = gdf.loc[common]
                indices = common

            # Extract arrays (small copy)
            Xg = X_df.loc[indices, features].values.astype("float32")
            yg = y_processed.loc[indices].values

            if price_col not in gdf.columns:
                raise ValueError(f"Price column '{price_col}' not found in dataframe")
            pg = gdf[price_col].astype(float).values

            n_samples = len(Xg)
            n_windows = n_samples - horizon - lookback

            if n_windows > 0:
                processed_groups.append((Xg, yg, pg, n_windows))
                total_sequences += n_windows

        if total_sequences == 0:
            return (np.zeros((0, lookback, n_features), dtype="float32"),
                    pd.Series([], dtype="object"),
                    np.zeros((0,), dtype="float32"),
                    np.zeros((0,), dtype="float32"),
                    np.zeros((0,), dtype="float32"))

        logging.info(
            f"Allocating memory for {total_sequences} sequences (approx {total_sequences * lookback * n_features * 4 / 1e9:.2f} GB)...")

        # 2. Pre-allocate arrays
        X_seq = np.zeros((total_sequences, lookback, n_features), dtype="float32")
        y_action_seq = np.empty((total_sequences,), dtype=object)
        y_return_seq = np.zeros((total_sequences,), dtype="float32")
        last_price_seq = np.zeros((total_sequences,), dtype="float32")
        future_price_seq = np.zeros((total_sequences,), dtype="float32")

        # 3. Fill Arrays
        current_idx = 0

        # Iterate and clear memory as we go
        while processed_groups:
            Xg, yg, pg, n_w = processed_groups.pop(0)

            # Vectorized Windowing
            # Shape: (n_w, n_features, lookback) -> need transpose
            windows = np.lib.stride_tricks.sliding_window_view(Xg, lookback, axis=0)
            windows = windows[:n_w]

            # Transpose to (Batch, Time, Features)
            # sliding_window_view puts window dim at the end: (Batch, Features, Time)
            # LSTM expects (Batch, Time, Features) -> swap last two axes
            windows = windows.transpose(0, 2, 1)

            X_seq[current_idx: current_idx + n_w] = windows

            # Targets
            y_action_seq[current_idx: current_idx + n_w] = yg[lookback: lookback + n_w]

            p0 = pg[lookback - 1: lookback - 1 + n_w]
            p1 = pg[lookback - 1 + horizon: lookback - 1 + horizon + n_w]

            with np.errstate(divide='ignore', invalid='ignore'):
                ret = (p1 / p0) - 1.0

            y_return_seq[current_idx: current_idx + n_w] = ret
            last_price_seq[current_idx: current_idx + n_w] = p0
            future_price_seq[current_idx: current_idx + n_w] = p1

            current_idx += n_w

            # Help GC
            del Xg, yg, pg, windows

        gc.collect()

        return X_seq, pd.Series(y_action_seq, name="target"), y_return_seq, last_price_seq, future_price_seq

    def split_data(self, X_seq: np.ndarray, yA_seq: pd.Series, yR_seq: np.ndarray,
                   p0_seq: np.ndarray, pf_seq: np.ndarray, test_size: float = 0.2, val_size: float = 0.2) -> Tuple[
        np.ndarray, ...]:
        """
        Split LSTM sequences into train/validation/test sets.
        
        Args:
            X_seq: Feature sequences
            yA_seq: Action target sequences
            yR_seq: Return target sequences
            p0_seq: Initial price sequences
            pf_seq: Future price sequences
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            
        Returns:
            Tuple containing all train/val/test splits
        """
        return self._split_sequences(
            X_seq, yA_seq, yR_seq, p0_seq, pf_seq, test_size, val_size
        )

    @staticmethod
    def _split_sequences(X: np.ndarray, y_action: Series, y_return: np.ndarray,
                         last_price: np.ndarray, future_price: np.ndarray,
                         test_size: float, val_size: float):
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

    def train_model(self, X_train: np.ndarray, yA_train: pd.Series, yR_train: np.ndarray, **kwargs) -> B3KerasMTLModel:
        """
        Train LSTM Multi-Task Learning model.
        
        Args:
            X_train: Training feature sequences
            yA_train: Training action targets
            yR_train: Training return targets
            **kwargs: Additional training parameters
            
        Returns:
            Trained B3KerasMTLModel
        """
        lookback = kwargs.get('lookback', self.config.lookback)
        n_features = kwargs.get('n_features', X_train.shape[2])

        lstm_config = LSTMConfig(
            lookback=lookback,
            horizon=self.config.horizon,
            units=self.config.units,
            dropout=self.config.dropout,
            learning_rate=self.config.learning_rate,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            loss_weight_action=self.config.loss_weight_action,
            loss_weight_return=self.config.loss_weight_return
        )

        logging.info(f"Training LSTM model with {lstm_config.epochs} epochs, batch_size={lstm_config.batch_size}")

        # Check for GPU availability and configure device
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logging.info(f"GPU detected: {len(gpus)} device(s). Training will use GPU.")
            try:
                # Currently only supported on physical GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                logging.warning(f"Could not set memory growth: {e}")
            device_name = '/GPU:0'
        else:
            logging.info("No GPU detected. Training will use CPU.")
            device_name = '/CPU:0'

        with tf.device(device_name):
            clf = B3KerasMTLModel(input_timesteps=lookback, input_features=n_features, config=lstm_config)
            clf.fit(X_train, yA_train, yR_train)

        logging.info("LSTM multi-task training completed")

        self.model = clf
        self.is_trained = True

        logging.info("LSTM training completed successfully")
        return clf

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make action predictions using trained LSTM model.
        
        Args:
            X: Feature sequences for prediction
            
        Returns:
            Array of action predictions
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")

        if not self.validate_data(X):
            raise ValueError("Invalid input data for prediction")

        return self.model.predict(X)

    def predict_return(self, X: np.ndarray) -> np.ndarray:
        """
        Make return predictions using trained LSTM model.
        
        Args:
            X: Feature sequences for prediction
            
        Returns:
            Array of return predictions
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")

        if not self.validate_data(X):
            raise ValueError("Invalid input data for prediction")

        return self.model.predict_return(X)

    def predict_price(self, X_window: pd.DataFrame, lookback: int = None, price_col: str = None) -> float:
        """
        Predict next-step price using trained LSTM model from a single asset window.
        
        Args:
            X_window: Last lookback rows of features including price_col
            lookback: Number of lookback steps (uses rf default if None)
            price_col: Price column name (uses rf default if None)
            
        Returns:
            Predicted price
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")

        lookback = lookback or self.config.lookback
        price_col = price_col or self.config.price_col

        # Get features (excluding price column for sequence building)
        feature_cols = [col for col in X_window.columns if col != price_col]
        Xw = X_window[feature_cols].values.astype("float32")

        if len(Xw) < lookback:
            raise ValueError(f"Need at least {lookback} rows to predict with LSTM")

        # Create sequence
        X_seq = Xw[-lookback:][None, :, :]

        # Predict return
        ret_pred = self.model.predict_return(X_seq)
        ret_val = float(ret_pred.reshape(-1)[0])

        # Calculate predicted price
        last_price = float(X_window[price_col].iloc[-1])
        predicted_price = last_price * (1.0 + ret_val)

        return predicted_price

    def save_model(self, model: B3KerasMTLModel, model_dir: str) -> str:
        """
        Save LSTM model to storage.
        
        Args:
            model: Trained B3KerasMTLModel
            model_dir: Directory to save the model
            
        Returns:
            Path to saved model
        """
        return self.saving_service.save_model(model.model, model_dir)

    def load_model(self, model_path: str) -> B3KerasMTLModel:
        """
        Load LSTM model from storage.
        
        Args:
            model_path: Path to saved model
            
        Returns:
            Loaded B3KerasMTLModel
        """
        keras_model = self.saving_service.load_model(model_path)

        # Create wrapper model
        model = B3KerasMTLModel(
            input_timesteps=self.config.lookback,
            input_features=keras_model.input_shape[2],  # Get feature count from loaded model
            config=LSTMConfig(
                lookback=self.config.lookback,
                horizon=self.config.horizon,
                units=self.config.units,
                dropout=self.config.dropout,
                learning_rate=self.config.learning_rate,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                loss_weight_action=self.config.loss_weight_action,
                loss_weight_return=self.config.loss_weight_return
            )
        )
        model.model = keras_model

        self.model = model
        self.is_trained = True
        return model

    def evaluate_model(self, model: B3KerasMTLModel, X_val: np.ndarray, yA_val: pd.Series,
                       X_test: np.ndarray, yA_test: pd.Series, df_processed: pd.DataFrame,
                       p0_val: np.ndarray = None, pf_val: np.ndarray = None,
                       p0_test: np.ndarray = None, pf_test: np.ndarray = None,
                       **kwargs) -> Dict[str, Any]:
        """
        Evaluate LSTM model with both classification and regression metrics.
        
        Args:
            model: Trained B3KerasMTLModel
            X_val: Validation feature sequences
            yA_val: Validation action targets
            X_test: Test feature sequences
            yA_test: Test action targets
            df_processed: Processed dataframe for visualization
            p0_val: Validation initial prices (for regression eval)
            pf_val: Validation future prices (for regression eval)
            p0_test: Test initial prices (for regression eval)
            pf_test: Test future prices (for regression eval)
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary containing evaluation results
        """
        model_name = kwargs.get('model_name', 'b3_lstm_mtl')
        persist_results = kwargs.get('persist_results', True)

        # Evaluate classification
        classification_results = self.evaluation_service.evaluate_model_comprehensive(
            model, X_val, yA_val, X_test, yA_test, df_processed,
            model_name=model_name, persist_results=persist_results
        )

        # Evaluate regression if price data is available
        regression_results = {}
        if p0_val is not None and pf_val is not None and p0_test is not None and pf_test is not None:
            # Predict returns
            ret_val_pred = model.predict_return(X_val)
            ret_test_pred = model.predict_return(X_test)

            # Calculate predicted prices
            price_val_pred = p0_val * (1.0 + ret_val_pred)
            price_test_pred = p0_test * (1.0 + ret_test_pred)

            # Evaluate regression
            reg_val = self.evaluation_service.evaluate_regression(pf_val, price_val_pred)
            reg_test = self.evaluation_service.evaluate_regression(pf_test, price_test_pred)

            regression_results = {
                'validation_regression': reg_val,
                'test_regression': reg_test
            }

            logging.info(f"Validation price regression: {reg_val}")
            logging.info(f"Test price regression: {reg_test}")

        return {
            'classification': classification_results,
            'regression': regression_results
        }
