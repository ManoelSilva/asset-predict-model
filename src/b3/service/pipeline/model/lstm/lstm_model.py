import logging
import gc
import os
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from asset_model_data_storage.data_storage_service import DataStorageService
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split

from b3.service.pipeline.model.lstm.pytorch_mtl_model import B3PytorchMTLModel
from b3.service.pipeline.model.lstm.lstm_config import LSTMConfig
from b3.service.pipeline.model.model import BaseModel
from b3.service.pipeline.model_evaluation_service import B3ModelEvaluationService
from b3.service.pipeline.persist.lstm_persist_service import LSTMPersistService
from constants import RANDOM_STATE, FEATURE_SET


class LSTMModel(BaseModel):
    """
    LSTM Multi-Task Learning implementation for B3 asset prediction using PyTorch.
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
        self._storage_service = storage_service or DataStorageService()
        self._saving_service = LSTMPersistService(self._storage_service)
        self._evaluation_service = B3ModelEvaluationService(self._storage_service)
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() and os.environ.get('cuda_enabled', False) else "cpu")

    @property
    def _config(self) -> LSTMConfig:
        """
        Get LSTM configuration, ensuring it's not None.
        
        Returns:
            LSTMConfig instance
            
        Raises:
            ValueError: If config is None or not an LSTMConfig instance
        """
        if self.config is None:
            raise ValueError("LSTM config is None. This should not happen as it's initialized with a default.")
        if not isinstance(self.config, LSTMConfig):
            raise ValueError(f"Expected LSTMConfig, got {type(self.config).__name__}")
        return self.config

    def run_pipeline(self, X: pd.DataFrame, y: pd.Series, df_processed: pd.DataFrame, config: Any) -> Tuple[str, Dict]:
        """
        Runs the full LSTM training pipeline.
        """
        logging.info("Starting LSTM training pipeline")

        # 1. Prepare
        lstm_config_dict = config.get_lstm_config_dict()
        X_seq, yA_seq, yR_seq, p0_seq, pf_seq = self.prepare_data(
            X, y, df_processed, **lstm_config_dict
        )

        # 2. Split
        (X_train, X_val, X_test,
         yA_train, yA_val, yA_test,
         yR_train, yR_val, yR_test,
         p0_train, p0_val, p0_test,
         pf_train, pf_val, pf_test) = self.split_data(
            X_seq, yA_seq, yR_seq, p0_seq, pf_seq, config.test_size, config.val_size
        )

        # 3. Train
        # Note: using X.shape[1] for n_features which is the number of features in the raw dataframe
        trained_model = self.train_model(
            X_train, yA_train, yR_train, lookback=config.lookback, n_features=X.shape[1]
        )

        # 4. Evaluate
        evaluation_results = self.evaluate_model(
            trained_model, X_val, yA_val, X_test, yA_test, df_processed,
            p0_val=p0_val, pf_val=pf_val, p0_test=p0_test, pf_test=pf_test
        )

        # Add training history to evaluation results
        if hasattr(trained_model, 'history'):
            evaluation_results['training_history'] = trained_model.history

        # 5. Save
        model_path = self.save_model(trained_model, config.model_dir)
        logging.info(f"LSTM-MTL training completed successfully. Model saved at: {model_path}")

        return model_path, evaluation_results

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

        lookback = kwargs.get('lookback', self._config.lookback)
        horizon = kwargs.get('horizon', self._config.horizon)
        price_col = kwargs.get('price_col', self._config.price_col)

        logging.info(f"Building LSTM sequences with lookback={lookback}, horizon={horizon}")

        X_seq, yA_seq, yR_seq, p0_seq, pf_seq = self._build_sequences(
            X, y, df_processed, price_col=price_col, lookback=lookback, horizon=horizon
        )

        if len(X_seq) == 0:
            raise ValueError("No sequences built for LSTM. Check ordering, ticker column, and lookback/horizon.")

        logging.info(f"Built {len(X_seq)} sequences for LSTM training")
        return X_seq, yA_seq, yR_seq, p0_seq, pf_seq

    def _build_sequences(self, X_df: DataFrame, y_action: Series, df_full: DataFrame,
                         price_col: str = "close", lookback: int = 32, horizon: int = 1
                         ) -> Tuple[np.ndarray, Series, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build sequences for multitask learning using PyTorch for acceleration.
        Optimized to avoid repeated dataframe filtering.
        """
        features = X_df.columns.tolist()
        n_features = len(features)
        device = self._device

        # Pre-process targets globally
        y_processed = y_action.astype(str).str.replace("_target", "", regex=False)

        # ---------------------------------------------------------
        # 1. OPTIMIZED PREPARATION (Replaces the slow ticker loop)
        # ---------------------------------------------------------

        # Ensure we only work with indices present in both DataFrames
        common_indices = df_full.index.intersection(X_df.index)
        if len(common_indices) <= lookback + horizon:
            return (np.zeros((0, lookback, n_features), dtype="float32"),
                    pd.Series([], dtype="object"),
                    np.zeros((0,), dtype="float32"),
                    np.zeros((0,), dtype="float32"),
                    np.zeros((0,), dtype="float32"))

        # Create a working subset aligned with X_df
        # We work with this subset to guarantee index alignment
        df_subset = df_full.loc[common_indices].copy()

        # Determine sort columns (Ticker + Date)
        date_col = next((c for c in ["date", "datetime", "timestamp"] if c in df_subset.columns), None)
        sort_cols = []
        if "ticker" in df_subset.columns:
            sort_cols.append("ticker")
        if date_col:
            sort_cols.append(date_col)

        # Global Sort: This is O(N log N) and much faster than N * O(N) filtering
        if sort_cols:
            df_subset = df_subset.sort_values(sort_cols)

        # Extract Aligned Data (CPU)
        # Using the sorted index ensures all arrays are perfectly aligned by row
        # .values creates a copy, which is fine and safer for the next steps
        sorted_indices = df_subset.index

        X_all = X_df.loc[sorted_indices, features].values.astype("float32")
        y_all = y_processed.loc[sorted_indices].values

        if price_col not in df_subset.columns:
            raise ValueError(f"Price column '{price_col}' not found")
        p_all = df_subset[price_col].astype(float).values

        # Calculate Group Sizes
        # Since data is sorted by ticker, we can simply count group sizes
        if "ticker" in df_subset.columns:
            # groupby size on a sorted dataframe is extremely fast
            group_sizes = df_subset.groupby("ticker", sort=False).size().values
        else:
            group_sizes = np.array([len(df_subset)])

        # Calculate valid windows per group
        # Each group produces (size - lookback - horizon) valid targets
        # Note: The original logic used `n_samples - horizon - lookback`
        valid_windows_per_group = np.maximum(0, group_sizes - horizon - lookback)
        total_sequences = valid_windows_per_group.sum()

        if total_sequences == 0:
            return (np.zeros((0, lookback, n_features), dtype="float32"),
                    pd.Series([], dtype="object"),
                    np.zeros((0,), dtype="float32"),
                    np.zeros((0,), dtype="float32"),
                    np.zeros((0,), dtype="float32"))

        logging.info(f"Allocating tensors for {total_sequences} sequences on {device}...")

        # ---------------------------------------------------------
        # 2. PRE-ALLOCATION
        # ---------------------------------------------------------
        try:
            X_seq = torch.zeros((total_sequences, lookback, n_features), dtype=torch.float32, device=device)
            y_return_seq = torch.zeros((total_sequences,), dtype=torch.float32, device=device)
            last_price_seq = torch.zeros((total_sequences,), dtype=torch.float32, device=device)
            future_price_seq = torch.zeros((total_sequences,), dtype=torch.float32, device=device)
        except RuntimeError as e:
            logging.warning(f"Allocation failed on {device}: {e}. Falling back to CPU.")
            device = torch.device('cpu')
            X_seq = torch.zeros((total_sequences, lookback, n_features), dtype=torch.float32, device=device)
            y_return_seq = torch.zeros((total_sequences,), dtype=torch.float32, device=device)
            last_price_seq = torch.zeros((total_sequences,), dtype=torch.float32, device=device)
            future_price_seq = torch.zeros((total_sequences,), dtype=torch.float32, device=device)

        y_action_seq = np.empty((total_sequences,), dtype=object)

        # ---------------------------------------------------------
        # 3. VECTORIZED PROCESSING (Sliced Iteration)
        # ---------------------------------------------------------

        start_idx = 0  # Index in the source arrays (X_all, etc.)
        current_seq_idx = 0  # Index in the destination tensors (X_seq, etc.)

        for size, n_w in zip(group_sizes, valid_windows_per_group):
            if n_w > 0:
                # 1. Slice the chunk for this ticker (Zero-copy view usually)
                Xg = X_all[start_idx: start_idx + size]
                yg = y_all[start_idx: start_idx + size]
                pg = p_all[start_idx: start_idx + size]

                # 2. Move Chunk to GPU
                Xg_tensor = torch.tensor(Xg, dtype=torch.float32, device=device)
                pg_tensor = torch.tensor(pg, dtype=torch.float32, device=device)

                # 3. Vectorized Windowing (PyTorch Unfold)
                # Shape: (n_windows_raw, n_features, lookback)
                windows = Xg_tensor.unfold(0, lookback, 1)

                # We only need the first n_w windows that have valid future targets
                # The unfold creates (size - lookback + 1) windows.
                # We need exactly n_w.
                windows = windows[:n_w]

                # Transpose to (Batch, Time, Features)
                windows = windows.transpose(1, 2)

                # 4. Fill Tensors
                X_seq[current_seq_idx: current_seq_idx + n_w] = windows

                # Targets (CPU)
                # The target for a window ending at t is the action at t
                # Our windows end at index `lookback-1` (0-based) relative to Xg start
                # So we take targets from lookback to lookback + n_w
                y_action_seq[current_seq_idx: current_seq_idx + n_w] = yg[lookback: lookback + n_w]

                # Price Calculations (GPU)
                p0 = pg_tensor[lookback - 1: lookback - 1 + n_w]
                p1 = pg_tensor[lookback - 1 + horizon: lookback - 1 + horizon + n_w]

                ret = (p1 / p0) - 1.0

                y_return_seq[current_seq_idx: current_seq_idx + n_w] = ret
                last_price_seq[current_seq_idx: current_seq_idx + n_w] = p0
                future_price_seq[current_seq_idx: current_seq_idx + n_w] = p1

                current_seq_idx += n_w

                # Clear GPU temp vars
                del Xg_tensor, pg_tensor, windows, p0, p1, ret

            # Move to next group
            start_idx += size

        gc.collect()

        logging.info("Moving sequences to CPU/Numpy...")
        return (X_seq.cpu().numpy(),
                pd.Series(y_action_seq, name="target"),
                y_return_seq.cpu().numpy(),
                last_price_seq.cpu().numpy(),
                future_price_seq.cpu().numpy())

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

    def train_model(self, X_train: np.ndarray, yA_train: pd.Series, yR_train: np.ndarray,
                    **kwargs) -> B3PytorchMTLModel:
        """
        Train LSTM Multi-Task Learning model.
        
        Args:
            X_train: Training feature sequences
            yA_train: Training action targets
            yR_train: Training return targets
            **kwargs: Additional training parameters
            
        Returns:
            Trained B3PytorchMTLModel
        """
        lookback = kwargs.get('lookback', self._config.lookback)
        n_features = kwargs.get('n_features', X_train.shape[2])

        lstm_config = LSTMConfig(
            lookback=lookback,
            horizon=self._config.horizon,
            units=self._config.units,
            dropout=self._config.dropout,
            learning_rate=self._config.learning_rate,
            epochs=self._config.epochs,
            batch_size=self._config.batch_size,
            loss_weight_action=self._config.loss_weight_action,
            loss_weight_return=self._config.loss_weight_return
        )

        logging.info(
            f"Training LSTM model (PyTorch) with {lstm_config.epochs} epochs, batch_size={lstm_config.batch_size}")

        clf = B3PytorchMTLModel(input_timesteps=lookback, input_features=n_features, config=lstm_config,
                                device=self._device)
        clf.fit(X_train, yA_train, yR_train)

        logging.info("LSTM multi-task training completed")

        self.model = clf
        self.is_trained = True

        logging.info("LSTM training completed successfully")
        return clf

    def _enrich_df_with_predictions(self, df: pd.DataFrame, max_samples: int = 5):
        """
        Add predictions to the dataframe for visualization.
        Operates in-place or returns modified dataframe.
        Optimized to batch predictions.
        """
        if df is None or df.empty:
            return

        # Pick sample tickers
        if 'ticker' not in df.columns:
            return

        all_tickers = df['ticker'].unique()
        # Prefer tickers that have enough data
        valid_tickers = []
        # Check a few tickers first to avoid scanning all if there are many
        check_limit = min(len(all_tickers), 20)

        for t in all_tickers[:check_limit]:
            if len(df[df['ticker'] == t]) > self._config.lookback + self._config.horizon + 10:
                valid_tickers.append(t)

        sample_tickers = valid_tickers[:max_samples]
        if not sample_tickers:
            sample_tickers = all_tickers[:max_samples]

        logging.info(f"Generating predictions for visualization tickers: {sample_tickers}")

        # Initialize columns
        df['predicted_action'] = None
        df['predicted_price'] = None

        # Filter dataframe for all sample tickers
        mask = df['ticker'].isin(sample_tickers)
        if not mask.any():
            return

        sub_df = df[mask].copy()

        # Sort by ticker and date to ensure alignment with _build_sequences
        # Note: _build_sequences does its own sorting internally on the passed dataframe,
        # but we need to replicate that sort to map indices back.
        sort_cols = ['ticker']
        date_col = next((c for c in ["date", "datetime", "timestamp"] if c in sub_df.columns), None)
        if date_col:
            sort_cols.append(date_col)

        sub_df = sub_df.sort_values(sort_cols)

        try:
            # Prepare features and targets for the whole batch
            # Use strict feature set to avoid shape mismatches or boolean column issues
            feature_cols = [c for c in FEATURE_SET if c in sub_df.columns]

            if not feature_cols:
                logging.warning(f"No valid features found for enrichment. Expected some of: {FEATURE_SET}")
                return

            X_sub = sub_df[feature_cols]
            y_sub = sub_df['target'] if 'target' in sub_df.columns else pd.Series(index=sub_df.index)

            # Build sequences for all tickers at once
            X_seq, _, _, _, _ = self._build_sequences(
                X_sub, y_sub, sub_df,
                price_col=self._config.price_col,
                lookback=self._config.lookback,
                horizon=self._config.horizon
            )

            if len(X_seq) == 0:
                return

            logging.info(f"Predicting on {len(X_seq)} sequences for visualization...")

            # Batch prediction
            pred_actions = self.predict(X_seq)
            pred_returns = self.predict_return(X_seq)

            # Now we need to map predictions back to the dataframe indices.
            # We iterate through the groups in the same order as _build_sequences did.

            current_idx = 0

            # Group sizes in the sorted dataframe
            groups = sub_df.groupby("ticker", sort=False)

            for ticker, group in groups:
                size = len(group)
                n_w = size - self._config.horizon - self._config.lookback

                if n_w > 0:
                    # The sequences correspond to the valid windows for this group
                    # Based on _build_sequences logic, y_action_seq[k] ~ yg[lookback + k]
                    # So predictions align with rows starting at `lookback` index within the group

                    # Indices in the original dataframe (group is a slice of sub_df)
                    target_indices = group.index[self._config.lookback: self._config.lookback + n_w]

                    # Extract predictions for this group
                    group_actions = pred_actions[current_idx: current_idx + n_w]
                    group_returns = pred_returns[current_idx: current_idx + n_w]

                    # Calculate prices
                    # p0 is at `lookback - 1 + k`
                    # We need the prices at (lookback - 1) relative to group start
                    p0_indices = group.index[self._config.lookback - 1: self._config.lookback - 1 + n_w]
                    p0_values = group.loc[p0_indices, self._config.price_col].values.astype(float)

                    group_prices = p0_values * (1.0 + group_returns.flatten())

                    # Assign to original dataframe (df)
                    df.loc[target_indices, 'predicted_action'] = group_actions
                    df.loc[target_indices, 'predicted_price'] = group_prices

                    current_idx += n_w

        except Exception as e:
            logging.warning(f"Failed to generate visualization predictions: {e}")

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

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Make probability predictions using trained LSTM model.
        
        Args:
            X: Feature sequences for prediction
            
        Returns:
            Array of probability predictions
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")

        if not self.validate_data(X):
            raise ValueError("Invalid input data for prediction")

        return self.model.predict_proba(X)

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

    def save_model(self, model: B3PytorchMTLModel, model_dir: str) -> str:
        """
        Save LSTM model to storage.
        
        Args:
            model: Trained B3PytorchMTLModel
            model_dir: Directory to save the model
            
        Returns:
            Path to saved model
        """
        return self._saving_service.save_model(model.model, model_dir)

    def load_model(self, model_path: str) -> B3PytorchMTLModel:
        """
        Load LSTM model from storage.
        
        Args:
            model_path: Path to saved model
            
        Returns:
            Loaded B3PytorchMTLModel
        """
        checkpoint = self._saving_service.load_model(model_path)

        # Extract config from checkpoint
        input_features = checkpoint['input_features']
        hidden_size = checkpoint['hidden_size']
        dropout = checkpoint['dropout']
        state_dict = checkpoint['state_dict']

        # Update config with loaded architecture params
        # Note: We prefer to use the loaded architecture over self._config for the model structure
        # to ensure the weights match.

        model_config = LSTMConfig(
            lookback=self._config.lookback,
            # This might be part of model/wrapper logic, but input_features drives the model input size.
            horizon=self._config.horizon,
            units=hidden_size,
            dropout=dropout,
            # Other params from current config or defaults
            learning_rate=self._config.learning_rate,
            epochs=self._config.epochs,
            batch_size=self._config.batch_size,
            loss_weight_action=self._config.loss_weight_action,
            loss_weight_return=self._config.loss_weight_return
        )

        # Create wrapper model
        model = B3PytorchMTLModel(
            input_timesteps=self._config.lookback,
            input_features=input_features,
            config=model_config,
            device=self._device
        )

        model.model.load_state_dict(state_dict)

        self.model = model
        self.is_trained = True
        return model

    def evaluate_model(self, model: B3PytorchMTLModel, X_val: np.ndarray, yA_val: pd.Series,
                       X_test: np.ndarray, yA_test: pd.Series, df_processed: pd.DataFrame,
                       p0_val: np.ndarray = None, pf_val: np.ndarray = None,
                       p0_test: np.ndarray = None, pf_test: np.ndarray = None,
                       **kwargs) -> Dict[str, Any]:
        """
        Evaluate LSTM model with both classification and regression metrics.
        
        Args:
            model: Trained B3PytorchMTLModel
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

        # Enrich dataframe with predictions for visualization
        # We work on a copy to avoid side effects
        df_viz = df_processed.copy()
        # Generate predictions for a larger pool of tickers (e.g. 20) to ensure the plotter
        # can find diverse examples (Buy/Sell/Hold).
        self._enrich_df_with_predictions(df_viz, max_samples=20)

        # Evaluate classification
        classification_results = self._evaluation_service.evaluate_model_comprehensive(
            model, X_val, yA_val, X_test, yA_test, df_viz,
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
            reg_val = self._evaluation_service.evaluate_regression(pf_val, price_val_pred)
            reg_test = self._evaluation_service.evaluate_regression(pf_test, price_test_pred)

            regression_results = {
                'validation_regression': reg_val,
                'test_regression': reg_test
            }

            logging.info(f"Validation price regression: {reg_val}")
            logging.info(f"Test price regression: {reg_test}")

            if persist_results:
                try:
                    # Update validation record
                    if 'validation' in classification_results and 'evaluation_id' in classification_results[
                        'validation']:
                        val_id = classification_results['validation']['evaluation_id']
                        self._evaluation_service.update_evaluation_metrics(val_id, {'price_regression': reg_val})
                        logging.info(f"Updated validation evaluation {val_id} with regression metrics")

                    # Update test record
                    if 'test' in classification_results and 'evaluation_id' in classification_results['test']:
                        test_id = classification_results['test']['evaluation_id']
                        self._evaluation_service.update_evaluation_metrics(test_id, {'price_regression': reg_test})
                        logging.info(f"Updated test evaluation {test_id} with regression metrics")
                except Exception as e:
                    logging.error(f"Error persisting regression metrics: {str(e)}")

        return {
            'classification': classification_results,
            'regression': regression_results
        }
