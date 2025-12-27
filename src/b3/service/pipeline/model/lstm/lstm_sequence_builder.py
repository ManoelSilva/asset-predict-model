import gc
import logging
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame, Series


class LSTMSequenceBuilder:
    """Handles building of LSTM sequences from dataframes."""

    def __init__(self, device: torch.device):
        """
        Initialize sequence builder.
        
        Args:
            device: PyTorch device for tensor operations
        """
        self._device = device

    @staticmethod
    def get_sort_columns(df: pd.DataFrame) -> list:
        """
        Determine sort columns for dataframe (ticker and date if available).
        
        Args:
            df: Dataframe to determine sort columns for
            
        Returns:
            List of column names to sort by
        """
        date_col = next((c for c in ["date", "datetime", "timestamp"] if c in df.columns), None)
        sort_cols = []
        if "ticker" in df.columns:
            sort_cols.append("ticker")
        if date_col:
            sort_cols.append(date_col)
        return sort_cols

    @staticmethod
    def _create_empty_sequences(lookback: int, n_features: int) -> Tuple[
        np.ndarray, pd.Series, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create empty sequences tuple for early return cases.
        
        Args:
            lookback: Sequence lookback length
            n_features: Number of features
            
        Returns:
            Tuple of empty arrays and series matching sequence structure
        """
        return (np.zeros((0, lookback, n_features), dtype="float32"),
                pd.Series([], dtype="object"),
                np.zeros((0,), dtype="float32"),
                np.zeros((0,), dtype="float32"),
                np.zeros((0,), dtype="float32"))

    @staticmethod
    def _allocate_sequences_tensors(total_sequences: int, lookback: int, n_features: int, device: torch.device) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.device]:
        """
        Allocate tensors for sequences with fallback to CPU if GPU allocation fails.
        
        Args:
            total_sequences: Total number of sequences to allocate
            lookback: Sequence lookback length
            n_features: Number of features
            device: Target device for allocation
            
        Returns:
            Tuple containing (X_seq, y_return_seq, last_price_seq, future_price_seq, device)
        """

        def _allocate_tensors(seqs: int, lb: int, n_feat: int, dvc: torch.device) -> \
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.device]:
            X_seq = torch.zeros((seqs, lb, n_feat), dtype=torch.float32, device=dvc)
            y_return_seq = torch.zeros((seqs,), dtype=torch.float32, device=dvc)
            last_price_seq = torch.zeros((seqs,), dtype=torch.float32, device=dvc)
            future_price_seq = torch.zeros((seqs,), dtype=torch.float32, device=dvc)
            return X_seq, y_return_seq, last_price_seq, future_price_seq, dvc

        try:
            return _allocate_tensors(total_sequences, lookback, n_features, device)
        except RuntimeError as e:
            logging.warning(f"Allocation failed on {device}: {e}. Falling back to CPU.")
            return _allocate_tensors(total_sequences, lookback, n_features, torch.device('cpu'))

    def build_sequences(self, X_df: DataFrame, y_action: Series, df_full: DataFrame,
                        price_col: str = "close", lookback: int = 32, horizon: int = 1
                        ) -> Tuple[np.ndarray, Series, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build sequences for multitask learning using PyTorch for acceleration.
        Optimized to avoid repeated dataframe filtering.
        
        Args:
            X_df: Feature dataframe
            y_action: Action target series
            df_full: Full processed dataframe
            price_col: Price column name
            lookback: Sequence lookback length
            horizon: Prediction horizon
            
        Returns:
            Tuple containing (X_seq, y_action_seq, y_return_seq, last_price_seq, future_price_seq)
        """
        features = X_df.columns.tolist()
        n_features = len(features)
        device = self._device

        # Pre-process targets globally
        y_processed = y_action.astype(str).str.replace("_target", "", regex=False)

        # Ensure we only work with indices present in both DataFrames
        common_indices = df_full.index.intersection(X_df.index)
        if len(common_indices) <= lookback + horizon:
            return self._create_empty_sequences(lookback, n_features)

        # Create a working subset aligned with X_df
        df_subset = df_full.loc[common_indices].copy()

        # Determine sort columns (Ticker + Date)
        sort_cols = self.get_sort_columns(df_subset)

        # Global Sort: This is O(N log N) and much faster than N * O(N) filtering
        if sort_cols:
            df_subset = df_subset.sort_values(sort_cols)

        # Extract Aligned Data (CPU)
        sorted_indices = df_subset.index
        X_all = X_df.loc[sorted_indices, features].values.astype("float32")
        y_all = y_processed.loc[sorted_indices].values

        if price_col not in df_subset.columns:
            raise ValueError(f"Price column '{price_col}' not found")
        p_all = df_subset[price_col].astype(float).values

        # Calculate Group Sizes
        if "ticker" in df_subset.columns:
            group_sizes = df_subset.groupby("ticker", sort=False).size().values
        else:
            group_sizes = np.array([len(df_subset)])

        # Calculate valid windows per group
        valid_windows_per_group = np.maximum(0, group_sizes - horizon - lookback)
        total_sequences = valid_windows_per_group.sum()

        if total_sequences == 0:
            return self._create_empty_sequences(lookback, n_features)

        logging.info(f"Allocating tensors for {total_sequences} sequences on {device}...")

        # Pre-allocation
        X_seq, y_return_seq, last_price_seq, future_price_seq, device = self._allocate_sequences_tensors(
            total_sequences, lookback, n_features, device
        )

        y_action_seq = np.empty((total_sequences,), dtype=object)

        # Vectorized Processing (Sliced Iteration)
        start_idx = 0
        current_seq_idx = 0

        for size, n_w in zip(group_sizes, valid_windows_per_group):
            if n_w > 0:
                # Slice the chunk for this ticker
                Xg = X_all[start_idx: start_idx + size]
                yg = y_all[start_idx: start_idx + size]
                pg = p_all[start_idx: start_idx + size]

                # Move Chunk to GPU
                Xg_tensor = torch.tensor(Xg, dtype=torch.float32, device=device)
                pg_tensor = torch.tensor(pg, dtype=torch.float32, device=device)

                # Vectorized Windowing (PyTorch Unfold)
                windows = Xg_tensor.unfold(0, lookback, 1)
                windows = windows[:n_w]
                windows = windows.transpose(1, 2)

                # Fill Tensors
                X_seq[current_seq_idx: current_seq_idx + n_w] = windows

                # Targets (CPU)
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

            start_idx += size

        gc.collect()

        logging.info("Moving sequences to CPU/Numpy...")
        return (X_seq.cpu().numpy(),
                pd.Series(y_action_seq, name="target"),
                y_return_seq.cpu().numpy(),
                last_price_seq.cpu().numpy(),
                future_price_seq.cpu().numpy())
