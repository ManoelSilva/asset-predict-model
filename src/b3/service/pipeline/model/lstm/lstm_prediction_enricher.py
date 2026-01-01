import logging

import pandas as pd

from b3.service.pipeline.model.lstm.lstm_sequence_builder import LSTMSequenceBuilder
from b3.service.pipeline.model.utils import prepare_enrichment_dataframe, get_feature_columns


class LSTMPredictionEnricher:
    """Handles enrichment of dataframes with LSTM predictions for visualization."""

    def __init__(self, sequence_builder: LSTMSequenceBuilder, config):
        """
        Initialize prediction enricher.
        
        Args:
            sequence_builder: LSTMSequenceBuilder instance
            config: LSTM configuration object
        """
        self._sequence_builder = sequence_builder
        self._config = config

    def enrich_df_with_predictions(self, df: pd.DataFrame, model, max_samples: int = 5, scaler=None) -> None:
        """
        Add predictions to the dataframe for visualization.
        Operates in-place on the dataframe.
        Optimized to batch predictions.
        
        Args:
            df: Dataframe to enrich (modified in-place)
            model: Trained LSTM model
            max_samples: Maximum number of tickers to sample for predictions
            scaler: Optional scaler to transform features before prediction
        """

        # Define ticker validator for LSTM (requires sufficient data points)
        def ticker_validator(df: pd.DataFrame, ticker: str) -> bool:
            return len(df[df['ticker'] == ticker]) > self._config.lookback + self._config.horizon + 10

        result = prepare_enrichment_dataframe(
            df, max_samples=max_samples, ticker_validator=ticker_validator, check_limit=20
        )
        df, sample_tickers, sub_df = result

        if df is None or sub_df is None:
            return

        # Initialize predicted_price column (LSTM-specific)
        df['predicted_price'] = None

        # Sort by ticker and date to ensure alignment with sequence building
        sort_cols = self._sequence_builder.get_sort_columns(sub_df)
        sub_df = sub_df.sort_values(sort_cols)

        try:
            # Get feature columns using common utility
            feature_cols = get_feature_columns(sub_df)

            if not feature_cols:
                return

            X_sub = sub_df[feature_cols]
            y_sub = sub_df['target'] if 'target' in sub_df.columns else pd.Series(index=sub_df.index)

            # Build sequences for all tickers at once
            X_seq, _, _, _, _ = self._sequence_builder.build_sequences(
                X_sub, y_sub, sub_df,
                price_col=self._config.price_col,
                lookback=self._config.lookback,
                horizon=self._config.horizon
            )

            if len(X_seq) == 0:
                return

            logging.info(f"Predicting on {len(X_seq)} sequences for visualization...")
            
            # Scale features if scaler is provided
            if scaler is not None:
                N, T, F = X_seq.shape
                X_seq_scaled = scaler.transform(X_seq.reshape(N * T, F)).reshape(N, T, F)
            else:
                X_seq_scaled = X_seq
                # Warn if no scaler provided but model might expect one?
                # We can't know for sure here unless we inspect the model, but usually scaler is needed.

            # Batch prediction
            if hasattr(model, 'predict_return'):
                pred_actions = model.predict(X_seq_scaled)
                pred_returns = model.predict_return(X_seq_scaled)
            else:
                logging.warning("Model does not have predict_return method")
                return

            # Map predictions back to the dataframe indices
            current_idx = 0
            groups = sub_df.groupby("ticker", sort=False)

            for ticker, group in groups:
                size = len(group)
                n_w = size - self._config.horizon - self._config.lookback

                if n_w > 0:
                    # Indices in the original dataframe
                    target_indices = group.index[self._config.lookback: self._config.lookback + n_w]

                    # Extract predictions for this group
                    group_actions = pred_actions[current_idx: current_idx + n_w]
                    group_returns = pred_returns[current_idx: current_idx + n_w]

                    # Calculate prices
                    p0_indices = group.index[self._config.lookback - 1: self._config.lookback - 1 + n_w]
                    p0_values = group.loc[p0_indices, self._config.price_col].values.astype(float)

                    group_prices = p0_values * (1.0 + group_returns.flatten())

                    # Assign to original dataframe
                    df.loc[target_indices, 'predicted_action'] = group_actions
                    df.loc[target_indices, 'predicted_price'] = group_prices

                    current_idx += n_w

        except Exception as e:
            logging.warning(f"Failed to generate visualization predictions: {e}")
            import traceback
            logging.warning(traceback.format_exc())
