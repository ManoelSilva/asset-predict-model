import logging
from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd
from asset_model_data_storage.data_storage_service import DataStorageService


class B3DashboardPlotter:
    def __init__(self, storage_service: DataStorageService = None):
        """
        Initialize the plotter with storage service.
        
        Args:
            storage_service: Data storage service instance (optional, creates default if not provided)
        """
        self.storage_service = storage_service or DataStorageService()

    def plot_action_samples_by_ticker(
            self,
            df: pd.DataFrame,
            ticker_col: str = 'ticker',
            price_col: str = 'close',
            target_col: str = 'target',
            predicted_col: str = 'predicted_action',
            predicted_price_col: str = 'predicted_price',
            max_samples: int = 5,
            save_path: str = None
    ):
        """
        Plot price history and BUY/SELL/HOLD actions for a sample of tickers.
        Can plot both ground truth targets and model predictions.

        For each ticker (up to max_samples), this function plots the price series and overlays action markers.
        If predicted_col is present, it uses predictions for markers. Otherwise uses target_col.
        If predicted_price_col is present, it adds a line for predicted prices.

        Args:
            df (pd.DataFrame): DataFrame containing columns for ticker, price, date, and targets/predictions.
            ticker_col (str): Name of the ticker column (default: 'ticker').
            price_col (str): Name of the price column (default: 'close').
            target_col (str): Name of the ground truth target column (default: 'target').
            predicted_col (str): Name of the predicted action column (default: 'predicted_action').
            predicted_price_col (str): Name of the predicted price column (default: 'predicted_price').
            max_samples (int): Maximum number of tickers to plot (default: 5).
            save_path (str): If provided, saves the plot to this path.
        """
        logging.info(f"Plotting samples by ticker.. df shape: {df.shape}")

        # Smart ticker selection to show diverse examples (Buy/Sell/Hold)
        all_tickers = df[ticker_col].dropna().unique()
        selected_tickers = []

        # 1. Identify tickers with predictions
        tickers_with_preds = []
        if predicted_col in df.columns:
            # Check which tickers have actual predictions (not all None/NaN)
            # GroupBy is expensive on large DF, let's filter first if possible or iterate
            # Optimization: check uniqueness in a simpler way if possible, or just iterate valid ones

            # Fast check: get unique tickers that have non-null predictions
            pred_mask = df[predicted_col].notna()
            if pred_mask.any():
                tickers_with_preds = df.loc[pred_mask, ticker_col].unique().tolist()

        candidates = tickers_with_preds if tickers_with_preds else all_tickers.tolist()

        # 2. Try to find one example for each action type if predictions exist
        if tickers_with_preds:
            # Helper to find ticker with specific action
            def find_ticker_with_action(action_type, exclude_list):
                # We look at the dataframe where predicted_col == action_type
                mask = (df[predicted_col].astype(str).str.lower().str.contains(action_type)) & \
                       (df[ticker_col].isin(candidates)) & \
                       (~df[ticker_col].isin(exclude_list))

                found = df.loc[mask, ticker_col].unique()
                if len(found) > 0:
                    return found[0]
                return None

            # Try to get one of each
            for action in ['buy', 'sell', 'hold']:
                t = find_ticker_with_action(action, selected_tickers)
                if t:
                    selected_tickers.append(t)

        # 3. Fill the rest with remaining candidates
        remaining = [t for t in candidates if t not in selected_tickers]

        # If we still have space and didn't fill from preds (or didn't have preds), 
        # add from all_tickers if needed (though candidates handles this)
        if len(selected_tickers) < max_samples:
            fill_count = max_samples - len(selected_tickers)
            selected_tickers.extend(remaining[:fill_count])

        # 4. If we still don't have enough (unlikely unless dataset is small), add from all_tickers
        if len(selected_tickers) < max_samples:
            remaining_all = [t for t in all_tickers if t not in selected_tickers]
            fill_count = max_samples - len(selected_tickers)
            selected_tickers.extend(remaining_all[:fill_count])

        sample_tickers = selected_tickers[:max_samples]

        n = len(sample_tickers)
        if n == 0:
            logging.warning("No tickers found in dataframe for plotting.")
            return

        _, axes = plt.subplots(n, 1, figsize=(12, 5 * n), sharex=False)
        if n == 1:
            axes = [axes]

        for i, ticker in enumerate(sample_tickers):
            sub_df = df[df[ticker_col] == ticker].sort_values('date')

            # Extract data
            dates = pd.to_datetime(sub_df['date'])
            prices = sub_df[price_col]

            # Plot Actual Price
            axes[i].plot(dates, prices, label='Actual Price', color='blue', linewidth=1.5)

            # Plot Predicted Price if available
            if predicted_price_col in sub_df.columns and sub_df[predicted_price_col].notna().any():
                pred_prices = sub_df[predicted_price_col]
                axes[i].plot(dates, pred_prices, label='Predicted Price', color='orange', linestyle='--', linewidth=1.5,
                             alpha=0.8)

            # Determine which actions to plot (Predictions preferred if available)
            action_source = predicted_col if predicted_col in sub_df.columns else target_col
            actions = sub_df[action_source].astype(str).str.lower().fillna('none')

            # Clean action names (remove _target suffix if present)
            actions = actions.str.replace('_target', '', regex=False)

            # Define markers
            buy_mask = actions == 'buy'
            sell_mask = actions == 'sell'
            hold_mask = actions == 'hold'

            # Plot markers
            if buy_mask.any():
                axes[i].scatter(dates[buy_mask], prices[buy_mask], color='green', label='Buy Signal', marker='^', s=100,
                                zorder=5)
            if sell_mask.any():
                axes[i].scatter(dates[sell_mask], prices[sell_mask], color='red', label='Sell Signal', marker='v',
                                s=100, zorder=5)
            if hold_mask.any():
                # Plot holds smaller and less opaque
                axes[i].scatter(dates[hold_mask], prices[hold_mask], color='gray', label='Hold Signal', marker='o',
                                s=30, alpha=0.3)

            title_suffix = "(Predictions)" if action_source == predicted_col else "(Ground Truth)"
            axes[i].set_title(f"{ticker} - {title_suffix}")
            axes[i].set_ylabel('Price')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        plt.xlabel('Date')
        plt.tight_layout()

        if save_path:
            try:
                # Create directory if using local storage
                if self.storage_service.is_local_storage():
                    import os
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                # Save plot to bytes
                plot_bytes = BytesIO()
                plt.savefig(plot_bytes, dpi=300, bbox_inches='tight', format='png')
                plot_bytes.seek(0)

                # Save using storage service
                saved_path = self.storage_service.save_file(save_path, plot_bytes, 'image/png')
                logging.info(f"Plot saved successfully to: {saved_path}")
            except Exception as e:
                logging.error(f"Failed to save plot to {save_path}: {e}")
                raise
            finally:
                plt.close()
        else:
            plt.show()
