import pandas as pd
import matplotlib.pyplot as plt
import logging


class B3DashboardPlotter:
    @staticmethod
    def plot_action_samples_by_ticker(
            df: pd.DataFrame,
            ticker_col: str = 'ticker',
            price_col: str = 'price_momentum_5',
            target_col: str = 'target',
            max_samples: int = 5,
            save_path: str = None
    ):
        """
        Plot price history and BUY/SELL/HOLD actions for a sample of tickers using the provided target column.

        For each ticker (up to max_samples), this function plots the price series and overlays action markers
        based on the values in the target column:
        - 'buy_target': green upward triangle (BUY)
        - 'sell_target': red downward triangle (SELL)
        - 'hold': gray circle (HOLD)

        Args:
            df (pd.DataFrame): DataFrame containing at least the columns for ticker, price, date, and target.
            ticker_col (str): Name of the ticker column in df (default: 'ticker').
            price_col (str): Name of the price column in df (default: 'price_momentum_5').
            target_col (str): Name of the action/target column in df (should contain 'buy_target', 'sell_target', 'hold').
            max_samples (int): Maximum number of tickers to plot (default: 5).
            save_path (str): If provided, saves the plot to this path instead of displaying it.
        """
        logging.info(f"Plotting action samples by ticker.. df: \n{df.head()}")

        sample_tickers = df[ticker_col].dropna().unique()[:max_samples]
        n = len(sample_tickers)
        _, axes = plt.subplots(n, 1, figsize=(10, 4 * n), sharex=False)
        if n == 1:
            axes = [axes]
        for i, ticker in enumerate(sample_tickers):
            sub_df = df[df[ticker_col] == ticker].sort_values('date')
            prices = sub_df[price_col]
            actions = sub_df[target_col].astype(str).str.lower()
            axes[i].plot(sub_df['date'], prices, label='Price', color='blue')
            # Plot BUY, SELL, HOLD markers
            buy_idx = actions[actions == 'buy_target'].index
            sell_idx = actions[actions == 'sell_target'].index
            hold_idx = actions[actions == 'hold'].index
            axes[i].scatter(sub_df.loc[buy_idx, 'date'], prices.loc[buy_idx], color='green', label='BUY', marker='^')
            axes[i].scatter(sub_df.loc[sell_idx, 'date'], prices.loc[sell_idx], color='red', label='SELL', marker='v')
            axes[i].scatter(sub_df.loc[hold_idx, 'date'], prices.loc[hold_idx], color='gray', label='HOLD', marker='o',
                            alpha=0.5)
            axes[i].set_title(f"{ticker} Price & Action Samples")
            axes[i].set_ylabel('Price')
            axes[i].legend()
        plt.xlabel('Date')
        plt.tight_layout()
        
        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logging.info(f"Plot saved successfully to: {save_path}")
            except Exception as e:
                logging.error(f"Failed to save plot to {save_path}: {e}")
                raise
            finally:
                plt.close()  # Close the figure to free memory
        else:
            plt.show()
