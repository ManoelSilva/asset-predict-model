from b3.data_loader import B3DataLoader
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
import numpy as np
import os
import logging

from b3.plotter import B3DashboardPlotter


class B3Model(object):
    FEATURE_SET = [
        'rolling_volatility_5',
        'moving_avg_10',
        'macd',
        'rsi_14',
        'volume_change',
        'avg_volume_10',
        'best_buy_sell_spread',
        'close_to_best_buy',
        'price_momentum_5',
        'high_breakout_20',
        'bollinger_upper',
        'stochastic_14'
    ]

    def __init__(self) -> None:
        self._loader = B3DataLoader()

    @staticmethod
    def predict(X: pd.DataFrame, model_dir: str = "models"):
        """
        Predicts BUY/SELL for new data using saved classifier.
        Args:
            X (pd.DataFrame): Feature data.
            model_dir (str): Directory where models are saved.
        Returns:
            pd.Series: Predicted actions (BUY/SELL).
        """
        clf_path = os.path.join(model_dir, 'rf_classifier.joblib')
        clf = joblib.load(clf_path)
        features = [f for f in B3Model.FEATURE_SET if f in X.columns]
        X_selected = X[features]
        preds = clf.predict(X_selected)
        return preds

    def train_and_save(self, model_dir: str = "models", features: list = None):
        """
        Loads data, creates BUY/SELL labels, trains a RandomForestClassifier, and saves the model.
        Args:
            model_dir (str): Directory to save models.
            features (list): List of feature column names to use. If None, use all except 'date' and non-numeric columns.
        """
        # Load data
        df = self._loader.fetch_all()
        features = [f for f in B3Model.FEATURE_SET if f in df.columns]
        missing_features = set(B3Model.FEATURE_SET) - set(features)
        if missing_features:
            logging.warning(f"Missing features in data: {missing_features}")
        X = df[features]

        # Clean data: replace infinite values with NaN and drop rows with NaN
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.dropna()
        df = df.loc[X.index]  # align df with cleaned X

        # --- Cluster-based signal logic ---
        # Buy: RSI < 30 and MACD > 0 and volume increasing
        df['buy_signal_cluster'] = (
                (df['rsi_14'] < 30) &
                (df['macd'] > 0) &
                (df['volume_change'] > 0)
        )
        # Sell: RSI > 70 and momentum negative and price at Bollinger upper with falling volume
        df['sell_signal_cluster'] = (
                (df['rsi_14'] > 70) &
                (df['price_momentum_5'] < 0) &
                (df['bollinger_upper'].notnull()) &
                (df['volume_change'] < 0)
        )
        # Assign target labels
        df['target'] = np.select(
            [df['buy_signal_cluster'], df['sell_signal_cluster']],
            ['buy_target', 'sell_target'],
            default='hold'
        )
        y = df['target']
        # Visualize sample actions using plotter (removed price_col)
        B3DashboardPlotter.plot_action_samples_by_ticker(df)

        # Train supervised model to predict BUY/SELL
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X, y)

        # Prepare model directory
        os.makedirs(model_dir, exist_ok=True)
        clf_path = os.path.join(model_dir, 'rf_classifier.joblib')

        # Save model
        joblib.dump(clf, clf_path)
        logging.info(f"Model saved: {clf_path}")

        return self._loader

    def get_loader(self):
        return self._loader
