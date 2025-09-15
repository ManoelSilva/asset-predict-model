import logging
import os

import joblib
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict, RandomizedSearchCV

from b3.service.data_loader import B3DataLoader
from b3.exception.model import B3ModelProcessException
from b3.service.plotter import B3DashboardPlotter


class B3Model(object):
    _RANDOM_STATE = 42

    _FEATURE_SET = [
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

    def get_loader(self):
        return self._loader

    @staticmethod
    def predict(X: pd.DataFrame, model_dir: str = "models"):
        """
        Predicts BUY/SELL for new data using saved model.
        Args:
            X (pd.DataFrame): Feature data.
            model_dir (str): Directory where models are saved.
        Returns:
            pd.Series: Predicted actions (BUY/SELL).
        """
        clf_path = os.path.join(model_dir, 'b3_model.joblib')
        clf = joblib.load(clf_path)
        features = [f for f in B3Model._FEATURE_SET if f in X.columns]
        return clf.predict(X[features])

    def train_and_save(self, model_dir: str = "models", n_jobs: int = 5):
        """
        Loads data, creates BUY/SELL labels, trains a RandomForestClassifier, and saves the model.
        Args:
            model_dir (str): Directory to save models.
            n_jobs (int): Number of jobs for parallelism in training.
        """
        # Load data
        df = self._loader.fetch_all()

        # Clean validated data
        X, df = self._clean_data(self._validate_returning_x_features(df), df)

        # Classify using specified features
        self._generate_buy_signal_cluster(df)
        self._generate_sell_signal_cluster(df)
        y = self._get_target_y(df)

        # Train supervised model to predict BUY/SELL
        # Tune hyperparameters and get the best model
        clf = self._tune_hyperparameters(X, y, n_jobs=n_jobs)

        # Evaluate the model
        self._evaluate_model(clf, X, y)

        # Prepare model directory
        os.makedirs(model_dir, exist_ok=True)
        clf_path = os.path.join(model_dir, 'b3_model.joblib')

        # Save model
        joblib.dump(clf, clf_path)
        logging.info(f"Model saved: {clf_path}")

        # Visualize sample actions using plotter
        B3DashboardPlotter.plot_action_samples_by_ticker(df)

    def _tune_hyperparameters(self, X, y, n_jobs: int = 5):
        """
        Tune RandomForestClassifier hyperparameters using RandomizedSearchCV.
        Returns the best estimator.
        """
        param_dist = {
            'min_samples_leaf': [1, 5],  # Reduced grid size
            'max_features': ['sqrt', 0.5]  # Reduced grid size
        }
        rf = RandomForestClassifier(random_state=self._RANDOM_STATE)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self._RANDOM_STATE)  # Fewer folds
        random_search = RandomizedSearchCV(
            rf, param_distributions=param_dist, n_iter=4, cv=cv, scoring='f1_weighted', n_jobs=n_jobs, verbose=1,
            random_state=self._RANDOM_STATE
        )
        random_search.fit(X, y)
        print("Best parameters found:", random_search.best_params_)
        return random_search.best_estimator_

    def _evaluate_model(self, clf, X, y):
        """
        Evaluate the classifier using cross-validation and print classification report.
        """
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self._RANDOM_STATE)
        y_pred = cross_val_predict(clf, X, y, cv=cv)
        print(classification_report(y, y_pred, target_names=['buy', 'sell', 'hold']))

    @staticmethod
    def _get_target_y(df: DataFrame) -> Series:
        # Assign target labels
        df['target'] = np.select(
            [df['buy_signal_cluster'], df['sell_signal_cluster']],
            ['buy_target', 'sell_target'],
            default='hold'
        )
        return df['target']

    @staticmethod
    def _generate_sell_signal_cluster(df: DataFrame):
        # Sell: RSI > 70 and momentum negative and price at Bollinger upper with falling volume
        logging.info("Generating sell signal cluster..")
        df['sell_signal_cluster'] = (
                (df['rsi_14'] > 70) &
                (df['price_momentum_5'] < 0) &
                (df['bollinger_upper'].notnull()) &
                (df['volume_change'] < 0)
        )

    @staticmethod
    def _generate_buy_signal_cluster(df: DataFrame):
        # --- Cluster-based signal logic ---
        # Buy: RSI < 30 and MACD > 0 and volume increasing
        logging.info("Generating buy signal cluster..")
        df['buy_signal_cluster'] = (
                (df['rsi_14'] < 30) &
                (df['macd'] > 0) &
                (df['volume_change'] > 0)
        )

    @staticmethod
    def _clean_data(X: DataFrame, df: DataFrame) -> tuple[DataFrame, DataFrame]:
        # Clean data: replace infinite values with NaN and drop rows with NaN
        logging.info("Cleaning data..")
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.dropna()
        df = df.loc[X.index]  # align df with cleaned X
        return X, df

    @staticmethod
    def _validate_returning_x_features(df: DataFrame) -> DataFrame:
        features = [f for f in B3Model._FEATURE_SET if f in df.columns]
        missing_features = set(B3Model._FEATURE_SET) - set(features)
        if missing_features:
            raise B3ModelProcessException(f"Missing features in data: {missing_features}")
        return df[features]
