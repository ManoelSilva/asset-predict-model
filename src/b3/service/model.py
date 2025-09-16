import logging
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split

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

    def _split_data(self, X: DataFrame, y: Series, test_size: float = 0.2, val_size: float = 0.2):
        """
        Split data into train/validation/test sets with stratification.
        Args:
            X: Feature data
            y: Target labels
            test_size: Proportion of data to use for testing
            val_size: Proportion of training data to use for validation
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: separate test set
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=test_size, random_state=self._RANDOM_STATE, stratify=y
        )
        
        # Second split: separate validation set from remaining data
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=self._RANDOM_STATE, stratify=y_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test

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

    def train_and_save(self, model_dir: str = "models", n_jobs: int = 5, test_size: float = 0.2, val_size: float = 0.2):
        """
        Loads data, creates BUY/SELL labels, trains a RandomForestClassifier, and saves the model.
        Args:
            model_dir (str): Directory to save models.
            n_jobs (int): Number of jobs for parallelism in training.
            test_size (float): Proportion of data to use for testing.
            val_size (float): Proportion of training data to use for validation.
        """
        # Load data
        df = self._loader.fetch_all()

        # Clean validated data
        X, df = self._clean_data(self._validate_returning_x_features(df), df)

        # Classify using specified features
        self._generate_buy_signal_cluster(df)
        self._generate_sell_signal_cluster(df)
        y = self._get_target_y(df)

        # Split data into train/validation/test sets
        X_train, X_val, X_test, y_train, y_val, y_test = self._split_data(X, y, test_size, val_size)
        
        logging.info(f"Data split - Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

        # Train supervised model to predict BUY/SELL
        # Tune hyperparameters using only training data
        clf = self._tune_hyperparameters(X_train, y_train, n_jobs=n_jobs)

        # Evaluate the model on validation set during development
        self._evaluate_model(clf, X_val, y_val, set_name="Validation")

        # Final evaluation on test set
        self._evaluate_model(clf, X_test, y_test, set_name="Test")

        # Prepare model directory
        os.makedirs(model_dir, exist_ok=True)
        clf_path = os.path.join(model_dir, 'b3_model.joblib')

        # Save model
        joblib.dump(clf, clf_path)
        logging.info(f"Model saved: {clf_path}")

        # Visualize sample actions using plotter (use full dataset for visualization)
        evaluation_dir = os.path.join("b3", "assets", "evaluation")
        os.makedirs(evaluation_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"action_samples_by_ticker_{timestamp}.png"
        plot_path = os.path.join(evaluation_dir, plot_filename)
        B3DashboardPlotter.plot_action_samples_by_ticker(df, save_path=plot_path)

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

    def _evaluate_model(self, clf, X, y, set_name: str = "Dataset"):
        """
        Evaluate the classifier and print classification report.
        Args:
            clf: Trained classifier
            X: Feature data
            y: Target labels
            set_name: Name of the dataset being evaluated (e.g., "Train", "Validation", "Test")
        """
        y_pred = clf.predict(X)
        print(f"\n{classification_report(y, y_pred, target_names=['buy', 'sell', 'hold'])}")
        logging.info(f"Model evaluation completed on {set_name} set")

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
