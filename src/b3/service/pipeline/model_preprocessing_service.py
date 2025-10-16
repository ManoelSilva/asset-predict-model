import logging

import numpy as np
from pandas import DataFrame, Series

from b3.exception.model import B3ModelProcessException
from constants import FEATURE_SET


class PreprocessingService:
    """
    Service responsible for preprocessing B3 market data including feature validation,
    data cleaning, and target label generation.
    """

    @staticmethod
    def validate_features(df: DataFrame) -> DataFrame:
        """
        Validates that all required features are present in the dataset.
        
        Args:
            df (DataFrame): Input dataframe
            
        Returns:
            DataFrame: Dataframe with only the required features
            
        Raises:
            B3ModelProcessException: If required features are missing
        """
        features = [f for f in FEATURE_SET if f in df.columns]
        missing_features = set(FEATURE_SET) - set(features)
        if missing_features:
            raise B3ModelProcessException(f"Missing features in data: {missing_features}")
        return df[features]

    @staticmethod
    def clean_data(X: DataFrame, df: DataFrame) -> tuple[DataFrame, DataFrame]:
        """
        Cleans data by replacing infinite values with NaN and dropping rows with NaN.
        
        Args:
            X (DataFrame): Feature data
            df (DataFrame): Full dataset
            
        Returns:
            tuple: (cleaned_X, aligned_df)
        """
        logging.info("Cleaning data...")
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.dropna()
        df = df.loc[X.index]  # align df with cleaned X
        return X, df

    @staticmethod
    def clean_prediction_data(x_prediction: DataFrame) -> DataFrame:
        """
        Clean prediction data by handling infinity and extreme values.
        This method is specifically designed for prediction scenarios where
        we cannot drop rows but need to handle problematic values.
        
        Args:
            x_prediction (DataFrame): Feature data for prediction
            
        Returns:
            DataFrame: Cleaned prediction data
        """
        logging.info("Cleaning prediction data...")

        # Replace infinity values with NaN
        x_prediction = x_prediction.replace([np.inf, -np.inf], np.nan)

        # Check for any remaining NaN values
        nan_columns = x_prediction.columns[x_prediction.isna().any()].tolist()
        if nan_columns:
            logging.warning(f"Found NaN values in columns: {nan_columns}")
            # For prediction, we'll fill NaN with 0 or median values
            # This is a fallback - ideally the API should provide clean data
            for col in nan_columns:
                if x_prediction[col].dtype in ['float64', 'float32']:
                    x_prediction[col] = x_prediction[col].fillna(0.0)
                else:
                    x_prediction[col] = x_prediction[col].fillna(0)

        # Check for extremely large values that might cause float32 overflow
        max_float32 = np.finfo(np.float32).max
        large_value_columns = []
        for col in x_prediction.select_dtypes(include=[np.number]).columns:
            if x_prediction[col].abs().max() > max_float32:
                large_value_columns.append(col)
                # Clip extreme values to prevent overflow
                x_prediction[col] = np.clip(x_prediction[col], -max_float32, max_float32)
                logging.warning(f"Clipped extreme values in column: {col}")

        if large_value_columns:
            logging.warning(f"Found extreme values in columns: {large_value_columns}")

        return x_prediction

    @staticmethod
    def generate_buy_signal_cluster(df: DataFrame):
        """
        Generates buy signal clusters based on technical indicators.
        Buy: RSI < 30 and MACD > 0 and volume increasing
        
        Args:
            df (DataFrame): Input dataframe
        """
        logging.info("Generating buy signal cluster...")
        df['buy_signal_cluster'] = (
                (df['rsi_14'] < 30) &
                (df['macd'] > 0) &
                (df['volume_change'] > 0)
        )

    @staticmethod
    def generate_sell_signal_cluster(df: DataFrame):
        """
        Generates sell signal clusters based on technical indicators.
        Sell: RSI > 70 and momentum negative and price at Bollinger upper with falling volume
        
        Args:
            df (DataFrame): Input dataframe
        """
        logging.info("Generating sell signal cluster...")
        df['sell_signal_cluster'] = (
                (df['rsi_14'] > 70) &
                (df['price_momentum_5'] < 0) &
                (df['bollinger_upper'].notnull()) &
                (df['volume_change'] < 0)
        )

    @staticmethod
    def get_target_labels(df: DataFrame) -> Series:
        """
        Creates target labels from buy/sell signal clusters.
        
        Args:
            df (DataFrame): Input dataframe with signal clusters
            
        Returns:
            Series: Target labels ('buy_target', 'sell_target', 'hold')
        """
        df['target'] = np.select(
            [df['buy_signal_cluster'], df['sell_signal_cluster']],
            ['buy_target', 'sell_target'],
            default='hold'
        )
        return df['target']

    def preprocess_data(self, df: DataFrame) -> tuple[DataFrame, DataFrame, Series]:
        """
        Complete preprocessing pipeline: validate features, clean data, generate signals, and create targets.
        
        Args:
            df (DataFrame): Raw input data
            
        Returns:
            tuple: (X_features, cleaned_df, y_targets)
        """
        # Validate and extract features
        X = self.validate_features(df)

        # Clean data
        X, df = self.clean_data(X, df)

        # Generate signal clusters
        self.generate_buy_signal_cluster(df)
        self.generate_sell_signal_cluster(df)

        # Create target labels
        y = self.get_target_labels(df)

        return X, df, y
