import logging
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from asset_model_data_storage.data_storage_service import DataStorageService

from b3.service.pipeline.model.lstm.lstm_config import LSTMConfig
from b3.service.pipeline.model.model import BaseModel
from b3.service.pipeline.model_evaluation_service import B3ModelEvaluationService
from b3.service.pipeline.persist.lstm_persist_service import LSTMPersistService
from b3.service.pipeline.training.lstm_mtl_training_service import B3LSTMMultiTaskTrainingService, LSTMMultiTaskConfig, \
    B3KerasMTLModel


class LSTMModel(BaseModel):
    """
    LSTM Multi-Task Learning implementation for B3 asset prediction.
    """

    def __init__(self, config: Optional[LSTMConfig] = None,
                 storage_service: Optional[DataStorageService] = None):
        """
        Initialize LSTM pipeline.
        
        Args:
            config: LSTM configuration
            storage_service: Data storage service for saving/loading models
        """
        super().__init__(config or LSTMConfig())
        self.storage_service = storage_service or DataStorageService()
        self.training_service = B3LSTMMultiTaskTrainingService()
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

        X_seq, yA_seq, yR_seq, p0_seq, pf_seq = self.training_service.build_sequences(
            X, y, df_processed, price_col=price_col, lookback=lookback, horizon=horizon
        )

        if len(X_seq) == 0:
            raise ValueError("No sequences built for LSTM. Check ordering, ticker column, and lookback/horizon.")

        logging.info(f"Built {len(X_seq)} sequences for LSTM training")
        return X_seq, yA_seq, yR_seq, p0_seq, pf_seq

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
        return self.training_service.split_sequences(
            X_seq, yA_seq, yR_seq, p0_seq, pf_seq, test_size, val_size
        )

    def train_model(self, X_train: np.ndarray, yA_train: pd.Series, yR_train: np.ndarray, **kwargs) -> B3KerasMTLModel:
        """
        Train LSTM Multi-Task Learning pipeline.
        
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

        # Create LSTM rf from pipeline rf
        lstm_config = LSTMMultiTaskConfig(
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

        logging.info(f"Training LSTM pipeline with {lstm_config.epochs} epochs, batch_size={lstm_config.batch_size}")

        model = self.training_service.train_model(
            X_train, yA_train, yR_train, lookback=lookback, n_features=n_features, config=lstm_config
        )

        self.model = model
        self.is_trained = True

        logging.info("LSTM training completed successfully")
        return model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make action predictions using trained LSTM pipeline.
        
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
        Make return predictions using trained LSTM pipeline.
        
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
        Predict next-step price using trained LSTM pipeline from a single asset window.
        
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
        Save LSTM pipeline to storage.
        
        Args:
            model: Trained B3KerasMTLModel
            model_dir: Directory to save the pipeline
            
        Returns:
            Path to saved pipeline
        """
        return self.saving_service.save_model(model.model, model_dir)

    def load_model(self, model_path: str) -> B3KerasMTLModel:
        """
        Load LSTM pipeline from storage.
        
        Args:
            model_path: Path to saved pipeline
            
        Returns:
            Loaded B3KerasMTLModel
        """
        keras_model = self.saving_service.load_model(model_path)

        # Create wrapper pipeline
        model = B3KerasMTLModel(
            input_timesteps=self.config.lookback,
            input_features=keras_model.input_shape[2],  # Get feature count from loaded pipeline
            config=LSTMMultiTaskConfig(
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
        Evaluate LSTM pipeline with both classification and regression metrics.
        
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
