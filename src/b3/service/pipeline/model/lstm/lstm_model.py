import logging
import os
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from asset_model_data_storage.data_storage_service import DataStorageService

from b3.service.pipeline.model.lstm.lstm_scale import LSTMScaler
from b3.service.pipeline.model.lstm.pytorch_mtl_model import B3PytorchMTLModel
from b3.service.pipeline.model.lstm.lstm_config import LSTMConfig
from b3.service.pipeline.model.lstm.lstm_sequence_builder import LSTMSequenceBuilder
from b3.service.pipeline.model.lstm.lstm_data_splitter import LSTMDataSplitter
from b3.service.pipeline.model.lstm.lstm_prediction_enricher import LSTMPredictionEnricher
from b3.service.pipeline.model.lstm.lstm_model_loader import LSTMModelLoader
from b3.service.pipeline.model.model import BaseModel
from b3.service.pipeline.model.params import SplitDataParams, TrainModelParams, PrepareDataParams
from b3.service.pipeline.model.utils import get_device_str
from b3.service.pipeline.model_evaluation_service import B3ModelEvaluationService
from b3.service.pipeline.persist.lstm_persist_service import LSTMPersistService


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
        self._device = torch.device(get_device_str())

        self._sequence_builder = LSTMSequenceBuilder(self._device)
        self._data_splitter = LSTMDataSplitter()
        self._prediction_enricher = LSTMPredictionEnricher(self._sequence_builder, self._config)
        self._model_loader = LSTMModelLoader(self._saving_service, self._config, self._device)
        self._scaler = LSTMScaler()

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

        # Prepare
        lstm_config_dict = config.get_lstm_config_dict()
        prepare_params = PrepareDataParams(
            X=X,
            y=y,
            df_processed=df_processed
        )
        X_seq, yA_seq, yR_seq, p0_seq, pf_seq = self.prepare_data(
            prepare_params, **lstm_config_dict
        )

        # Split
        split_params = SplitDataParams(
            X=X_seq,
            y=yA_seq,
            test_size=config.test_size,
            val_size=config.val_size,
            yR=yR_seq,
            p0=p0_seq,
            pf=pf_seq
        )
        (X_train, X_val, X_test,
         yA_train, yA_val, yA_test,
         yR_train, yR_val, yR_test,
         p0_train, p0_val, p0_test,
         pf_train, pf_val, pf_test) = self.split_data(split_params)

        # Scale Features
        logging.info("Fitting StandardScaler on training data...")
        # Reset scaler for new training
        self._scaler.reset()

        # Fit and Transform Train
        X_train = self._scaler.fit_transform(X_train)

        # Transform Val
        if X_val.shape[0] > 0:
            X_val = self._scaler.transform(X_val)

        # Transform Test
        if X_test.shape[0] > 0:
            X_test = self._scaler.transform(X_test)

        # Train
        train_params = TrainModelParams(
            X_train=X_train,
            y_train=yA_train,
            yR_train=yR_train,
            X_val=X_val,
            y_val=yA_val,
            yR_val=yR_val
        )
        trained_model = self.train_model(
            train_params, lookback=config.lookback, n_features=X.shape[1]
        )

        # Evaluate
        evaluation_results = self.evaluate_model(
            trained_model, X_val, yA_val, X_test, yA_test, df_processed,
            p0_val=p0_val, pf_val=pf_val, p0_test=p0_test, pf_test=pf_test
        )

        # Add training history to evaluation results
        if hasattr(trained_model, 'history'):
            evaluation_results['training_history'] = trained_model.history

        # Save Model & Scaler
        model_path = self.save_model(trained_model, config.model_dir)

        logging.info(f"LSTM-MTL training completed successfully. Model saved at: {model_path}")

        return model_path, evaluation_results

    def prepare_data(self, params: PrepareDataParams, **kwargs) -> Tuple[
        np.ndarray, pd.Series, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM training by building sequences.
        
        Args:
            params: PrepareDataParams object containing all parameters for data preparation
            **kwargs: Additional parameters
            
        Returns:
            Tuple containing (X_seq, y_action_seq, y_return_seq, last_price_seq, future_price_seq)
        """
        if not self.validate_data(params.X, params.y):
            raise ValueError("Invalid input data for LSTM")

        if not params.has_lstm_params() or params.df_processed is None:
            raise ValueError("df_processed is required for LSTM sequence building")

        lookback = kwargs.get('lookback', self._config.lookback)
        horizon = kwargs.get('horizon', self._config.horizon)
        price_col = kwargs.get('price_col', self._config.price_col)

        logging.info(f"Building LSTM sequences with lookback={lookback}, horizon={horizon}")

        X_seq, yA_seq, yR_seq, p0_seq, pf_seq = self._sequence_builder.build_sequences(
            params.X, params.y, params.df_processed, price_col=price_col, lookback=lookback, horizon=horizon
        )

        if len(X_seq) == 0:
            raise ValueError("No sequences built for LSTM. Check ordering, ticker column, and lookback/horizon.")

        logging.info(f"Built {len(X_seq)} sequences for LSTM training")
        return X_seq, yA_seq, yR_seq, p0_seq, pf_seq

    def split_data(self, params: SplitDataParams) -> Tuple[np.ndarray, ...]:
        """
        Split LSTM sequences into train/validation/test sets.
        
        Args:
            params: SplitDataParams object containing all parameters for data splitting
            
        Returns:
            Tuple containing all train/val/test splits
        """
        if not params.has_lstm_params():
            raise ValueError("LSTM split_data requires LSTM-specific parameters (yR, p0, pf)")

        return self._data_splitter.split_sequences(
            params.X, params.y, params.yR, params.p0, params.pf, params.test_size, params.val_size
        )

    def train_model(self, params: TrainModelParams, **kwargs) -> B3PytorchMTLModel:
        """
        Train LSTM Multi-Task Learning model.
        
        Args:
            params: TrainModelParams object containing all parameters for model training
            **kwargs: Additional training parameters
            
        Returns:
            Trained B3PytorchMTLModel
        """
        if not params.has_lstm_params():
            raise ValueError("LSTM train_model requires LSTM-specific parameters (yR_train)")

        lookback = kwargs.get('lookback', self._config.lookback)
        n_features = kwargs.get('n_features', params.X_train.shape[2])

        if not self._scaler.is_fitted():
            logging.info("Scaler not fitted (Standalone Training). Fitting StandardScaler on training data...")
            params.X_train = self._scaler.fit_transform(params.X_train)

            # Transform Val if present
            if params.X_val is not None and params.X_val.shape[0] > 0:
                params.X_val = self._scaler.transform(params.X_val)

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
        clf.fit(params.X_train, params.y_train, params.yR_train,
                X_val=params.X_val, y_val=params.y_val, yR_val=params.yR_val)

        logging.info("LSTM multi-task training completed")

        self.model = clf
        self.is_trained = True

        return clf

    def _scale_input(self, X: np.ndarray) -> np.ndarray:
        """Helper to scale 3D input sequences."""
        return self._scaler.transform(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make action predictions using trained LSTM model.
        
        Args:
            X: Feature sequences for prediction
            
        Returns:
            Array of action predictions
        """
        self._validate_for_prediction(X)
        X_scaled = self._scale_input(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Make probability predictions using trained LSTM model.
        
        Args:
            X: Feature sequences for prediction
            
        Returns:
            Array of probability predictions
        """
        self._validate_for_prediction(X)
        X_scaled = self._scale_input(X)
        return self.model.predict_proba(X_scaled)

    def predict_return(self, X: np.ndarray) -> np.ndarray:
        """
        Make return predictions using trained LSTM model.
        
        Args:
            X: Feature sequences for prediction
            
        Returns:
            Array of return predictions
        """
        self._validate_for_prediction(X)
        X_scaled = self._scale_input(X)
        return self.model.predict_return(X_scaled)

    def save_model(self, model: B3PytorchMTLModel, model_dir: str) -> str:
        model_path = self._saving_service.save_model(model.model, model_dir)

        # Save Scaler
        if self._scaler.is_fitted():
            model_filename = os.path.basename(model_path)
            scaler_name = model_filename.replace(".pt", "_scaler.joblib")
            if scaler_name == model_filename:
                scaler_name = f"{model_filename}_scaler.joblib"

            self._saving_service.save_scaler(self._scaler.scaler, model_dir, scaler_name)
            logging.info(f"Saved scaler to {scaler_name}")

        return model_path

    def load_model(self, model_path: str) -> B3PytorchMTLModel:
        model = self._model_loader.load_model(model_path)
        self.model = model
        self.is_trained = True

        # Derive scaler path from model path to ensure we load the matching scaler
        model_dir = os.path.dirname(model_path)
        model_filename = os.path.basename(model_path)

        scaler_name = model_filename.replace(".pt", "_scaler.joblib")

        if scaler_name == model_filename:
            scaler_name = f"{model_filename}_scaler.joblib"

        try:
            scaler = self._saving_service.load_scaler(model_dir=model_dir, scaler_name=scaler_name)
            logging.info(f"Loaded scaler {scaler_name} from {model_dir}")
        except FileNotFoundError:
            logging.warning(f"Scaler {scaler_name} not found in {model_dir}, trying default scaler name")
            scaler = self._saving_service.load_scaler(model_dir=model_dir)

        self._scaler = LSTMScaler(scaler)

        return model

    def evaluate_model(self, model: B3PytorchMTLModel, X_val: np.ndarray, yA_val: pd.Series,
                       X_test: np.ndarray, yA_test: pd.Series, df_processed: pd.DataFrame,
                       p0_val: np.ndarray = None, pf_val: np.ndarray = None,
                       p0_test: np.ndarray = None, pf_test: np.ndarray = None,
                       **kwargs) -> Dict[str, Any]:
        """
        Evaluate LSTM model with both classification and regression metrics.
        """
        model_name = kwargs.get('model_name', 'b3_lstm_mtl')
        persist_results = kwargs.get('persist_results', True)

        # Enrich dataframe with predictions for visualization
        df_viz = df_processed.copy()
        self._prediction_enricher.enrich_df_with_predictions(df_viz, model, max_samples=100, scaler=self._scaler.scaler)

        # Evaluate classification
        classification_results = self._evaluation_service.evaluate_model_comprehensive(
            model, X_val, yA_val, X_test, yA_test, df_viz,
            model_name=model_name, persist_results=persist_results
        )

        # Evaluate regression if price data is available
        regression_results = {}
        if p0_val is not None and pf_val is not None and p0_test is not None and pf_test is not None:
            ret_val_pred = model.predict_return(X_val)
            ret_test_pred = model.predict_return(X_test)

            price_val_pred = p0_val * (1.0 + ret_val_pred)
            price_test_pred = p0_test * (1.0 + ret_test_pred)

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
                    if 'validation' in classification_results and 'evaluation_id' in classification_results[
                        'validation']:
                        val_id = classification_results['validation']['evaluation_id']
                        self._evaluation_service.update_evaluation_metrics(val_id, {'price_regression': reg_val})
                        logging.info(f"Updated validation evaluation {val_id} with regression metrics")

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

    def _validate_for_prediction(self, X: np.ndarray) -> None:
        """
        Validate model and input data before making predictions.
        
        Args:
            X: Feature sequences for prediction
            
        Raises:
            ValueError: If model is not trained or data is invalid
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")

        if not self.validate_data(X):
            raise ValueError("Invalid input data for prediction")
