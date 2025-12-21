import logging
from typing import Tuple, Dict

import pandas as pd
from asset_model_data_storage.data_storage_service import DataStorageService

from b3.service.data.db.b3_featured.data_loader import B3DataLoader
from b3.service.data.db.b3_featured.data_loading_service import DataLoadingService
from b3.service.pipeline.model.factory import ModelFactory
from b3.service.pipeline.model.training_config import TrainingConfig
from b3.service.pipeline.model.utils import is_lstm_model, is_rf_model, normalize_model_type
from b3.service.pipeline.model_evaluation_service import B3ModelEvaluationService
from b3.service.pipeline.model_preprocessing_service import PreprocessingService
from constants import FEATURE_SET


class ModelManagerService(object):
    def __init__(self, storage_service: DataStorageService = None) -> None:
        self._loader = B3DataLoader()
        # Initialize storage service
        self.storage_service = storage_service or DataStorageService()
        # Initialize service classes with shared storage service
        self.data_loading_service = DataLoadingService()
        self.preprocessing_service = PreprocessingService()
        self.evaluation_service = B3ModelEvaluationService(self.storage_service)

    def get_loader(self):
        return self._loader

    def predict(self, X: pd.DataFrame, model_dir: str = "models", model_type: str = "rf"):
        """
        Predicts BUY/SELL for new data using saved model.
        
        Args:
            X (pd.DataFrame): Feature data.
            model_dir (str): Directory where models are saved.
            model_type (str): Type of model to use for prediction ("rf", "random_forest", "lstm", "lstm_mtl").
                            Automatically normalized to canonical form.
        Returns:
            pd.Series: Predicted actions (BUY/SELL).
        """
        try:
            # Normalize model type to canonical form
            model_type = normalize_model_type(model_type)

            # Create model instance using factory
            model = ModelFactory.get_model(model_type)

            # Get appropriate saving service for model type
            saving_service = ModelFactory.get_persist_service(model_type, self.storage_service)
            model_path = saving_service.get_model_path(model_dir)

            # Load model
            clf = model.load_model(model_path)

            features = [f for f in FEATURE_SET if f in X.columns]
            return clf.predict(X[features])
        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            raise

    def train(self, config: TrainingConfig = None, **kwargs):
        """
        Loads data, creates BUY/SELL labels, trains a model using the factory pattern, and saves the model.
        Uses the new service classes for modular training.
        
        Args:
            config (TrainingConfig): Training configuration object. If None, creates default config.
            **kwargs: Optional parameters to override config values. Accepts same parameters as TrainingConfig.
        
        Returns:
            Tuple[str, Dict]: Path to saved model and evaluation results
        """
        # Create config from kwargs if provided, otherwise use provided config or default
        if config is None:
            config = TrainingConfig(**kwargs)
        else:
            # Allow overriding config values via kwargs
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        # Normalize model type to canonical form
        config.model_type = normalize_model_type(config.model_type)

        # Step 1: Load data using data loading service
        df = self.data_loading_service.load_data()

        # Step 2: Preprocess data using preprocessing service
        X, df_processed, y = self.preprocessing_service.preprocess_data(df)

        # Step 3: Create model using factory pattern with automatic config creation
        try:
            model = ModelFactory.get_model(config.model_type, **config.get_all_config_dict())
        except ValueError as e:
            logging.error(f"Error creating model: {str(e)}")
            raise

        # Step 4: Train model using unified interface
        model_path, evaluation_results = self._train_model_unified(model, config, X, y, df_processed)

        return model_path, evaluation_results

    @staticmethod
    def _train_model_unified(model, config: TrainingConfig, X, y, df_processed) -> Tuple[str, Dict]:
        """
        Unified training method that handles both Random Forest and LSTM models.
        Delegates model-specific logic to the model classes themselves.
        
        Args:
            model: Model instance created by factory
            config: Training configuration object
            X: Feature data
            y: Target labels
            df_processed: Processed dataframe
            
        Returns:
            Tuple[str, Dict]: Path to saved model and evaluation results
        """
        if is_rf_model(config.model_type):
            # Random Forest training flow - delegate to model
            X_prepared, y_prepared = model.prepare_data(X, y)
            X_train, X_val, X_test, y_train, y_val, y_test = model.split_data(
                X_prepared, y_prepared, config.test_size, config.val_size
            )
            trained_model = model.train_model(X_train, y_train, n_jobs=config.n_jobs)
            evaluation_results = model.evaluate_model(
                trained_model, X_val, y_val, X_test, y_test, df_processed
            )
            model_path = model.save_model(trained_model, config.model_dir)
            logging.info(f"Random Forest training completed successfully. Model saved at: {model_path}")

        elif is_lstm_model(config.model_type):
            # LSTM training flow - delegate to model
            lstm_config = config.get_lstm_config_dict()
            X_seq, yA_seq, yR_seq, p0_seq, pf_seq = model.prepare_data(
                X, y, df_processed, **lstm_config
            )

            (X_train, X_val, X_test,
             yA_train, yA_val, yA_test,
             yR_train, yR_val, yR_test,
             p0_train, p0_val, p0_test,
             pf_train, pf_val, pf_test) = model.split_data(
                X_seq, yA_seq, yR_seq, p0_seq, pf_seq, config.test_size, config.val_size
            )

            trained_model = model.train_model(
                X_train, yA_train, yR_train, lookback=config.lookback, n_features=X.shape[1]
            )

            evaluation_results = model.evaluate_model(
                trained_model, X_val, yA_val, X_test, yA_test, df_processed,
                p0_val=p0_val, pf_val=pf_val, p0_test=p0_test, pf_test=pf_test
            )

            model_path = model.save_model(trained_model, config.model_dir)
            logging.info(f"LSTM-MTL training completed successfully. Model saved at: {model_path}")

        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")

        return model_path, evaluation_results

    def predict_lstm_price(self, X_window: pd.DataFrame, model_dir: str = "models",
                           lookback: int = 32, price_col: str = "close"):
        """
        Predict next-step price using saved MTL Keras model from a single asset window.
        X_window must be the last `lookback` ordered rows of features including `price_col` for price mapping.
        
        This method delegates to LSTMModel.predict_price() to avoid code duplication.
        
        Args:
            X_window: DataFrame with last `lookback` ordered rows of features including `price_col`
            model_dir: Directory where the model is saved
            lookback: Number of lookback steps (default: 32)
            price_col: Price column name (default: "close")
            
        Returns:
            float: Predicted price
        """
        try:
            # Create LSTM model instance using factory
            lstm_model = ModelFactory.get_model("lstm")

            # Get appropriate saving service for LSTM
            saving_service = ModelFactory.get_persist_service("lstm", self.storage_service)
            model_path = saving_service.get_model_path(model_dir)

            # Load model using the model's load_model method
            lstm_model.load_model(model_path)

            # Delegate to model's predict_price method (eliminates code duplication)
            return lstm_model.predict_price(X_window, lookback=lookback, price_col=price_col)
        except Exception as e:
            logging.error(f"Error in LSTM price prediction: {str(e)}")
            raise
