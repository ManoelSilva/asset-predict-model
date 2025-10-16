import logging

import pandas as pd
from asset_model_data_storage.data_storage_service import DataStorageService

from b3.service.data.db.b3_featured.data_loader import B3DataLoader
from b3.service.data.db.b3_featured.data_loading_service import DataLoadingService
from b3.service.pipeline.model.factory import ModelFactory
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
        Predicts BUY/SELL for new data using saved pipeline.
        Args:
            X (pd.DataFrame): Feature data.
            model_dir (str): Directory where models are saved.
            model_type (str): Type of pipeline to use for prediction ("rf", "random_forest", "lstm", "lstm_mtl").
        Returns:
            pd.Series: Predicted actions (BUY/SELL).
        """
        try:
            # Create pipeline instance using factory
            model = ModelFactory.get_model(model_type)

            # Get appropriate saving service for pipeline type
            saving_service = ModelFactory.get_persist_service(model_type, self.storage_service)
            model_path = saving_service.get_model_path(model_dir)

            # Load pipeline
            clf = model.load_model(model_path)

            features = [f for f in FEATURE_SET if f in X.columns]
            return clf.predict(X[features])
        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            raise

    def train(self, model_dir: str = "models", n_jobs: int = 5, test_size: float = 0.8, val_size: float = 0.2,
              model_type: str = "rf", lookback: int = 32, horizon: int = 1, epochs: int = 25,
              batch_size: int = 128, lr: float = 1e-3, units: int = 96, dropout: float = 0.2,
              price_col: str = "close"):
        """
        Loads data, creates BUY/SELL labels, trains a pipeline using the factory pattern, and saves the pipeline.
        Uses the new service classes for modular training pipeline.
        Args:
            model_dir (str): Directory to save models.
            n_jobs (int): Number of jobs for parallelism in training.
            test_size (float): Proportion of data to use for testing.
            val_size (float): Proportion of training data to use for validation.
            model_type (str): Type of pipeline to train ("rf", "random_forest", "lstm", "lstm_mtl").
            lookback (int): Number of lookback steps for LSTM.
            horizon (int): Prediction horizon for LSTM.
            epochs (int): Number of training epochs for LSTM.
            batch_size (int): Batch size for LSTM training.
            lr (float): Learning rate for LSTM.
            units (int): Number of LSTM units.
            dropout (float): Dropout rate for LSTM.
            price_col (str): Price column name for LSTM.
        """
        # Step 1: Load data using data loading service
        df = self.data_loading_service.load_data()

        # Step 2: Preprocess data using preprocessing service
        X, df_processed, y = self.preprocessing_service.preprocess_data(df)

        # Step 3: Create pipeline using factory pattern with automatic rf creation
        try:
            # Prepare kwargs for configuration
            config_kwargs = {
                'n_jobs': n_jobs,
                'lookback': lookback,
                'horizon': horizon,
                'units': units,
                'dropout': dropout,
                'learning_rate': lr,
                'epochs': epochs,
                'batch_size': batch_size,
                'price_col': price_col
            }

            model = ModelFactory.get_model(model_type, **config_kwargs)
        except ValueError as e:
            logging.error(f"Error creating pipeline: {str(e)}")
            raise

        # Step 4: Train pipeline using unified interface
        model_path = self._train_model_unified(
            model, model_type, X, y, df_processed, test_size, val_size,
            model_dir, n_jobs, lookback, horizon, price_col
        )

        return model_path

    @staticmethod
    def _train_model_unified(model, model_type: str, X, y, df_processed,
                             test_size: float, val_size: float, model_dir: str,
                             n_jobs: int, lookback: int, horizon: int, price_col: str) -> str:
        """
        Unified training method that handles both Random Forest and LSTM models.
        
        Args:
            model: Model instance created by factory
            model_type: Type of pipeline being trained
            X: Feature data
            y: Target labels
            df_processed: Processed dataframe
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            model_dir: Directory to save pipeline
            n_jobs: Number of parallel jobs
            lookback: Lookback steps for LSTM
            horizon: Prediction horizon for LSTM
            price_col: Price column name
            
        Returns:
            Path to saved pipeline
        """
        if model_type in ["rf", "random_forest"]:
            # Random Forest training flow
            X_prepared, y_prepared = model.prepare_data(X, y)
            X_train, X_val, X_test, y_train, y_val, y_test = model.split_data(
                X_prepared, y_prepared, test_size, val_size
            )
            trained_model = model.train_model(X_train, y_train, n_jobs=n_jobs)
            evaluation_results = model.evaluate_model(
                trained_model, X_val, y_val, X_test, y_test, df_processed
            )
            model_path = model.save_model(trained_model, model_dir)
            logging.info(f"Random Forest training completed successfully. Model saved at: {model_path}")

        elif model_type in ["lstm", "lstm_mtl"]:
            # LSTM training flow
            X_seq, yA_seq, yR_seq, p0_seq, pf_seq = model.prepare_data(
                X, y, df_processed, lookback=lookback, horizon=horizon, price_col=price_col
            )

            (X_train, X_val, X_test,
             yA_train, yA_val, yA_test,
             yR_train, yR_val, yR_test,
             p0_train, p0_val, p0_test,
             pf_train, pf_val, pf_test) = model.split_data(
                X_seq, yA_seq, yR_seq, p0_seq, pf_seq, test_size, val_size
            )

            trained_model = model.train_model(
                X_train, yA_train, yR_train, lookback=lookback, n_features=X.shape[1]
            )

            evaluation_results = model.evaluate_model(
                trained_model, X_val, yA_val, X_test, yA_test, df_processed,
                p0_val=p0_val, pf_val=pf_val, p0_test=p0_test, pf_test=pf_test
            )

            model_path = model.save_model(trained_model, model_dir)
            logging.info(f"LSTM-MTL training completed successfully. Model saved at: {model_path}")

        else:
            raise ValueError(f"Unsupported pipeline type: {model_type}")

        return model_path

    def predict_lstm_price(self, X_window: pd.DataFrame, model_dir: str = "models",
                           lookback: int = 32, price_col: str = "close"):
        """
        Predict next-step price using saved MTL Keras pipeline from a single asset window.
        X_window must be the last `lookback` ordered rows of features including `price_col` for price mapping.
        """
        try:
            # Get appropriate saving service for LSTM
            saving_service = ModelFactory.get_persist_service("lstm", self.storage_service)
            model_path = saving_service.get_model_path(model_dir)

            # Load pipeline
            keras_model = saving_service.load_model(model_path)

            features = [f for f in FEATURE_SET if f in X_window.columns]
            Xw = X_window[features].values.astype("float32")
            if len(Xw) < lookback:
                raise ValueError(f"Need at least {lookback} rows to predict with LSTM MTL")
            X_seq = Xw[-lookback:][None, :, :]

            # Predict return head
            outs = keras_model.predict(X_seq, verbose=0)
            ret_pred = outs["ret"] if isinstance(outs, dict) else outs[1]
            ret_val = float(ret_pred.reshape(-1)[0])
            last_price = float(X_window[price_col].iloc[-1])
            return last_price * (1.0 + ret_val)
        except Exception as e:
            logging.error(f"Error in LSTM price prediction: {str(e)}")
            raise
