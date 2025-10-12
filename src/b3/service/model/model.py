import logging

import pandas as pd
from asset_model_data_storage.data_storage_service import DataStorageService

from b3.service.data.db.b3_featured.data_loader import B3DataLoader
from b3.service.data.db.b3_featured.data_loading_service import B3DataLoadingService
from b3.service.model.model_evaluation_service import B3ModelEvaluationService
from b3.service.model.model_preprocessing_service import B3ModelPreprocessingService
from b3.service.model.model_saving_service import B3ModelSavingService
from b3.service.model.model_training_service import B3ModelTrainingService
from b3.service.model.lstm_mtl_training_service import B3LSTMMultiTaskTrainingService, LSTMMultiTaskConfig
from b3.service.model.keras_model_saving_service import B3KerasModelSavingService
from constants import FEATURE_SET


class B3Model(object):
    def __init__(self, storage_service: DataStorageService = None) -> None:
        self._loader = B3DataLoader()
        # Initialize storage service
        self.storage_service = storage_service or DataStorageService()
        # Initialize service classes with shared storage service
        self.data_loading_service = B3DataLoadingService()
        self.preprocessing_service = B3ModelPreprocessingService()
        self.training_service = B3ModelTrainingService()
        self.evaluation_service = B3ModelEvaluationService(self.storage_service)
        self.saving_service = B3ModelSavingService(self.storage_service)
        self.keras_saving_service = B3KerasModelSavingService(self.storage_service)
        self.lstm_mtl_training_service = B3LSTMMultiTaskTrainingService()

    def get_loader(self):
        return self._loader

    def predict(self, X: pd.DataFrame, model_dir: str = "models"):
        """
        Predicts BUY/SELL for new data using saved model.
        Args:
            X (pd.DataFrame): Feature data.
            model_dir (str): Directory where models are saved.
        Returns:
            pd.Series: Predicted actions (BUY/SELL).
        """
        clf_path = self.saving_service.get_model_path(model_dir)
        clf = self.saving_service.load_model(clf_path)
        features = [f for f in FEATURE_SET if f in X.columns]
        return clf.predict(X[features])

    def train(self, model_dir: str = "models", n_jobs: int = 5, test_size: float = 0.8, val_size: float = 0.2,
              model_type: str = "rf", lookback: int = 32, horizon: int = 1, epochs: int = 25,
              batch_size: int = 128, lr: float = 1e-3, units: int = 96, dropout: float = 0.2,
              price_col: str = "close"):
        """
        Loads data, creates BUY/SELL labels, trains a RandomForestClassifier, and saves the model.
        Uses the new service classes for modular training pipeline.
        Args:
            model_dir (str): Directory to save models.
            n_jobs (int): Number of jobs for parallelism in training.
            test_size (float): Proportion of data to use for testing.
            val_size (float): Proportion of training data to use for validation.
        """
        # Step 1: Load data using data loading service
        df = self.data_loading_service.load_data()

        # Step 2: Preprocess data using preprocessing service
        X, df_processed, y = self.preprocessing_service.preprocess_data(df)

        if model_type == "rf":
            # Split data using training service
            X_train, X_val, X_test, y_train, y_val, y_test = self.training_service.split_data(
                X, y, test_size, val_size
            )
            # Train RF
            clf = self.training_service.train_model(X_train, y_train, n_jobs=n_jobs)
            # Evaluate
            self.evaluation_service.evaluate_model_comprehensive(
                clf, X_val, y_val, X_test, y_test, df_processed
            )
            # Save
            model_path = self.saving_service.save_model(clf, model_dir)
            logging.info(f"Training completed successfully. Model saved at: {model_path}")
            return

        if model_type == "lstm_mtl":
            # Build sequences for MTL
            X_seq, yA_seq, yR_seq, p0_seq, pf_seq = self.lstm_mtl_training_service.build_sequences(
                X, y, df_processed, price_col=price_col, lookback=lookback, horizon=horizon
            )
            if len(X_seq) == 0:
                raise ValueError("No sequences built for LSTM MTL. Check ordering, ticker column, and lookback/horizon.")

            # Split sequences
            (X_train, X_val, X_test,
             yA_train, yA_val, yA_test,
             yR_train, yR_val, yR_test,
             p0_train, p0_val, p0_test,
             pf_train, pf_val, pf_test) = self.lstm_mtl_training_service.split_sequences(
                X_seq, yA_seq, yR_seq, p0_seq, pf_seq, test_size, val_size
            )

            # Train model
            cfg = LSTMMultiTaskConfig(lookback=lookback, horizon=horizon, units=units,
                                      dropout=dropout, learning_rate=lr, epochs=epochs, batch_size=batch_size)
            clf = self.lstm_mtl_training_service.train_model(
                X_train, yA_train, yR_train, lookback=lookback, n_features=X.shape[1], config=cfg
            )

            # Evaluate classification using existing method
            self.evaluation_service.evaluate_model_comprehensive(
                clf, X_val, yA_val, X_test, yA_test, df_processed, model_name="b3_lstm_mtl"
            )

            # Evaluate regression (validation and test)
            # Prepare predicted prices
            ret_val_pred = clf.predict_return(X_val)
            ret_test_pred = clf.predict_return(X_test)
            price_val_pred = p0_val * (1.0 + ret_val_pred)
            price_test_pred = p0_test * (1.0 + ret_test_pred)
            reg_val = self.evaluation_service.evaluate_regression(pf_val, price_val_pred)
            reg_test = self.evaluation_service.evaluate_regression(pf_test, price_test_pred)
            logging.info(f"Validation price regression: {reg_val}")
            logging.info(f"Test price regression: {reg_test}")

            # Save Keras model
            model_path = self.keras_saving_service.save_model(clf.model, model_dir)
            logging.info(f"LSTM-MTL training completed successfully. Model saved at: {model_path}")
            return

        raise ValueError(f"Unknown model_type: {model_type}")

    def predict_lstm_price(self, X_window: pd.DataFrame, model_dir: str = "models",
                            lookback: int = 32, price_col: str = "close"):
        """
        Predict next-step price using saved MTL Keras model from a single asset window.
        X_window must be the last `lookback` ordered rows of features including `price_col` for price mapping.
        """
        keras_path = self.keras_saving_service.get_model_path(model_dir)
        keras_model = self.keras_saving_service.load_model(keras_path)
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
