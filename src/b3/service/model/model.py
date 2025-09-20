import logging
import pandas as pd

from b3.service.data.data_loader import B3DataLoader
from b3.service.data.data_loading_service import B3DataLoadingService
from b3.service.data.data_preprocessing_service import B3DataPreprocessingService
from b3.service.model.model_training_service import B3ModelTrainingService
from b3.service.model.model_evaluation_service import B3ModelEvaluationService
from b3.service.model.model_saving_service import B3ModelSavingService


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
        # Initialize service classes
        self.data_loading_service = B3DataLoadingService()
        self.preprocessing_service = B3DataPreprocessingService()
        self.training_service = B3ModelTrainingService()
        self.evaluation_service = B3ModelEvaluationService()
        self.saving_service = B3ModelSavingService()

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
        features = [f for f in B3Model._FEATURE_SET if f in X.columns]
        return clf.predict(X[features])

    def train(self, model_dir: str = "models", n_jobs: int = 5, test_size: float = 0.2, val_size: float = 0.2):
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

        # Step 3: Split data using training service
        X_train, X_val, X_test, y_train, y_val, y_test = self.training_service.split_data(
            X, y, test_size, val_size
        )

        # Step 4: Train model using training service
        clf = self.training_service.train_model(X_train, y_train, n_jobs=n_jobs)

        # Step 5: Evaluate model using evaluation service
        self.evaluation_service.evaluate_model_comprehensive(
            clf, X_val, y_val, X_test, y_test, df_processed
        )

        # Step 6: Save model using saving service
        model_path = self.saving_service.save_model(clf, model_dir)
        
        logging.info(f"Training completed successfully. Model saved at: {model_path}")

