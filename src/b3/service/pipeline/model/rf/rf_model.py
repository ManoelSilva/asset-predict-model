import logging
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from asset_model_data_storage.data_storage_service import DataStorageService
from sklearn.ensemble import RandomForestClassifier

from b3.service.pipeline.model.model import BaseModel
from b3.service.pipeline.model.rf.rf_config import RandomForestConfig
from b3.service.pipeline.model_evaluation_service import B3ModelEvaluationService
from b3.service.pipeline.persist.rf_persist_service import RandomForestPersistService
from b3.service.pipeline.training.model_training_service import B3ModelTrainingService


class RandomForestModel(BaseModel):
    """
    Random Forest implementation for B3 asset prediction.
    """

    def __init__(self, config: Optional[RandomForestConfig] = None,
                 storage_service: Optional[DataStorageService] = None):
        """
        Initialize Random Forest pipeline.
        
        Args:
            config: Random Forest configuration
            storage_service: Data storage service for saving/loading models
        """
        super().__init__(config or RandomForestConfig())
        self.storage_service = storage_service or DataStorageService()
        self.training_service = B3ModelTrainingService()
        self.saving_service = RandomForestPersistService(self.storage_service)
        self.evaluation_service = B3ModelEvaluationService(self.storage_service)

    def prepare_data(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for Random Forest training.
        For RF, data preparation is minimal - just ensure proper format.
        
        Args:
            X: Feature data
            y: Target labels
            **kwargs: Additional parameters (not used for RF)
            
        Returns:
            Tuple of prepared X and y data
        """
        if not self.validate_data(X, y):
            raise ValueError("Invalid input data for Random Forest")

        # Ensure proper data types
        X_prepared = X.copy()
        y_prepared = y.copy()

        # Convert any non-numeric columns to numeric if possible
        for col in X_prepared.columns:
            if X_prepared[col].dtype == 'object':
                try:
                    X_prepared[col] = pd.to_numeric(X_prepared[col], errors='coerce')
                except:
                    logging.warning(f"Could not convert column {col} to numeric")

        # Remove any rows with NaN values
        mask = ~(X_prepared.isna().any(axis=1) | y_prepared.isna())
        X_prepared = X_prepared[mask]
        y_prepared = y_prepared[mask]

        logging.info(f"Prepared data for Random Forest: {X_prepared.shape[0]} samples, {X_prepared.shape[1]} features")
        return X_prepared, y_prepared

    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, val_size: float = 0.2) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Split data into train/validation/test sets for Random Forest.
        
        Args:
            X: Feature data
            y: Target labels
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            
        Returns:
            Tuple containing (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        return self.training_service.split_data(X, y, test_size, val_size)

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> RandomForestClassifier:
        """
        Train Random Forest pipeline.
        
        Args:
            X_train: Training features
            y_train: Training targets
            **kwargs: Additional training parameters
            
        Returns:
            Trained RandomForestClassifier
        """
        n_jobs = kwargs.get('n_jobs', self.config.n_jobs)

        logging.info(f"Training Random Forest pipeline with {n_jobs} jobs")
        model = self.training_service.train_model(X_train, y_train, n_jobs=n_jobs)

        self.model = model
        self.is_trained = True

        logging.info("Random Forest training completed successfully")
        return model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using trained Random Forest pipeline.
        
        Args:
            X: Feature data for prediction
            
        Returns:
            Array of predictions
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")

        if not self.validate_data(X):
            raise ValueError("Invalid input data for prediction")

        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities from Random Forest pipeline.
        
        Args:
            X: Feature data for prediction
            
        Returns:
            Array of prediction probabilities
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")

        if not self.validate_data(X):
            raise ValueError("Invalid input data for prediction")

        return self.model.predict_proba(X)

    def save_model(self, model: RandomForestClassifier, model_dir: str) -> str:
        """
        Save Random Forest pipeline to storage.
        
        Args:
            model: Trained RandomForestClassifier
            model_dir: Directory to save the pipeline
            
        Returns:
            Path to saved pipeline
        """
        return self.saving_service.save_model(model, model_dir)

    def load_model(self, model_path: str) -> RandomForestClassifier:
        """
        Load Random Forest pipeline from storage.
        
        Args:
            model_path: Path to saved pipeline
            
        Returns:
            Loaded RandomForestClassifier
        """
        model = self.saving_service.load_model(model_path)
        self.model = model
        self.is_trained = True
        return model

    def evaluate_model(self, model: RandomForestClassifier, X_val: pd.DataFrame, y_val: pd.Series,
                       X_test: pd.DataFrame, y_test: pd.Series, df_processed: pd.DataFrame,
                       **kwargs) -> Dict[str, Any]:
        """
        Evaluate Random Forest pipeline.
        
        Args:
            model: Trained RandomForestClassifier
            X_val: Validation features
            y_val: Validation targets
            X_test: Test features
            y_test: Test targets
            df_processed: Processed dataframe for visualization
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary containing evaluation results
        """
        model_name = kwargs.get('model_name', 'b3_random_forest')
        persist_results = kwargs.get('persist_results', True)

        return self.evaluation_service.evaluate_model_comprehensive(
            model, X_val, y_val, X_test, y_test, df_processed,
            model_name=model_name, persist_results=persist_results
        )

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance from trained Random Forest pipeline.
        
        Returns:
            Array of feature importances or None if pipeline not trained
        """
        if not self.is_trained or self.model is None:
            return None
        return self.model.feature_importances_
