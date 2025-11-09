import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd

from b3.service.pipeline.model.config import ModelConfig


class BaseModel(ABC):
    """
    Abstract base class for all B3 prediction models.
    Defines the common interface that all model implementations must follow.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize the model with optional configuration.
        
        Args:
            config: Model configuration object
        """
        self.config = config
        self.model = None
        self.is_trained = False
        self.model_type = self.__class__.__name__

    @abstractmethod
    def prepare_data(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Tuple[Any, Any]:
        """
        Prepare data for training/prediction based on model requirements.
        
        Args:
            X: Feature data
            y: Target labels
            **kwargs: Additional parameters specific to model type
            
        Returns:
            Tuple of prepared X and y data
        """
        pass

    @abstractmethod
    def split_data(self, X: Any, y: Any, test_size: float = 0.2, val_size: float = 0.2) -> Tuple[Any, ...]:
        """
        Split data into train/validation/test sets.
        
        Args:
            X: Feature data
            y: Target labels
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            
        Returns:
            Tuple containing train/val/test splits
        """
        pass

    @abstractmethod
    def train_model(self, X_train: Any, y_train: Any, **kwargs) -> Any:
        """
        Train the model on provided data.
        
        Args:
            X_train: Training features
            y_train: Training targets
            **kwargs: Additional training parameters
            
        Returns:
            Trained model object
        """
        pass

    @abstractmethod
    def predict(self, X: Any) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature data for prediction
            
        Returns:
            Array of predictions
        """
        pass

    @abstractmethod
    def save_model(self, model: Any, model_dir: str) -> str:
        """
        Save the trained model to storage.
        
        Args:
            model: Trained model object
            model_dir: Directory to save the model
            
        Returns:
            Path to saved model
        """
        pass

    @abstractmethod
    def load_model(self, model_path: str) -> Any:
        """
        Load a saved model from storage.
        
        Args:
            model_path: Path to saved model
            
        Returns:
            Loaded model object
        """
        pass

    def evaluate_model(self, model: Any, X_val: Any, y_val: Any, X_test: Any, y_test: Any,
                       df_processed: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Evaluate the trained model (default implementation).
        Can be overridden by subclasses for model-specific evaluation.
        
        Args:
            model: Trained model
            X_val: Validation features
            y_val: Validation targets
            X_test: Test features
            y_test: Test targets
            df_processed: Processed dataframe for visualization
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary containing evaluation results
        """
        # Default implementation - can be overridden
        logging.info(f"Evaluating {self.model_type} model")
        return {"model_type": self.model_type, "evaluation": "completed"}

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model_type": self.model_type,
            "is_trained": self.is_trained,
            "config": self.config.to_dict() if self.config else None
        }

    @staticmethod
    def validate_data(X: Any, y: Any = None) -> bool:
        """
        Validate input data format and requirements.
        
        Args:
            X: Feature data
            y: Target data (optional)
            
        Returns:
            True if data is valid, False otherwise
        """
        if X is None or (isinstance(X, (pd.DataFrame, np.ndarray)) and len(X) == 0):
            logging.error("Invalid input data: X is None or empty")
            return False
        return True
