import logging
from abc import ABC, abstractmethod
from typing import Any

from b3.service.pipeline.training.lstm_mtl_training_service import B3LSTMMultiTaskTrainingService
from b3.service.pipeline.training.model_training_service import B3ModelTrainingService


class BaseTrainingService(ABC):
    """
    Abstract base class for all training services.
    Defines the common interface that all training service implementations must follow.
    """

    @abstractmethod
    def train_model(self, *args, **kwargs) -> Any:
        """
        Train a model using the specific training service.
        
        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
            
        Returns:
            Trained model object
        """
        pass

    @abstractmethod
    def get_service_type(self) -> str:
        """
        Get the type identifier for this training service.
        
        Returns:
            String identifier for the service type
        """
        pass


class RandomForestTrainingService(BaseTrainingService):
    """
    Training service wrapper for Random Forest models.
    """

    def __init__(self):
        self.service = B3ModelTrainingService()

    def train_model(self, X_train, y_train, **kwargs):
        """
        Train Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            **kwargs: Additional training parameters
            
        Returns:
            Trained RandomForestClassifier
        """
        return self.service.train_model(X_train, y_train, **kwargs)

    def split_data(self, X, y, test_size=0.2, val_size=0.2):
        """
        Split data for Random Forest training.
        
        Args:
            X: Feature data
            y: Target labels
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            
        Returns:
            Tuple containing train/val/test splits
        """
        return self.service.split_data(X, y, test_size, val_size)

    def get_service_type(self) -> str:
        return "random_forest"


class LSTMTrainingService(BaseTrainingService):
    """
    Training service wrapper for LSTM models.
    """

    def __init__(self):
        self.service = B3LSTMMultiTaskTrainingService()

    def train_model(self, X_train, yA_train, yR_train, **kwargs):
        """
        Train LSTM Multi-Task Learning model.
        
        Args:
            X_train: Training feature sequences
            yA_train: Training action targets
            yR_train: Training return targets
            **kwargs: Additional training parameters
            
        Returns:
            Trained B3KerasMTLModel
        """
        return self.service.train_model(X_train, yA_train, yR_train, **kwargs)

    def build_sequences(self, X_df, y_action, df_full, **kwargs):
        """
        Build sequences for LSTM training.
        
        Args:
            X_df: Feature dataframe
            y_action: Action target series
            df_full: Full processed dataframe
            **kwargs: Additional parameters
            
        Returns:
            Tuple containing sequences and targets
        """
        return self.service.build_sequences(X_df, y_action, df_full, **kwargs)

    def split_sequences(self, X, y_action, y_return, last_price, future_price, test_size, val_size):
        """
        Split LSTM sequences into train/validation/test sets.
        
        Args:
            X: Feature sequences
            y_action: Action target sequences
            y_return: Return target sequences
            last_price: Initial price sequences
            future_price: Future price sequences
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            
        Returns:
            Tuple containing all train/val/test splits
        """
        return self.service.split_sequences(X, y_action, y_return, last_price, future_price, test_size, val_size)

    def get_service_type(self) -> str:
        return "lstm"


class TrainingServiceFactory:
    """
    Factory class for creating training service instances based on model type.
    """

    _service_registry = {}

    @classmethod
    def register_service(cls, model_type: str, service_class: type):
        """
        Register a training service class for a specific model type.
        
        Args:
            model_type: String identifier for the model type
            service_class: Training service class to register
        """
        if not issubclass(service_class, BaseTrainingService):
            raise ValueError(f"Service class {service_class} must inherit from BaseTrainingService")
        cls._service_registry[model_type] = service_class
        logging.info(f"Registered training service type '{model_type}' with class {service_class.__name__}")

    @classmethod
    def get_training_service(cls, model_type: str) -> BaseTrainingService:
        """
        Create a training service instance based on model type.
        
        Args:
            model_type: Type of model to create training service for
            
        Returns:
            Instance of the requested training service type
            
        Raises:
            ValueError: If model_type is not registered
        """
        if model_type not in cls._service_registry:
            available_types = list(cls._service_registry.keys())
            raise ValueError(f"Unknown model type '{model_type}'. Available types: {available_types}")

        service_class = cls._service_registry[model_type]
        return service_class()

    @classmethod
    def get_available_services(cls) -> list:
        """
        Get list of available training service types.
        
        Returns:
            List of registered service type strings
        """
        return list(cls._service_registry.keys())

    @classmethod
    def is_service_registered(cls, model_type: str) -> bool:
        """
        Check if a training service type is registered.
        
        Args:
            model_type: Service type to check
            
        Returns:
            True if service type is registered, False otherwise
        """
        return model_type in cls._service_registry


# Register the training services with the factory
TrainingServiceFactory.register_service("rf", RandomForestTrainingService)
TrainingServiceFactory.register_service("random_forest", RandomForestTrainingService)
TrainingServiceFactory.register_service("lstm", LSTMTrainingService)
TrainingServiceFactory.register_service("lstm_mtl", LSTMTrainingService)
