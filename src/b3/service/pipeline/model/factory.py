import logging
from typing import Optional

from b3.service.pipeline.model.config import ModelConfig
from b3.service.pipeline.model.lstm.lstm_model import LSTMConfig, LSTMModel
from b3.service.pipeline.model.model import BaseModel
from b3.service.pipeline.model.rf.rf_model import RandomForestConfig, RandomForestModel
from b3.service.pipeline.persist.lstm_persist_service import LSTMPersistService
from b3.service.pipeline.persist.rf_persist_service import RandomForestPersistService


class ModelFactory:
    """
    Factory class for creating model instances based on model type.
    Handles model type logic and configuration creation.
    """

    _model_registry = {}

    @classmethod
    def register_model(cls, model_type: str, model_class: type):
        """
        Register a model class for a specific model type.

        Args:
            model_type: String identifier for the model type
            model_class: Model class to register
        """
        if not issubclass(model_class, BaseModel):
            raise ValueError(f"Model class {model_class} must inherit from BaseModel")
        cls._model_registry[model_type] = model_class
        logging.info(f"Registered model type '{model_type}' with class {model_class.__name__}")

    @classmethod
    def get_model(cls, model_type: str, config: Optional[ModelConfig] = None, **kwargs) -> BaseModel:
        """
        Create a model instance based on model type with automatic configuration creation.

        Args:
            model_type: Type of model to create
            config: Optional configuration for the model
            **kwargs: Additional parameters for configuration creation

        Returns:
            Instance of the requested model type

        Raises:
            ValueError: If model_type is not registered
        """
        if model_type not in cls._model_registry:
            available_types = list(cls._model_registry.keys())
            raise ValueError(f"Unknown model type '{model_type}'. Available types: {available_types}")

        model_class = cls._model_registry[model_type]

        # If no config provided, create one based on model type and kwargs
        if config is None:
            config = cls._create_config_for_model_type(model_type, **kwargs)

        return model_class(config)

    @classmethod
    def _create_config_for_model_type(cls, model_type: str, **kwargs) -> ModelConfig:
        """
        Create appropriate configuration for the given model type.

        Args:
            model_type: Type of model
            **kwargs: Configuration parameters

        Returns:
            ModelConfig instance appropriate for the model type
        """
        if model_type in ["rf", "random_forest"]:
            return RandomForestConfig(**kwargs)
        elif model_type in ["lstm", "lstm_mtl"]:
            return LSTMConfig(**kwargs)
        else:
            raise ValueError(f"No configuration class found for model type: {model_type}")

    @classmethod
    def get_persist_service(cls, model_type: str, storage_service):
        """
        Get the appropriate persist service for the model type.

        Args:
            model_type: Type of model
            storage_service: Storage service instance

        Returns:
            Appropriate persist service instance
        """
        if model_type in ["rf", "random_forest"]:
            return RandomForestPersistService(storage_service)
        elif model_type in ["lstm", "lstm_mtl"]:
            return LSTMPersistService(storage_service)
        else:
            raise ValueError(f"No persist service found for model type: {model_type}")

    @classmethod
    def get_available_models(cls) -> list:
        """
        Get list of available model types.

        Returns:
            List of registered model type strings
        """
        return list(cls._model_registry.keys())

    @classmethod
    def is_model_registered(cls, model_type: str) -> bool:
        """
        Check if a model type is registered.

        Args:
            model_type: Model type to check

        Returns:
            True if model type is registered, False otherwise
        """
        return model_type in cls._model_registry


# Register the Random Forest model with the factory
ModelFactory.register_model("rf", RandomForestModel)
ModelFactory.register_model("random_forest", RandomForestModel)
# Register the LSTM model with the factory
ModelFactory.register_model("lstm", LSTMModel)
ModelFactory.register_model("lstm_mtl", LSTMModel)
