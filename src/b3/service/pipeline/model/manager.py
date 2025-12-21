import logging
from typing import Tuple, Dict

from asset_model_data_storage.data_storage_service import DataStorageService

from b3.service.data.db.b3_featured.data_loading_service import DataLoadingService
from b3.service.pipeline.model.factory import ModelFactory
from b3.service.pipeline.model.training_config import TrainingConfig
from b3.service.pipeline.model.utils import normalize_model_type
from b3.service.pipeline.model_preprocessing_service import PreprocessingService


class CompletePipelineManagerService(object):
    def __init__(self, storage_service: DataStorageService = None, data_loading_service: DataLoadingService = None,
                 pre_processing_service: PreprocessingService = None) -> None:
        self._storage_service = storage_service or DataStorageService()
        self._data_loading_service = data_loading_service or DataLoadingService()
        self._preprocessing_service = pre_processing_service or PreprocessingService()

    def run(self, config: TrainingConfig = None, df=None, X=None, y=None, df_processed=None, **kwargs):
        """
        Loads data (if not provided), creates BUY/SELL labels, trains a model using the factory pattern, and saves the model.
        Uses the new service classes for modular training.
        
        Args:
            config (TrainingConfig): Training configuration object. If None, creates default config.
            df: Optional raw dataframe to avoid reloading
            X: Optional preprocessed features
            y: Optional preprocessed targets
            df_processed: Optional processed dataframe
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

        # Step 1: Load data using data loading service if not provided
        if df is None and (X is None or df_processed is None):
            df = self._data_loading_service.load_data()

        # Step 2: Preprocess data using preprocessing service if not provided
        if X is None or y is None or df_processed is None:
            X, df_processed, y = self._preprocessing_service.preprocess_data(df)

        # Step 3: Create model using factory pattern with automatic config creation
        try:
            model = ModelFactory.get_model(config.model_type, **config.get_all_config_dict())
        except ValueError as e:
            logging.error(f"Error creating model: {str(e)}")
            raise

        # Step 4: Train model using unified interface
        model_path, evaluation_results = model.run_pipeline(X, y, df_processed, config)

        return model_path, evaluation_results
