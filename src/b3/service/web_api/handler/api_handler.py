import base64
import io
from concurrent.futures import ThreadPoolExecutor

from asset_model_data_storage.data_storage_service import DataStorageService

from b3.service.data.db.training.data_loader import TrainingRequestDataLoader
from b3.service.model.model_saving_service import B3ModelSavingService
from b3.service.web_api.handler.data_loading_handler import DataLoadingHandler
from b3.service.web_api.handler.data_preprocessing_handler import DataPreprocessingHandler
from b3.service.web_api.handler.data_splitting_handler import DataSplittingHandler
from b3.service.web_api.handler.model_complete_training_handler import CompleteTrainingHandler
from b3.service.web_api.handler.model_evaluation_handler import ModelEvaluationHandler
from b3.service.web_api.handler.model_predict_handler import ModelPredictHandler
from b3.service.web_api.handler.model_saving_handler import ModelSavingHandler
from b3.service.web_api.handler.model_training_handler import ModelTrainingHandler
from b3.service.web_api.handler.pipeline_status_handler import PipelineStatusHandler


# Utility functions for serialization/deserialization

def serialize_model(model):
    """Serialize model to base64 string for storage."""
    import joblib
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)
    model_bytes = buffer.getvalue()
    return base64.b64encode(model_bytes).decode('utf-8')


def deserialize_model(model_b64):
    """Deserialize model from base64 string."""
    import joblib
    model_bytes = base64.b64decode(model_b64)
    buffer = io.BytesIO(model_bytes)
    return joblib.load(buffer)


# Logging utility

def make_log_api_activity(request_data_loader, executor):
    def _log_api_activity(endpoint, request_data, response_data, status, error_message=None):
        def log_task():
            if request_data_loader:
                log_kwargs = {
                    'endpoint': endpoint,
                    'request_data': request_data,
                    'response_data': response_data,
                    'status': status
                }
                if error_message is not None:
                    log_kwargs['error_message'] = error_message
                request_data_loader.log_api_activity(**log_kwargs)

        executor.submit(log_task)

    return _log_api_activity


# Factory to create all handlers with shared state

def create_b3_api_handlers(data_loading_service, preprocessing_service, training_service, evaluation_service,
                           saving_service, storage_service=None, request_data_loader=None):
    storage_service = storage_service or DataStorageService()
    model_saving_service = B3ModelSavingService(storage_service)
    pipeline_state = {}
    executor = ThreadPoolExecutor(max_workers=4)
    request_data_loader = request_data_loader or TrainingRequestDataLoader()
    log_api_activity = make_log_api_activity(request_data_loader, executor)

    handlers = {
        'data_loading': DataLoadingHandler(data_loading_service, pipeline_state, log_api_activity),
        'data_preprocessing': DataPreprocessingHandler(preprocessing_service, pipeline_state, log_api_activity),
        'data_splitting': DataSplittingHandler(training_service, pipeline_state, log_api_activity),
        'model_training': ModelTrainingHandler(training_service, pipeline_state, log_api_activity, serialize_model),
        'model_evaluation': ModelEvaluationHandler(evaluation_service, pipeline_state, log_api_activity,
                                                   deserialize_model),
        'model_saving': ModelSavingHandler(model_saving_service, pipeline_state, log_api_activity, deserialize_model),
        'pipeline_status': PipelineStatusHandler(pipeline_state, log_api_activity),
        'complete_training': CompleteTrainingHandler(
            data_loading_service, preprocessing_service, training_service, evaluation_service, saving_service,
            pipeline_state, log_api_activity, serialize_model
        ),
        'predict': ModelPredictHandler(log_api_activity, data_loading_service, preprocessing_service,
                                       model_saving_service, pipeline_state, deserialize_model)
    }
    return handlers
