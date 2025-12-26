import logging
import os
from concurrent.futures.thread import ThreadPoolExecutor

from asset_model_data_storage.data_storage_service import DataStorageService
from flask import Flask, send_from_directory
from flask_cors import CORS
from flask_swagger_ui import get_swaggerui_blueprint
from waitress import serve

from b3.service.data.db.b3_featured.data_loading_service import DataLoadingService
from b3.service.data.db.training.data_loader import TrainingRequestDataLoader
from b3.service.pipeline.model_preprocessing_service import PreprocessingService
from b3.service.web_api.handler.complete_pipeline.complete_pipeline_handler import CompletePipelineHandler
from b3.service.web_api.handler.data_loading_handler import DataLoadingHandler
from b3.service.web_api.handler.data_preprocessing_handler import DataPreprocessingHandler
from b3.service.web_api.handler.data_splitting_handler import DataSplittingHandler
from b3.service.web_api.handler.model_evaluation_handler import ModelEvaluationHandler
from b3.service.web_api.handler.model_predict_handler import ModelPredictHandler
from b3.service.web_api.handler.model_saving_handler import ModelSavingHandler
from b3.service.web_api.handler.model_training_handler import ModelTrainingHandler
from b3.service.web_api.handler.pipeline_status_handler import PipelineStatusHandler
from b3.service.web_api.handler.transfer_learning_handler import TransferLearningHandler
from b3.utils.api_handler_utils import ApiHandlerUtils
from constants import DEFAULT_API_HOST, DEFAULT_API_PORT


class B3ModelAPI:
    """
    Flask API for B3 model training services.
    Exposes individual training stages as separate endpoints.
    """

    def __init__(self):
        self._app = Flask(__name__)
        CORS(self._app)

        # Serve swagger.yml as static file
        @self._app.route('/swagger/swagger.yml')
        def swagger_spec():
            swagger_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../config/web_api/swagger'))
            return send_from_directory(swagger_dir, 'swagger.yml')

        # Swagger UI setup
        swaggerui_blueprint = get_swaggerui_blueprint(
            '/swagger',
            '/swagger/swagger.yml'
        )
        self._app.register_blueprint(swaggerui_blueprint, url_prefix='/swagger')

        # Initialize shared services that are not model-type dependent
        self._storage_service = DataStorageService()
        self._data_loading_service = DataLoadingService()
        self._preprocessing_service = PreprocessingService()

        # Initialize modular handlers
        self._handlers = self._create_handlers(
            storage_service=self._storage_service,
            data_loading_service=self._data_loading_service,
            preprocessing_service=self._preprocessing_service,
            request_data_loader=None
        )

        self._setup_routes()

    @staticmethod
    def _create_handlers(storage_service: DataStorageService = None,
                         data_loading_service=None,
                         preprocessing_service=None,
                         request_data_loader=None):
        executor = ThreadPoolExecutor(max_workers=4)
        request_data_loader = request_data_loader or TrainingRequestDataLoader()

        pipeline_state = {}
        log_api_activity = ApiHandlerUtils.make_log_api_activity(request_data_loader, executor)
        handlers = {
            'data_loading': DataLoadingHandler(data_loading_service, pipeline_state, log_api_activity),
            'data_preprocessing': DataPreprocessingHandler(preprocessing_service, pipeline_state, log_api_activity),
            'data_splitting': DataSplittingHandler(pipeline_state, log_api_activity),
            'model_training': ModelTrainingHandler(pipeline_state, log_api_activity, storage_service),
            'model_evaluation': ModelEvaluationHandler(pipeline_state, log_api_activity, storage_service),
            'model_saving': ModelSavingHandler(pipeline_state, log_api_activity, storage_service),
            'pipeline_status': PipelineStatusHandler(pipeline_state, log_api_activity),
            'complete_pipeline': CompletePipelineHandler(pipeline_state, log_api_activity, data_loading_service,
                                                         preprocessing_service, storage_service),
            'predict': ModelPredictHandler(pipeline_state, log_api_activity, preprocessing_service, storage_service),
            'transfer_learning': TransferLearningHandler(pipeline_state, log_api_activity, storage_service)
        }
        return handlers

    def _setup_routes(self):
        """Setup all API routes."""
        self._app.route('/api/b3/load-data', methods=['POST'])(self._handlers['data_loading'].load_data_handler)
        self._app.route('/api/b3/preprocess-data', methods=['POST'])(
            self._handlers['data_preprocessing'].preprocess_data_handler)
        self._app.route('/api/b3/split-data', methods=['POST'])(self._handlers['data_splitting'].split_data_handler)
        self._app.route('/api/b3/train-model', methods=['POST'])(self._handlers['model_training']
                                                                 .train_model_handler)
        self._app.route('/api/b3/evaluate-model', methods=['POST'])(
            self._handlers['model_evaluation'].evaluate_model_handler)
        self._app.route('/api/b3/save-model', methods=['POST'])(self._handlers['model_saving'].save_model_handler)
        self._app.route('/api/b3/complete-pipeline', methods=['POST'])(
            self._handlers['complete_pipeline'].complete_pipeline_handler)
        self._app.route('/api/b3/pipeline-status', methods=['GET'])(
            self._handlers['pipeline_status'].get_status_handler)
        self._app.route('/api/b3/clear-state', methods=['POST'])(self._handlers['pipeline_status']
                                                                 .clear_state_handler)
        self._app.route('/api/b3/predict', methods=['POST'])(self._handlers['predict'].predict_data_handler)
        self._app.route('/api/b3/transfer-learning', methods=['POST'])(
            self._handlers['transfer_learning'].run_transfer_learning)

    def run(self, host=DEFAULT_API_HOST, port=DEFAULT_API_PORT):
        logging.info(f"Starting B3 Model API on {host}:{port}")
        serve(self._app, host=host, port=port)
