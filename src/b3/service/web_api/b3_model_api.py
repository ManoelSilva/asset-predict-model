import logging
import os

from dotenv import load_dotenv
from flask import Flask, send_from_directory
from flask_cors import CORS
from flask_swagger_ui import get_swaggerui_blueprint
from waitress import serve

from b3.service.data.db.b3_featured.data_loading_service import B3DataLoadingService
from b3.service.model.model_evaluation_service import B3ModelEvaluationService
from b3.service.model.model_preprocessing_service import B3ModelPreprocessingService
from b3.service.model.model_saving_service import B3ModelSavingService
from b3.service.model.model_training_service import B3ModelTrainingService
from b3.service.web_api.handler.api_handler import create_b3_api_handlers
from constants import DEFAULT_API_HOST, DEFAULT_API_PORT


class B3ModelAPI:
    """
    Flask API for B3 model training services.
    Exposes individual training stages as separate endpoints.
    """

    def __init__(self):
        load_dotenv()
        self.app = Flask(__name__)
        CORS(self.app)

        # Serve swagger.yml as static file
        @self.app.route('/swagger/swagger.yml')
        def swagger_spec():
            swagger_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../config/web_api/swagger'))
            return send_from_directory(swagger_dir, 'swagger.yml')

        # Swagger UI setup
        swaggerui_blueprint = get_swaggerui_blueprint(
            '/swagger',
            '/swagger/swagger.yml'
        )
        self.app.register_blueprint(swaggerui_blueprint, url_prefix='/swagger')

        # Initialize services
        self.data_loading_service = B3DataLoadingService()
        self.preprocessing_service = B3ModelPreprocessingService()
        self.training_service = B3ModelTrainingService()
        self.evaluation_service = B3ModelEvaluationService()
        self.saving_service = B3ModelSavingService()

        # Initialize modular handlers
        self.handlers = create_b3_api_handlers(
            self.data_loading_service,
            self.preprocessing_service,
            self.training_service,
            self.evaluation_service,
            self.saving_service
        )

        self._setup_routes()

    def _setup_routes(self):
        """Setup all API routes."""
        self.app.route('/api/b3/load-data', methods=['POST'])(self.handlers['data_loading'].load_data_handler)
        self.app.route('/api/b3/preprocess-data', methods=['POST'])(
            self.handlers['data_preprocessing'].preprocess_data_handler)
        self.app.route('/api/b3/split-data', methods=['POST'])(self.handlers['data_splitting'].split_data_handler)
        self.app.route('/api/b3/train-model', methods=['POST'])(self.handlers['model_training'].train_model_handler)
        self.app.route('/api/b3/evaluate-model', methods=['POST'])(
            self.handlers['model_evaluation'].evaluate_model_handler)
        self.app.route('/api/b3/save-model', methods=['POST'])(self.handlers['model_saving'].save_model_handler)
        self.app.route('/api/b3/complete-training', methods=['POST'])(
            self.handlers['complete_training'].complete_training_handler)
        self.app.route('/api/b3/status', methods=['GET'])(self.handlers['pipeline_status'].get_status_handler)
        self.app.route('/api/b3/training-status', methods=['GET'])(
            self.handlers['pipeline_status'].get_training_status_handler)
        self.app.route('/api/b3/clear-state', methods=['POST'])(self.handlers['pipeline_status'].clear_state_handler)
        self.app.route('/api/b3/predict', methods=['POST'])(self.handlers['predict'].predict_data_handler)

    def run(self, host=DEFAULT_API_HOST, port=DEFAULT_API_PORT, debug=False):
        """Run the Flask application with waitress for production."""
        logging.info(f"Starting B3 Training API on {host}:{port}")
        serve(self.app, host=host, port=port)
