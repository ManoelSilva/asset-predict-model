import logging

from dotenv import load_dotenv
from flask import Flask
from flask_cors import CORS

from b3.service.data.data_loading_service import B3DataLoadingService
from b3.service.data.data_preprocessing_service import B3DataPreprocessingService
from b3.service.model.model_evaluation_service import B3ModelEvaluationService
from b3.service.model.model_saving_service import B3ModelSavingService
from b3.service.model.model_training_service import B3ModelTrainingService
from b3.service.web_api.api_handler import B3APIHandlers


class B3TrainingAPI:
    """
    Flask API for B3 model training services.
    Exposes individual training stages as separate endpoints.
    """

    def __init__(self):
        load_dotenv()
        self.app = Flask(__name__)
        CORS(self.app)

        # Initialize services
        self.data_loading_service = B3DataLoadingService()
        self.preprocessing_service = B3DataPreprocessingService()
        self.training_service = B3ModelTrainingService()
        self.evaluation_service = B3ModelEvaluationService()
        self.saving_service = B3ModelSavingService()

        # Initialize handlers (pipeline_state is now managed internally)
        self.handlers = B3APIHandlers(
            self.data_loading_service,
            self.preprocessing_service,
            self.training_service,
            self.evaluation_service,
            self.saving_service
        )

        self._setup_routes()

    def _setup_routes(self):
        """Setup all API routes."""
        self.app.route('/api/b3/load-data', methods=['POST'])(self.handlers.load_data_handler)
        self.app.route('/api/b3/preprocess-data', methods=['POST'])(self.handlers.preprocess_data_handler)
        self.app.route('/api/b3/split-data', methods=['POST'])(self.handlers.split_data_handler)
        self.app.route('/api/b3/train-model', methods=['POST'])(self.handlers.train_model_handler)
        self.app.route('/api/b3/evaluate-model', methods=['POST'])(self.handlers.evaluate_model_handler)
        self.app.route('/api/b3/save-model', methods=['POST'])(self.handlers.save_model_handler)
        self.app.route('/api/b3/train-complete', methods=['POST'])(self.handlers.train_complete_handler)
        self.app.route('/api/b3/predict', methods=['POST'])(self.handlers.predict_data_handler)
        self.app.route('/api/b3/status', methods=['GET'])(self.handlers.get_status_handler)
        self.app.route('/api/b3/training-status', methods=['GET'])(self.handlers.get_training_status_handler)
        self.app.route('/api/b3/clear-state', methods=['POST'])(self.handlers.clear_state_handler)

    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the Flask application."""
        logging.info(f"Starting B3 Training API on {host}:{port}")
        try:
            self.app.run(host=host, port=port, debug=debug)
        finally:
            # Clean up resources when the app stops
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'handlers'):
            self.handlers.cleanup()
