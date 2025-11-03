import logging

import pandas as pd
from flask import request, jsonify

from b3.service.pipeline.model.utils import is_rf_model
from b3.service.pipeline.training.training_service_factory import TrainingServiceFactory


class ModelTrainingHandler:
    def __init__(self, pipeline_state, log_api_activity, serialize_model, storage_service=None):
        self._pipeline_state = pipeline_state
        self._log_api_activity = log_api_activity
        self._serialize_model = serialize_model
        self._storage_service = storage_service

    def train_model_handler(self):
        req_data = request.get_json() if request.is_json else None
        try:
            if 'X_train' not in self._pipeline_state or 'y_train' not in self._pipeline_state:
                resp = {'status': 'error', 'message': 'No training data found. Please split data first.'}
                self._log_api_activity(
                    endpoint='train_model_handler',
                    request_data=req_data,
                    response_data=resp,
                    status='error',
                    error_message=resp['message']
                )
                return jsonify(resp), 400
            data = req_data or {}
            n_jobs = data.get('n_jobs', 5)
            model_type = data.get('model_type', 'rf')  # Extract model_type from request

            x_train = pd.DataFrame(self._pipeline_state['X_train'])
            y_train = pd.Series(self._pipeline_state['y_train'])

            # Get training service at runtime based on model_type
            training_service = TrainingServiceFactory.get_training_service(model_type)
            trained_model = training_service.train_model(x_train, y_train, n_jobs)

            # Store model_type in pipeline_state
            self._pipeline_state['model_type'] = model_type

            # Only serialize models that can be serialized (like Random Forest)
            if is_rf_model(model_type):
                model_b64 = self._serialize_model(trained_model)
                self._pipeline_state['trained_model'] = model_b64
            else:
                # For LSTM models, store the model type instead
                self._pipeline_state['trained_model_type'] = model_type
            resp = {
                'status': 'success',
                'message': 'Model trained successfully',
                'model_type': type(trained_model).__name__,
                'requested_model_type': model_type,
                'best_params': trained_model.get_params() if hasattr(trained_model, 'get_params') else {}
            }
            self._log_api_activity(
                endpoint='train_model_handler',
                request_data=req_data,
                response_data=resp,
                status='success'
            )
            return jsonify(resp)
        except Exception as e:
            logging.error(f"Error training pipeline: {str(e)}")
            resp = {'status': 'error', 'message': str(e)}
            self._log_api_activity(
                endpoint='train_model_handler',
                request_data=req_data,
                response_data=resp,
                status='error',
                error_message=str(e)
            )
            return jsonify(resp), 500
