import logging

import pandas as pd
from flask import request, jsonify


class ModelTrainingHandler:
    def __init__(self, training_service, pipeline_state, log_api_activity, serialize_model):
        self.training_service = training_service
        self.pipeline_state = pipeline_state
        self._log_api_activity = log_api_activity
        self._serialize_model = serialize_model

    def train_model_handler(self):
        req_data = request.get_json() if request.is_json else None
        try:
            if 'X_train' not in self.pipeline_state or 'y_train' not in self.pipeline_state:
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
            x_train = pd.DataFrame(self.pipeline_state['X_train'])
            y_train = pd.Series(self.pipeline_state['y_train'])
            model = self.training_service.train_model(x_train, y_train, n_jobs)
            model_b64 = self._serialize_model(model)
            self.pipeline_state['trained_model'] = model_b64
            resp = {
                'status': 'success',
                'message': 'Model trained successfully',
                'model_type': type(model).__name__,
                'best_params': model.get_params()
            }
            self._log_api_activity(
                endpoint='train_model_handler',
                request_data=req_data,
                response_data=resp,
                status='success'
            )
            return jsonify(resp)
        except Exception as e:
            logging.error(f"Error training model: {str(e)}")
            resp = {'status': 'error', 'message': str(e)}
            self._log_api_activity(
                endpoint='train_model_handler',
                request_data=req_data,
                response_data=resp,
                status='error',
                error_message=str(e)
            )
            return jsonify(resp), 500
