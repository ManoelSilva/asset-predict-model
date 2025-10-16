import logging

import pandas as pd
from asset_model_data_storage.data_storage_service import DataStorageService
from flask import request, jsonify

from b3.service.pipeline.model.factory import ModelFactory
from b3.service.pipeline.model_evaluation_service import B3ModelEvaluationService
from constants import HTTP_STATUS_BAD_REQUEST, HTTP_STATUS_INTERNAL_SERVER_ERROR, MODEL_STORAGE_KEY


class ModelEvaluationHandler:
    def __init__(self, pipeline_state, log_api_activity, deserialize_model, storage_service=None):
        self._pipeline_state = pipeline_state
        self._log_api_activity = log_api_activity
        self._deserialize_model = deserialize_model
        self._storage_service = storage_service or DataStorageService()

    def evaluate_model_handler(self):
        req_data = request.get_json() if request.is_json else None
        try:
            if MODEL_STORAGE_KEY not in self._pipeline_state:
                resp = {'status': 'error', 'message': 'No trained pipeline found. Please train pipeline first.'}
                self._log_api_activity(
                    endpoint='evaluate_model_handler',
                    request_data=req_data,
                    response_data=resp,
                    status='error',
                    error_message=resp['message']
                )
                return jsonify(resp), HTTP_STATUS_BAD_REQUEST

            data = req_data or {}
            # Try to get model_type from request or pipeline_state
            model_type = data.get('model_type') or self._pipeline_state.get('model_type', 'rf')

            # For LSTM models, we need to handle evaluation differently
            if model_type in ['lstm', 'lstm_mtl']:
                # LSTM models are not serialized in the same way, so we need to load from storage
                saving_service = ModelFactory.get_persist_service(model_type, self._storage_service)
                model_path = saving_service.get_model_path('models')
                model = saving_service.load_model(model_path)
            else:
                # For Random Forest and other serializable models
                model = self._deserialize_model(self._pipeline_state[MODEL_STORAGE_KEY])

            x_val = pd.DataFrame(self._pipeline_state['X_val'])
            y_val = pd.Series(self._pipeline_state['y_val'])
            x_test = pd.DataFrame(self._pipeline_state['X_test'])
            y_test = pd.Series(self._pipeline_state['y_test'])
            df_processed = pd.DataFrame(self._pipeline_state['processed_data'])

            # Instantiate evaluation service at runtime
            evaluation_service = B3ModelEvaluationService(self._storage_service)
            results = evaluation_service.evaluate_model_comprehensive(
                model, x_val, y_val, x_test, y_test, df_processed
            )
            resp = {
                'status': 'success',
                'message': 'Model evaluated successfully',
                'evaluation_results': results
            }
            self._log_api_activity(
                endpoint='evaluate_model_handler',
                request_data=req_data,
                response_data=resp,
                status='success'
            )
            return jsonify(resp)
        except Exception as e:
            logging.error(f"Error evaluating pipeline: {str(e)}")
            resp = {'status': 'error', 'message': str(e)}
            self._log_api_activity(
                endpoint='evaluate_model_handler',
                request_data=req_data,
                response_data=resp,
                status='error',
                error_message=str(e)
            )
            return jsonify(resp), HTTP_STATUS_INTERNAL_SERVER_ERROR
