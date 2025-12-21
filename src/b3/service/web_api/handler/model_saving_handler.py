import logging

from asset_model_data_storage.data_storage_service import DataStorageService
from flask import request, jsonify

from b3.service.pipeline.model.factory import ModelFactory
from b3.service.pipeline.model.utils import is_lstm_model
from constants import HTTP_STATUS_BAD_REQUEST, HTTP_STATUS_INTERNAL_SERVER_ERROR, MODEL_STORAGE_KEY


class ModelSavingHandler:
    def __init__(self, pipeline_state, log_api_activity, deserialize_model, storage_service=None):
        self._pipeline_state = pipeline_state
        self._log_api_activity = log_api_activity
        self._deserialize_model = deserialize_model
        self._storage_service = storage_service or DataStorageService()

    def save_model_handler(self):
        req_data = request.get_json() if request.is_json else None
        try:
            data = req_data or {}
            # Get model_type from request or pipeline_state
            model_type = data.get('model_type') or self._pipeline_state.get('model_type', 'rf')

            has_rf_model = MODEL_STORAGE_KEY in self._pipeline_state

            has_lstm_model = is_lstm_model(model_type) and \
                             (self._pipeline_state.get('training_status') == 'completed' or
                              self._pipeline_state.get('model_path'))

            if not (has_rf_model or has_lstm_model):
                resp = {'status': 'error', 'message': 'No trained model found. Please train model first.'}
                self._log_api_activity(
                    endpoint='save_model_handler',
                    request_data=req_data,
                    response_data=resp,
                    status='error',
                    error_message=resp['message']
                )
                return jsonify(resp), HTTP_STATUS_BAD_REQUEST

            model_dir = data.get('model_dir', 'models')
            model_name = data.get('model_name', 'b3_model.joblib')

            # Get persist service at runtime based on model_type
            persist_service = ModelFactory.get_persist_service(model_type, self._storage_service)

            if is_lstm_model(model_type):
                model_instance = ModelFactory.get_model(model_type)
                current_model_path = self._pipeline_state.get('model_path') or persist_service.get_model_path('models')
                model_wrapper = model_instance.load_model(current_model_path)
                model_to_save = model_wrapper.model
            else:
                model_to_save = self._deserialize_model(self._pipeline_state[MODEL_STORAGE_KEY])

            model_path = persist_service.save_model(model_to_save, model_dir, model_name)
            resp = {
                'status': 'success',
                'message': 'Model saved successfully',
                'model_path': model_path
            }
            self._log_api_activity(
                endpoint='save_model_handler',
                request_data=req_data,
                response_data=resp,
                status='success'
            )
            return jsonify(resp)
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")
            resp = {'status': 'error', 'message': str(e)}
            self._log_api_activity(
                endpoint='save_model_handler',
                request_data=req_data,
                response_data=resp,
                status='error',
                error_message=str(e)
            )
            return jsonify(resp), HTTP_STATUS_INTERNAL_SERVER_ERROR
