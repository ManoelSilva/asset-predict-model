from flask import request, jsonify
import logging
from constants import HTTP_STATUS_BAD_REQUEST, HTTP_STATUS_INTERNAL_SERVER_ERROR, MODEL_STORAGE_KEY


class ModelSavingHandler:
    def __init__(self, saving_service, pipeline_state, log_api_activity, deserialize_model):
        self.saving_service = saving_service
        self.pipeline_state = pipeline_state
        self._log_api_activity = log_api_activity
        self._deserialize_model = deserialize_model

    def save_model_handler(self):
        req_data = request.get_json() if request.is_json else None
        try:
            if MODEL_STORAGE_KEY not in self.pipeline_state:
                resp = {'status': 'error', 'message': 'No trained model found. Please train model first.'}
                self._log_api_activity(
                    endpoint='save_model_handler',
                    request_data=req_data,
                    response_data=resp,
                    status='error',
                    error_message=resp['message']
                )
                return jsonify(resp), HTTP_STATUS_BAD_REQUEST
            data = req_data or {}
            model_dir = data.get('model_dir', 'models')
            model_name = data.get('model_name', 'b3_model.joblib')
            model = self._deserialize_model(self.pipeline_state[MODEL_STORAGE_KEY])
            model_path = self.saving_service.save_model(model, model_dir, model_name)
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
