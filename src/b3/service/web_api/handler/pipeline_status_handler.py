import logging

from flask import request, jsonify


class PipelineStatusHandler:
    def __init__(self, pipeline_state, log_api_activity):
        self._pipeline_state = pipeline_state
        self._log_api_activity = log_api_activity

    def get_status_handler(self):
        req_data = request.get_json() if request.is_json else None
        try:
            status_info = {
                'has_raw_data': 'raw_data' in self._pipeline_state,
                'has_processed_data': 'X_features' in self._pipeline_state,
                'has_split_data': 'X_train' in self._pipeline_state,
                'has_trained_model': 'trained_model' in self._pipeline_state
            }
            if 'training_status' in self._pipeline_state:
                status_info['training_status'] = self._pipeline_state['training_status']
            if 'training_results' in self._pipeline_state:
                status_info['training_results'] = self._pipeline_state['training_results']
            if 'training_error' in self._pipeline_state:
                status_info['training_error'] = self._pipeline_state['training_error']
            resp = {
                'status': 'success',
                'pipeline_state': status_info
            }
            self._log_api_activity(
                endpoint='get_status_handler',
                request_data=req_data,
                response_data=resp,
                status='success'
            )
            return jsonify(resp)
        except Exception as e:
            logging.error(f"Error getting status: {str(e)}")
            resp = {'status': 'error', 'message': str(e)}
            self._log_api_activity(
                endpoint='get_status_handler',
                request_data=req_data,
                response_data=resp,
                status='error',
                error_message=str(e)
            )
            return jsonify(resp), 500

    def clear_state_handler(self):
        req_data = request.get_json() if request.is_json else None
        try:
            if 'training_future' in self._pipeline_state:
                future = self._pipeline_state['training_future']
                if not future.done():
                    future.cancel()
            self._pipeline_state.clear()
            resp = {'status': 'success', 'message': 'Pipeline state cleared'}
            self._log_api_activity(
                endpoint='clear_state_handler',
                request_data=req_data,
                response_data=resp,
                status='success'
            )
            return jsonify(resp)
        except Exception as e:
            logging.error(f"Error clearing state: {str(e)}")
            resp = {'status': 'error', 'message': str(e)}
            self._log_api_activity(
                endpoint='clear_state_handler',
                request_data=req_data,
                response_data=resp,
                status='error',
                error_message=str(e)
            )
            return jsonify(resp), 500
