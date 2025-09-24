from flask import request, jsonify
import logging
from constants import HTTP_STATUS_INTERNAL_SERVER_ERROR


class DataLoadingHandler:
    def __init__(self, data_loading_service, pipeline_state, log_api_activity):
        self.data_loading_service = data_loading_service
        self.pipeline_state = pipeline_state
        self._log_api_activity = log_api_activity

    def load_data_handler(self):
        req_data = request.get_json() if request.is_json else None
        try:
            df = self.data_loading_service.load_data()
            self.pipeline_state['raw_data'] = df.to_dict('records')
            self.pipeline_state['data_shape'] = df.shape
            resp = {
                'status': 'success',
                'message': 'Data loaded successfully',
                'data_shape': df.shape,
                'columns': list(df.columns)
            }
            self._log_api_activity(
                endpoint='load_data_handler',
                request_data=req_data,
                response_data=resp,
                status='success'
            )
            return jsonify(resp)
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            resp = {'status': 'error', 'message': str(e)}
            self._log_api_activity(
                endpoint='load_data_handler',
                request_data=req_data,
                response_data=resp,
                status='error',
                error_message=str(e)
            )
            return jsonify(resp), HTTP_STATUS_INTERNAL_SERVER_ERROR
