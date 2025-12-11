import logging

import pandas as pd
from flask import request, jsonify

from b3.service.pipeline.model.factory import ModelFactory
from b3.service.pipeline.model.utils import normalize_model_type
from constants import HTTP_STATUS_BAD_REQUEST, HTTP_STATUS_INTERNAL_SERVER_ERROR, DEFAULT_TEST_SIZE, DEFAULT_VAL_SIZE


class DataSplittingHandler:
    def __init__(self, pipeline_state, log_api_activity):
        self._pipeline_state = pipeline_state
        self._log_api_activity = log_api_activity

    def split_data_handler(self):
        req_data = request.get_json() if request.is_json else None
        try:
            if 'X_features' not in self._pipeline_state or 'y_targets' not in self._pipeline_state:
                resp = {'status': 'error', 'message': 'No preprocessed data found. Please preprocess data first.'}
                self._log_api_activity(
                    endpoint='split_data_handler',
                    request_data=req_data,
                    response_data=resp,
                    status='error',
                    error_message=resp['message']
                )
                return jsonify(resp), HTTP_STATUS_BAD_REQUEST
            data = req_data or {}
            model_type = data.get('model_type', 'rf')  # Extract model_type from request
            model_type = normalize_model_type(model_type)
            test_size = data.get('test_size', DEFAULT_TEST_SIZE)
            val_size = data.get('val_size', DEFAULT_VAL_SIZE)
            x = pd.DataFrame(self._pipeline_state['X_features'])
            y = pd.Series(self._pipeline_state['y_targets'])

            # Get model instance using factory
            model = ModelFactory.get_model(model_type)

            # Use model's split_data method
            x_train, x_val, x_test, y_train, y_val, y_test = model.split_data(
                x, y, test_size, val_size
            )
            self._pipeline_state['X_train'] = x_train.to_dict('records')
            self._pipeline_state['X_val'] = x_val.to_dict('records')
            self._pipeline_state['X_test'] = x_test.to_dict('records')
            self._pipeline_state['y_train'] = y_train.to_dict()
            self._pipeline_state['y_val'] = y_val.to_dict()
            self._pipeline_state['y_test'] = y_test.to_dict()
            resp = {
                'status': 'success',
                'message': 'Data split successfully',
                'train_size': len(x_train),
                'validation_size': len(x_val),
                'test_size': len(x_test)
            }
            self._log_api_activity(
                endpoint='split_data_handler',
                request_data=req_data,
                response_data=resp,
                status='success'
            )
            return jsonify(resp)
        except Exception as e:
            logging.error(f"Error splitting data: {str(e)}")
            resp = {'status': 'error', 'message': str(e)}
            self._log_api_activity(
                endpoint='split_data_handler',
                request_data=req_data,
                response_data=resp,
                status='error',
                error_message=str(e)
            )
            return jsonify(resp), HTTP_STATUS_INTERNAL_SERVER_ERROR
