from flask import request, jsonify
import pandas as pd
import logging


class DataPreprocessingHandler:
    def __init__(self, preprocessing_service, pipeline_state, log_api_activity):
        self.preprocessing_service = preprocessing_service
        self.pipeline_state = pipeline_state
        self._log_api_activity = log_api_activity

    def preprocess_data_handler(self):
        req_data = request.get_json() if request.is_json else None
        try:
            if 'raw_data' not in self.pipeline_state:
                resp = {'status': 'error', 'message': 'No data loaded. Please load data first.'}
                self._log_api_activity(
                    endpoint='preprocess_data_handler',
                    request_data=req_data,
                    response_data=resp,
                    status='error',
                    error_message=resp['message']
                )
                return jsonify(resp), 400
            df = pd.DataFrame(self.pipeline_state['raw_data'])
            X, df_processed, y = self.preprocessing_service.preprocess_data(df)
            self.pipeline_state['X_features'] = X.to_dict('records')
            self.pipeline_state['y_targets'] = y.to_dict()
            self.pipeline_state['processed_data'] = df_processed.to_dict('records')
            resp = {
                'status': 'success',
                'message': 'Data preprocessed successfully',
                'features_shape': X.shape,
                'targets_shape': y.shape,
                'feature_columns': list(X.columns),
                'target_distribution': y.value_counts().to_dict()
            }
            self._log_api_activity(
                endpoint='preprocess_data_handler',
                request_data=req_data,
                response_data=resp,
                status='success'
            )
            return jsonify(resp)
        except Exception as e:
            logging.error(f"Error preprocessing data: {str(e)}")
            resp = {'status': 'error', 'message': str(e)}
            self._log_api_activity(
                endpoint='preprocess_data_handler',
                request_data=req_data,
                response_data=resp,
                status='error',
                error_message=str(e)
            )
            return jsonify(resp), 500
