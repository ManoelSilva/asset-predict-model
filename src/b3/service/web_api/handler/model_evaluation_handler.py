from flask import request, jsonify
import pandas as pd
import logging


class ModelEvaluationHandler:
    def __init__(self, evaluation_service, pipeline_state, log_api_activity, deserialize_model):
        self.evaluation_service = evaluation_service
        self.pipeline_state = pipeline_state
        self._log_api_activity = log_api_activity
        self._deserialize_model = deserialize_model

    def evaluate_model_handler(self):
        req_data = request.get_json() if request.is_json else None
        try:
            if 'trained_model' not in self.pipeline_state:
                resp = {'status': 'error', 'message': 'No trained model found. Please train model first.'}
                self._log_api_activity(
                    endpoint='evaluate_model_handler',
                    request_data=req_data,
                    response_data=resp,
                    status='error',
                    error_message=resp['message']
                )
                return jsonify(resp), 400
            model = self._deserialize_model(self.pipeline_state['trained_model'])
            x_val = pd.DataFrame(self.pipeline_state['X_val'])
            y_val = pd.Series(self.pipeline_state['y_val'])
            x_test = pd.DataFrame(self.pipeline_state['X_test'])
            y_test = pd.Series(self.pipeline_state['y_test'])
            df_processed = pd.DataFrame(self.pipeline_state['processed_data'])
            results = self.evaluation_service.evaluate_model_comprehensive(
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
            logging.error(f"Error evaluating model: {str(e)}")
            resp = {'status': 'error', 'message': str(e)}
            self._log_api_activity(
                endpoint='evaluate_model_handler',
                request_data=req_data,
                response_data=resp,
                status='error',
                error_message=str(e)
            )
            return jsonify(resp), 500
