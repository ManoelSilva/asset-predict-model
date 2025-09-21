import logging

import pandas as pd
from flask import request, jsonify

from constants import FEATURE_SET


class ModelPredictHandler:
    def __init__(self, log_api_activity, data_loading_service, preprocessing_service, model_saving_service,
                 pipeline_state, deserialize_model):
        self._log_api_activity = log_api_activity
        self._data_loading_service = data_loading_service
        self._preprocessing_service = preprocessing_service
        self._model_saving_service = model_saving_service
        self._pipeline_state = pipeline_state
        self._deserialize_model = deserialize_model

    def _load_model(self):
        """Helper to load model from pipeline or storage."""
        model = None
        model_source = None
        if 'trained_model' in self._pipeline_state:
            model = self._deserialize_model(self._pipeline_state['trained_model'])
            model_source = 'pipeline'
        else:
            try:
                if self._model_saving_service.model_exists():
                    model_path = self._model_saving_service.get_model_relative_path()
                    model = self._model_saving_service.load_model(model_path)
                    model_source = 'storage'
                    logging.info(f"Loaded model from storage: {model_path}")
                else:
                    return None, None, jsonify({'status': 'error',
                                                'message': 'No trained model found in pipeline or storage. '
                                                           'Please train model first.'}), 400
            except Exception as e:
                logging.error(f"Error loading model from storage: {str(e)}")
                return None, None, jsonify(
                    {'status': 'error', 'message': f'Error loading model from storage: {str(e)}'}), 500
        return model, model_source, None, None

    def _get_prediction_features(self, new_data):
        """Helper to extract and validate features for prediction."""
        features = [col for col in new_data.select_dtypes(include=['number']).columns if col != 'date']
        x_new = new_data[features]
        if 'X_features' in self._pipeline_state:
            expected_features = list(pd.DataFrame(self._pipeline_state['X_features']).columns)
        else:
            expected_features = FEATURE_SET
            logging.info(f"Using fallback feature set for file-based model: {expected_features}")
        available_features = [f for f in expected_features if f in x_new.columns]
        missing_features = set(expected_features) - set(available_features)
        if missing_features:
            return None, None, jsonify({
                'status': 'error',
                'message': f'Missing required features in data for ticker: {list(missing_features)}',
                'required_features': expected_features,
                'available_features': list(x_new.columns)
            }), 400
        x_prediction = x_new[available_features]
        return x_prediction, available_features, None, None

    def predict_data_handler(self):
        """Make predictions using the trained model."""
        try:
            data = request.get_json() or {}
            ticker = data.get('ticker')

            model, model_source, error_response, error_code = self._load_model()
            if error_response:
                resp = error_response.get_json() if hasattr(error_response, 'get_json') else str(error_response)
                self._log_api_activity(
                    endpoint='predict_data_handler',
                    request_data=data,
                    response_data=resp,
                    status='error'
                )
                return error_response, error_code

            if not ticker:
                resp = {'status': 'error', 'message': 'No ticker provided. Please include "ticker" field in request.'}
                self._log_api_activity(
                    endpoint='predict_data_handler',
                    request_data=data,
                    response_data=resp,
                    status='error'
                )
                return jsonify(resp), 400

            data_loader = self._data_loading_service.get_loader()
            new_data = data_loader.fetch(ticker=ticker)
            if new_data.empty:
                resp = {'status': 'error', 'message': f'No data found for ticker: {ticker}'}
                self._log_api_activity(
                    endpoint='predict_data_handler',
                    request_data=data,
                    response_data=resp,
                    status='error'
                )
                return jsonify(resp), 400

            x_prediction, available_features, error_response, error_code = self._get_prediction_features(new_data)
            if error_response:
                resp = error_response.get_json() if hasattr(error_response, 'get_json') else str(error_response)
                self._log_api_activity(
                    endpoint='predict_data_handler',
                    request_data=data,
                    response_data=resp,
                    status='error'
                )
                return error_response, error_code

            predictions = model.predict(x_prediction)
            prediction_probs = model.predict_proba(x_prediction).tolist() if hasattr(model, 'predict_proba') else None
            feature_importance = model.feature_importances_.tolist() if hasattr(model, 'feature_importances_') else None

            logging.info(f"Predicted actions for ticker {ticker}: {predictions}")
            resp = {
                'status': 'success',
                'message': f'Predictions generated successfully for ticker: {ticker}',
                'ticker': ticker,
                'predictions': predictions.tolist(),
                'prediction_probabilities': prediction_probs,
                'feature_importance': feature_importance,
                'input_shape': x_prediction.shape,
                'features_used': available_features,
                'model_type': type(model).__name__,
                'model_source': model_source
            }
            self._log_api_activity(
                endpoint='predict_data_handler',
                request_data=data,
                response_data=resp,
                status='success'
            )
            return jsonify(resp)
        except Exception as e:
            logging.error(f"Error making predictions: {str(e)}")
            resp = {'status': 'error', 'message': str(e)}
            self._log_api_activity(
                endpoint='predict_data_handler',
                request_data=request.get_json() or {},
                response_data=resp,
                status='error'
            )
            return jsonify(resp), 500
