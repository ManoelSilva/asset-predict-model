import logging

import pandas as pd
from asset_model_data_storage.data_storage_service import DataStorageService
from flask import request, jsonify

from b3.service.pipeline.model.factory import ModelFactory
from b3.service.web_api.asset_api_client import AssetApiClient
from constants import FEATURE_SET


class ModelPredictHandler:
    def __init__(self, preprocessing_service, log_api_activity, pipeline_state, deserialize_model,
                 storage_service=None):
        self._preprocessing_service = preprocessing_service
        self._log_api_activity = log_api_activity
        self._pipeline_state = pipeline_state
        self._deserialize_model = deserialize_model
        self._storage_service = storage_service or DataStorageService()

    def _load_model(self, model_type='rf'):
        """Helper to load model from pipeline or storage using factory pattern."""
        model = None
        model_source = None

        if 'trained_model' in self._pipeline_state:
            model = self._deserialize_model(self._pipeline_state['trained_model'])
            model_source = 'pipeline'
        else:
            try:
                model_instance = ModelFactory.get_model(model_type)
                # Get persist service at runtime with storage service
                saving_service = ModelFactory.get_persist_service(model_type, self._storage_service)

                # Check if model exists and load it
                if saving_service.model_exists():
                    model_path = saving_service.get_model_relative_path()
                    model = model_instance.load_model(model_path)
                    model_source = 'storage'
                    logging.info(f"Loaded {model_type} model from storage: {model_path}")
                else:
                    return None, None, jsonify({'status': 'error',
                                                'message': f'No trained {model_type} model found in storage. '
                                                           'Please train model first.'}), 400
            except ValueError as e:
                # Handle unknown model type
                available_types = ModelFactory.get_available_models()
                return None, None, jsonify({'status': 'error',
                                            'message': f'Unknown model type: {model_type}. '
                                                       f'Available types: {available_types}'}), 400
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

        # Clean the prediction data using injected preprocessing service
        x_prediction = self._preprocessing_service.clean_prediction_data(x_prediction)
        
        return x_prediction, available_features, None, None

    def predict_data_handler(self):
        """Make predictions using the trained pipeline."""
        try:
            data = request.get_json() or {}
            ticker = data.get('ticker')
            model_type = data.get('model_type', 'rf')  # Default to Random Forest for backward compatibility

            model, model_source, error_response, error_code = self._load_model(model_type)
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

            # Fetch ticker info from external API instead of data_loader
            ticker_info, api_error = AssetApiClient.fetch_ticker_info(ticker)
            if api_error or ticker_info is None:
                resp = {'status': 'error', 'message': f'Error fetching ticker info: {api_error or "No data returned"}'}
                self._log_api_activity(
                    endpoint='predict_data_handler',
                    request_data=data,
                    response_data=resp,
                    status='error'
                )
                return jsonify(resp), 400

            # Convert the API response to a DataFrame using only the 'data' field
            asset_data = ticker_info.get('data') if isinstance(ticker_info, dict) else None
            if asset_data is None:
                resp = {'status': 'error', 'message': f'No asset data found in API response for ticker: {ticker}'}
                self._log_api_activity(
                    endpoint='predict_data_handler',
                    request_data=data,
                    response_data=resp,
                    status='error'
                )
                return jsonify(resp), 400
            new_data = pd.DataFrame([asset_data])
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

            # Log data quality information before prediction
            logging.info(f"Prediction data shape: {x_prediction.shape}")
            logging.info(f"Data types: {x_prediction.dtypes.to_dict()}")
            logging.info(f"Data ranges: {x_prediction.describe().to_dict()}")
            
            predictions = model.predict(x_prediction)
            prediction_probs = model.predict_proba(x_prediction).tolist() if hasattr(model, 'predict_proba') else None
            feature_importance = model.feature_importances_.tolist() if hasattr(model, 'feature_importances_') else None

            logging.info(f"Predicted actions for ticker {ticker}: {predictions}")

            # Get model type name for response
            model_type_name = type(model).__name__
            if hasattr(model, 'model') and hasattr(model.model, '__class__'):
                # For wrapped models (like B3KerasMTLModel), get the actual model type
                model_type_name = type(model.model).__name__

            resp = {
                'status': 'success',
                'message': f'Predictions generated successfully for ticker: {ticker}',
                'ticker': ticker,
                'predictions': predictions.tolist(),
                'prediction_probabilities': prediction_probs,
                'feature_importance': feature_importance,
                'input_shape': x_prediction.shape,
                'features_used': available_features,
                'model_type': model_type_name,
                'requested_model_type': model_type,
                'model_source': model_source
            }
            self._log_api_activity(
                endpoint='predict_data_handler',
                request_data=data,
                response_data=resp,
                status='success'
            )
            return jsonify(resp)
        except ValueError as e:
            if "infinity" in str(e).lower() or "float32" in str(e).lower():
                logging.error(f"Data quality issue in prediction: {str(e)}")
                resp = {
                    'status': 'error',
                    'message': 'Data quality issue: Input contains infinity or values too large for processing. '
                               'Please check the ticker data quality.',
                    'technical_details': str(e)
                }
            else:
                logging.error(f"Value error in prediction: {str(e)}")
                resp = {'status': 'error', 'message': str(e)}
            self._log_api_activity(
                endpoint='predict_data_handler',
                request_data=request.get_json() or {},
                response_data=resp,
                status='error'
            )
            return jsonify(resp), 400
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
