import logging

import pandas as pd
from asset_model_data_storage.data_storage_service import DataStorageService
from flask import request, jsonify

from b3.service.pipeline.model.factory import ModelFactory
from b3.service.pipeline.model.utils import is_lstm_model
from b3.service.web_api.asset_api_client import AssetApiClient
from constants import FEATURE_SET


class ModelPredictHandler:
    """
    Handler for making predictions using trained models.
    
    This handler supports both Random Forest and LSTM models:
    - Random Forest: Expects single-row feature data (n_samples, n_features)
    - LSTM: Expects sequence data (n_samples, lookback, n_features) where lookback=32
    
    For LSTM models, the handler will:
    1. Try to build sequences from historical data in pipeline_state if available
    2. If no historical data is available, return an informative error message
    3. Suggest using Random Forest for single-point predictions
    """
    
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

    def _get_expected_features(self):
        """
        Get the expected features for prediction, matching training features.
        This method consolidates the logic for getting expected features from either
        pipeline state or fallback to FEATURE_SET.
        
        Returns:
            List of feature names that should be used for prediction
        """
        if 'X_features' in self._pipeline_state:
            # Use features from training pipeline state
            expected_features = list(pd.DataFrame(self._pipeline_state['X_features']).columns)
            logging.info(f"Using features from pipeline state for prediction: {expected_features}")
        else:
            # Fallback to FEATURE_SET (same as training)
            expected_features = FEATURE_SET
            logging.info(f"Using FEATURE_SET for prediction: {expected_features}")

        return expected_features

    def _align_and_validate_features(self, data_df, expected_features=None, data_source="unknown",
                                     require_all_features=False):
        """
        Consolidated method to align and validate features for prediction.
        This method handles both RF and LSTM feature alignment logic.
        
        Args:
            data_df: DataFrame containing the data
            expected_features: List of expected feature names (if None, uses _get_expected_features())
            data_source: String describing the data source for logging
            require_all_features: If True, raises error on missing features (for RF). 
                                 If False, warns but continues (for LSTM)
            
        Returns:
            Tuple of (aligned_features_df, aligned_features_list, missing_features, error_response)
            - aligned_features_df: DataFrame with aligned features (None if error)
            - aligned_features_list: List of aligned feature names (None if error)
            - missing_features: Set of missing features
            - error_response: Flask jsonify response if error, None otherwise
        """
        if expected_features is None:
            expected_features = self._get_expected_features()
        
        # Filter to only include features that exist in data and match training features
        available_features = [f for f in expected_features if f in data_df.columns]
        missing_features = set(expected_features) - set(available_features)

        # Ensure we have numeric features from the expected feature set
        if len(available_features) > 0:
            numeric_features = data_df[available_features].select_dtypes(include=['number']).columns.tolist()
        else:
            numeric_features = []

        if len(numeric_features) == 0:
            error_msg = f'No numeric features found in {data_source} data from expected feature set'
            logging.warning(error_msg)
            if require_all_features:
                return None, None, missing_features, jsonify({
                    'status': 'error',
                    'message': error_msg,
                    'required_features': expected_features,
                    'available_features': list(data_df.columns)
                })
            return None, None, missing_features, None

        # Validate feature count matches model expectations
        if len(numeric_features) != len(expected_features):
            warning_msg = f"Feature count mismatch in {data_source} data: expected {len(expected_features)}, got {len(numeric_features)}. Available numeric features: {list(numeric_features)}"
            logging.warning(warning_msg)

        # Ensure features are in the same order as training
        aligned_features = [f for f in expected_features if f in numeric_features]

        # If we require all features and some are missing, return error
        if require_all_features and missing_features:
            return None, None, missing_features, jsonify({
                'status': 'error',
                'message': f'Missing required features in data for ticker: {list(missing_features)}',
                'required_features': expected_features,
                'available_features': list(data_df.columns)
            })

        # Create aligned dataframe
        x_aligned = data_df[aligned_features].copy()

        # Clean the prediction data using injected preprocessing service
        x_aligned = self._preprocessing_service.clean_prediction_data(x_aligned)

        return x_aligned, aligned_features, missing_features, None

    def _get_prediction_features(self, new_data, model_type='rf'):
        """
        Helper to extract and validate features for Random Forest prediction.
        Uses consolidated feature alignment logic.
        """
        features_df, features_list, missing_features, error_response = self._align_and_validate_features(
            new_data,
            require_all_features=True
        )

        if error_response:
            return None, None, error_response, 400

        return features_df, features_list, None, None

    def _get_lstm_expected_features(self):
        """
        Get the expected features for LSTM prediction, matching training features.
        Wrapper method for backward compatibility.
        
        Returns:
            List of feature names that should be used for LSTM prediction
        """
        return self._get_expected_features()

    def _align_features_for_lstm(self, data_df, expected_features=None, data_source="unknown"):
        """
        Align features in the data to match training features for LSTM prediction.
        Wrapper method for backward compatibility - uses consolidated alignment logic.
        
        Args:
            data_df: DataFrame containing the data
            expected_features: List of expected feature names (optional)
            data_source: String describing the data source for logging
            
        Returns:
            Tuple of (aligned_features, missing_features, validation_warnings)
        """
        features_df, features_list, missing_features, error_response = self._align_and_validate_features(
            data_df,
            expected_features=expected_features,
            data_source=data_source,
            require_all_features=False
        )

        validation_warnings = []
        if error_response:
            validation_warnings.append("Could not align features")
        elif missing_features:
            validation_warnings.append(f"Missing features: {missing_features}")

        return features_list, missing_features, validation_warnings

    def _prepare_lstm_prediction_data(self, ticker, model, lookback=32):
        """
        Prepare data for LSTM prediction by building sequences from historical data.
        This method checks multiple sources for historical data:
        1. Pipeline state (processed data)
        2. Asset API historical data
        3. Raw data in pipeline state
        
        Args:
            ticker: Ticker symbol
            model: LSTM model instance
            lookback: Number of timesteps for sequence building
            
        Returns:
            Tuple of (X_seq, error_response, error_code) or (None, None, None) if no historical data
        """
        # First, try to get historical data from pipeline state
        if 'processed_data' in self._pipeline_state:
            df_processed = pd.DataFrame(self._pipeline_state['processed_data'])

            # Filter data for the specific ticker
            if 'ticker' in df_processed.columns:
                ticker_data = df_processed[df_processed['ticker'] == ticker].copy()

                if len(ticker_data) >= lookback:
                    # Sort by date to ensure proper sequence
                    if 'date' in ticker_data.columns:
                        ticker_data = ticker_data.sort_values('date')
                    elif 'datetime' in ticker_data.columns:
                        ticker_data = ticker_data.sort_values('datetime')

                    # Get the last lookback rows
                    recent_data = ticker_data.tail(lookback)

                    # Get expected features and align them with available data
                    expected_features = self._get_lstm_expected_features()
                    feature_cols, missing_features, validation_warnings = self._align_features_for_lstm(
                        recent_data, expected_features, "pipeline state"
                    )

                    if feature_cols is None:
                        logging.error(f"Could not align features for LSTM prediction from pipeline state")
                        return None, None, None

                    # Build sequence
                    X_seq = recent_data[feature_cols].values.astype('float32')
                    X_seq = X_seq.reshape(1, lookback, len(feature_cols))  # Add batch dimension

                    logging.info(
                        f"Built LSTM sequence from pipeline state for {ticker}: shape {X_seq.shape}, features: {feature_cols}")
                    if validation_warnings:
                        logging.warning(f"Validation warnings for {ticker}: {validation_warnings}")
                    return X_seq, None, None

        # If no pipeline data, try to fetch historical data from Asset API
        logging.info(f"Attempting to fetch historical data from Asset API for ticker {ticker}")
        historical_data, api_error = AssetApiClient.fetch_historical_data(ticker)

        if historical_data and len(historical_data) >= lookback:
            try:
                # Convert API data to DataFrame
                df_historical = pd.DataFrame(historical_data)

                # Sort by date if available
                date_cols = ['date', 'datetime', 'timestamp', 'created_at', 'updated_at']
                for date_col in date_cols:
                    if date_col in df_historical.columns:
                        df_historical = df_historical.sort_values(date_col)
                        break

                # Get the last lookback rows
                recent_data = df_historical.tail(lookback)

                # Get expected features and align them with available data
                expected_features = self._get_lstm_expected_features()
                feature_cols, missing_features, validation_warnings = self._align_features_for_lstm(
                    recent_data, expected_features, "Asset API"
                )

                if feature_cols is None:
                    logging.error(f"Could not align features for LSTM prediction from Asset API for {ticker}")
                    return None, None, None

                # Build sequence with aligned features
                X_seq = recent_data[feature_cols].values.astype('float32')
                X_seq = X_seq.reshape(1, lookback, len(feature_cols))  # Add batch dimension

                logging.info(
                    f"Built LSTM sequence from Asset API for {ticker}: shape {X_seq.shape}, features: {feature_cols}")
                if validation_warnings:
                    logging.warning(f"Validation warnings for {ticker}: {validation_warnings}")
                return X_seq, None, None

            except Exception as e:
                logging.error(f"Error processing historical data from API: {str(e)}")
                return None, None, None

        # Check if we have raw data that could be processed
        if 'raw_data' in self._pipeline_state:
            logging.info(f"Found raw data in pipeline state, but no processed data for LSTM sequence building")

        # No historical data available from any source
        logging.warning(f"No historical data available for LSTM prediction of ticker {ticker}")
        if api_error:
            logging.warning(f"Asset API error: {api_error}")

        return None, None, None

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

            # Handle LSTM models differently - they need sequences, not single rows
            if is_lstm_model(model_type):
                # Try to build sequence from historical data if available
                X_seq, seq_error, seq_code = self._prepare_lstm_prediction_data(ticker, model)

                if X_seq is not None:
                    # We have historical data, make LSTM prediction
                    logging.info(f"Making LSTM prediction with sequence shape: {X_seq.shape}")
                    predictions = model.predict(X_seq)

                    # Get prediction probabilities if available
                    prediction_probs = None
                    if hasattr(model, 'predict_proba'):
                        prediction_probs = model.predict_proba(X_seq).tolist()
                    elif hasattr(model, 'model') and hasattr(model.model, 'predict'):
                        # For LSTM models, get probabilities from the underlying Keras model
                        try:
                            keras_pred = model.model.predict(X_seq, verbose=0)
                            if isinstance(keras_pred, dict) and 'action' in keras_pred:
                                prediction_probs = keras_pred['action'].tolist()
                        except Exception as e:
                            logging.warning(f"Could not get prediction probabilities: {e}")

                    # Get model type name for response
                    model_type_name = type(model).__name__
                    if hasattr(model, 'model') and hasattr(model.model, '__class__'):
                        model_type_name = type(model.model).__name__

                    resp = {
                        'status': 'success',
                        'message': f'LSTM predictions generated successfully for ticker: {ticker}',
                        'ticker': ticker,
                        'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else predictions,
                        'prediction_probabilities': prediction_probs,
                        'input_shape': X_seq.shape,
                        'model_type': model_type_name,
                        'requested_model_type': model_type,
                        'model_source': model_source,
                        'sequence_length': X_seq.shape[1],
                        'features_count': X_seq.shape[2]
                    }
                    self._log_api_activity(
                        endpoint='predict_data_handler',
                        request_data=data,
                        response_data=resp,
                        status='success'
                    )
                    return jsonify(resp)
                else:
                    # No historical data available, return informative error with steps to fix
                    pipeline_status = {
                        'has_raw_data': 'raw_data' in self._pipeline_state,
                        'has_processed_data': 'processed_data' in self._pipeline_state,
                        'has_trained_model': 'trained_model' in self._pipeline_state
                    }

                    resp = {
                        'status': 'error',
                        'message': f'LSTM models require historical data (lookback window of 32 timesteps) for prediction. '
                                   f'No historical data is available from pipeline state or Asset API. '
                                   f'Please ensure your Asset API supports historical data endpoints, '
                                   f'load and preprocess historical data first, or use Random Forest for single-point predictions.',
                        'model_type': model_type,
                        'required_data': 'Historical sequence of 32 timesteps',
                        'available_data': 'Single current data point',
                        'pipeline_status': pipeline_status,
                        'data_sources_checked': [
                            'Pipeline state processed_data',
                            'Asset API historical endpoints'
                        ],
                        'solutions': [
                            'Use model_type="rf" for single-point predictions',
                            'Ensure Asset API supports historical data endpoints',
                            'Run POST /api/b3/load-data to load historical data',
                            'Run POST /api/b3/preprocess-data to preprocess data',
                            'Then retry LSTM prediction',
                            'Or run POST /api/b3/complete-training with model_type="lstm"'
                        ],
                        'asset_api_endpoint': f'{AssetApiClient.BASE_URL}{ticker}/history?days=32'
                    }
                    self._log_api_activity(
                        endpoint='predict_data_handler',
                        request_data=data,
                        response_data=resp,
                        status='error'
                    )
                    return jsonify(resp), 400

            x_prediction, available_features, error_response, error_code = self._get_prediction_features(new_data,
                                                                                                         model_type)
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
