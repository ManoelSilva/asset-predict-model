import logging
import pandas as pd
from flask import request, jsonify
import joblib
import io
import base64
from concurrent.futures import ThreadPoolExecutor

from b3.service.data.db.training.data_loader import TrainingRequestDataLoader
from b3.service.model.model_preprocessing_service import B3ModelPreprocessingService
from b3.service.data.storage.data_storage_service import DataStorageService
from b3.service.model.model_saving_service import B3ModelSavingService


class B3APIHandlers:
    """
    Handlers for B3 Training API endpoints.
    Contains all the business logic for processing API requests.
    """

    def __init__(self, data_loading_service, preprocessing_service, training_service,
                 evaluation_service, saving_service, storage_service=None, request_data_loader=None):
        self.data_loading_service = data_loading_service
        self.preprocessing_service = preprocessing_service
        self.training_service = training_service
        self.evaluation_service = evaluation_service
        self.saving_service = saving_service
        # Initialize storage service and model saving service
        self.storage_service = storage_service or DataStorageService()
        self.model_saving_service = B3ModelSavingService(self.storage_service)
        # In-memory storage for pipeline state
        self.pipeline_state = {}
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=1)
        # Add training_data_loader for persistent API logging
        self.request_data_loader = request_data_loader or TrainingRequestDataLoader()

    def load_data_handler(self):
        """Load B3 market data."""
        req_data = request.get_json() if request.is_json else None
        try:
            df = self.data_loading_service.load_data()
            # Store in pipeline state
            self.pipeline_state['raw_data'] = df.to_dict('records')
            self.pipeline_state['data_shape'] = df.shape
            resp = {
                'status': 'success',
                'message': 'Data loaded successfully',
                'data_shape': df.shape,
                'columns': list(df.columns)
            }
            if self.request_data_loader:
                self.request_data_loader.log_api_activity(
                    endpoint='load_data_handler',
                    request_data=req_data,
                    response_data=resp,
                    status='success'
                )
            return jsonify(resp)
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            resp = {'status': 'error', 'message': str(e)}
            if self.request_data_loader:
                self.request_data_loader.log_api_activity(
                    endpoint='load_data_handler',
                    request_data=req_data,
                    response_data=resp,
                    status='error',
                    error_message=str(e)
                )
            return jsonify(resp), 500

    def preprocess_data_handler(self):
        """Preprocess loaded data."""
        req_data = request.get_json() if request.is_json else None
        try:
            if 'raw_data' not in self.pipeline_state:
                resp = {'status': 'error', 'message': 'No data loaded. Please load data first.'}
                if self.request_data_loader:
                    self.request_data_loader.log_api_activity(
                        endpoint='preprocess_data_handler',
                        request_data=req_data,
                        response_data=resp,
                        status='error',
                        error_message=resp['message']
                    )
                return jsonify(resp), 400

            # Reconstruct DataFrame from stored data
            df = pd.DataFrame(self.pipeline_state['raw_data'])

            # Preprocess data
            X, df_processed, y = self.preprocessing_service.preprocess_data(df)

            # Store processed data
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
            if self.request_data_loader:
                self.request_data_loader.log_api_activity(
                    endpoint='preprocess_data_handler',
                    request_data=req_data,
                    response_data=resp,
                    status='success'
                )
            return jsonify(resp)
        except Exception as e:
            logging.error(f"Error preprocessing data: {str(e)}")
            resp = {'status': 'error', 'message': str(e)}
            if self.request_data_loader:
                self.request_data_loader.log_api_activity(
                    endpoint='preprocess_data_handler',
                    request_data=req_data,
                    response_data=resp,
                    status='error',
                    error_message=str(e)
                )
            return jsonify(resp), 500

    def split_data_handler(self):
        """Split preprocessed data into train/validation/test sets."""
        req_data = request.get_json() if request.is_json else None
        try:
            if 'X_features' not in self.pipeline_state or 'y_targets' not in self.pipeline_state:
                resp = {'status': 'error', 'message': 'No preprocessed data found. Please preprocess data first.'}
                if self.request_data_loader:
                    self.request_data_loader.log_api_activity(
                        endpoint='split_data_handler',
                        request_data=req_data,
                        response_data=resp,
                        status='error',
                        error_message=resp['message']
                    )
                return jsonify(resp), 400

            # Get parameters from request
            data = req_data or {}
            test_size = data.get('test_size', 0.2)
            val_size = data.get('val_size', 0.2)

            # Reconstruct DataFrames
            X = pd.DataFrame(self.pipeline_state['X_features'])
            y = pd.Series(self.pipeline_state['y_targets'])

            # Split data
            X_train, X_val, X_test, y_train, y_val, y_test = self.training_service.split_data(
                X, y, test_size, val_size
            )

            # Store split data
            self.pipeline_state['X_train'] = X_train.to_dict('records')
            self.pipeline_state['X_val'] = X_val.to_dict('records')
            self.pipeline_state['X_test'] = X_test.to_dict('records')
            self.pipeline_state['y_train'] = y_train.to_dict()
            self.pipeline_state['y_val'] = y_val.to_dict()
            self.pipeline_state['y_test'] = y_test.to_dict()

            resp = {
                'status': 'success',
                'message': 'Data split successfully',
                'train_size': len(X_train),
                'validation_size': len(X_val),
                'test_size': len(X_test)
            }
            if self.request_data_loader:
                self.request_data_loader.log_api_activity(
                    endpoint='split_data_handler',
                    request_data=req_data,
                    response_data=resp,
                    status='success'
                )
            return jsonify(resp)
        except Exception as e:
            logging.error(f"Error splitting data: {str(e)}")
            resp = {'status': 'error', 'message': str(e)}
            if self.request_data_loader:
                self.request_data_loader.log_api_activity(
                    endpoint='split_data_handler',
                    request_data=req_data,
                    response_data=resp,
                    status='error',
                    error_message=str(e)
                )
            return jsonify(resp), 500

    def train_model_handler(self):
        """Train the model with hyperparameter tuning."""
        req_data = request.get_json() if request.is_json else None
        try:
            if 'X_train' not in self.pipeline_state or 'y_train' not in self.pipeline_state:
                resp = {'status': 'error', 'message': 'No training data found. Please split data first.'}
                if self.request_data_loader:
                    self.request_data_loader.log_api_activity(
                        endpoint='train_model_handler',
                        request_data=req_data,
                        response_data=resp,
                        status='error',
                        error_message=resp['message']
                    )
                return jsonify(resp), 400

            # Get parameters from request
            data = req_data or {}
            n_jobs = data.get('n_jobs', 5)

            # Reconstruct DataFrames
            X_train = pd.DataFrame(self.pipeline_state['X_train'])
            y_train = pd.Series(self.pipeline_state['y_train'])

            # Train model
            model = self.training_service.train_model(X_train, y_train, n_jobs)

            # Store trained model (serialize for storage)
            model_b64 = self._serialize_model(model)
            self.pipeline_state['trained_model'] = model_b64

            resp = {
                'status': 'success',
                'message': 'Model trained successfully',
                'model_type': type(model).__name__,
                'best_params': model.get_params()
            }
            if self.request_data_loader:
                self.request_data_loader.log_api_activity(
                    endpoint='train_model_handler',
                    request_data=req_data,
                    response_data=resp,
                    status='success'
                )
            return jsonify(resp)
        except Exception as e:
            logging.error(f"Error training model: {str(e)}")
            resp = {'status': 'error', 'message': str(e)}
            if self.request_data_loader:
                self.request_data_loader.log_api_activity(
                    endpoint='train_model_handler',
                    request_data=req_data,
                    response_data=resp,
                    status='error',
                    error_message=str(e)
                )
            return jsonify(resp), 500

    def evaluate_model_handler(self):
        """Evaluate the trained model."""
        req_data = request.get_json() if request.is_json else None
        try:
            if 'trained_model' not in self.pipeline_state:
                resp = {'status': 'error', 'message': 'No trained model found. Please train model first.'}
                if self.request_data_loader:
                    self.request_data_loader.log_api_activity(
                        endpoint='evaluate_model_handler',
                        request_data=req_data,
                        response_data=resp,
                        status='error',
                        error_message=resp['message']
                    )
                return jsonify(resp), 400

            # Reconstruct model
            model = self._deserialize_model(self.pipeline_state['trained_model'])

            # Reconstruct validation and test data
            X_val = pd.DataFrame(self.pipeline_state['X_val'])
            y_val = pd.Series(self.pipeline_state['y_val'])
            X_test = pd.DataFrame(self.pipeline_state['X_test'])
            y_test = pd.Series(self.pipeline_state['y_test'])
            df_processed = pd.DataFrame(self.pipeline_state['processed_data'])

            # Evaluate model
            results = self.evaluation_service.evaluate_model_comprehensive(
                model, X_val, y_val, X_test, y_test, df_processed
            )

            resp = {
                'status': 'success',
                'message': 'Model evaluated successfully',
                'evaluation_results': results
            }
            if self.request_data_loader:
                self.request_data_loader.log_api_activity(
                    endpoint='evaluate_model_handler',
                    request_data=req_data,
                    response_data=resp,
                    status='success'
                )
            return jsonify(resp)
        except Exception as e:
            logging.error(f"Error evaluating model: {str(e)}")
            resp = {'status': 'error', 'message': str(e)}
            if self.request_data_loader:
                self.request_data_loader.log_api_activity(
                    endpoint='evaluate_model_handler',
                    request_data=req_data,
                    response_data=resp,
                    status='error',
                    error_message=str(e)
                )
            return jsonify(resp), 500

    def save_model_handler(self):
        """Save the trained model."""
        req_data = request.get_json() if request.is_json else None
        try:
            if 'trained_model' not in self.pipeline_state:
                resp = {'status': 'error', 'message': 'No trained model found. Please train model first.'}
                if self.request_data_loader:
                    self.request_data_loader.log_api_activity(
                        endpoint='save_model_handler',
                        request_data=req_data,
                        response_data=resp,
                        status='error',
                        error_message=resp['message']
                    )
                return jsonify(resp), 400

            # Get parameters from request
            data = req_data or {}
            model_dir = data.get('model_dir', 'models')
            model_name = data.get('model_name', 'b3_model.joblib')

            # Reconstruct model
            model = self._deserialize_model(self.pipeline_state['trained_model'])

            # Save model
            model_path = self.saving_service.save_model(model, model_dir, model_name)

            resp = {
                'status': 'success',
                'message': 'Model saved successfully',
                'model_path': model_path
            }
            if self.request_data_loader:
                self.request_data_loader.log_api_activity(
                    endpoint='save_model_handler',
                    request_data=req_data,
                    response_data=resp,
                    status='success'
                )
            return jsonify(resp)
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")
            resp = {'status': 'error', 'message': str(e)}
            if self.request_data_loader:
                self.request_data_loader.log_api_activity(
                    endpoint='save_model_handler',
                    request_data=req_data,
                    response_data=resp,
                    status='error',
                    error_message=str(e)
                )
            return jsonify(resp), 500

    def train_complete_handler(self):
        """Run the complete training pipeline on a separate thread."""
        req_data = request.get_json() if request.is_json else None
        try:
            # Get parameters from request
            data = req_data or {}
            model_dir = data.get('model_dir', 'models')
            n_jobs = data.get('n_jobs', 5)
            test_size = data.get('test_size', 0.2)
            val_size = data.get('val_size', 0.2)

            # Submit the training task to thread pool
            future = self.executor.submit(self._run_complete_training_pipeline,
                                          model_dir, n_jobs, test_size, val_size)

            # Store the future for status checking
            self.pipeline_state['training_future'] = future
            self.pipeline_state['training_status'] = 'running'

            resp = {
                'status': 'success',
                'message': 'Complete training pipeline started in background',
                'training_status': 'running'
            }
            if self.request_data_loader:
                self.request_data_loader.log_api_activity(
                    endpoint='train_complete_handler',
                    request_data=req_data,
                    response_data=resp,
                    status='success'
                )
            return jsonify(resp)
        except Exception as e:
            logging.error(f"Error starting complete training pipeline: {str(e)}")
            resp = {'status': 'error', 'message': str(e)}
            if self.request_data_loader:
                self.request_data_loader.log_api_activity(
                    endpoint='train_complete_handler',
                    request_data=req_data,
                    response_data=resp,
                    status='error',
                    error_message=str(e)
                )
            return jsonify(resp), 500

    def _run_complete_training_pipeline(self, model_dir, n_jobs, test_size, val_size):
        """Internal method to run the complete training pipeline."""
        try:
            # Step 1: Load data
            df = self.data_loading_service.load_data()
            self.pipeline_state['raw_data'] = df.to_dict('records')
            self.pipeline_state['data_shape'] = df.shape

            # Step 2: Preprocess data
            X, df_processed, y = self.preprocessing_service.preprocess_data(df)
            self.pipeline_state['X_features'] = X.to_dict('records')
            self.pipeline_state['y_targets'] = y.to_dict()
            self.pipeline_state['processed_data'] = df_processed.to_dict('records')

            # Step 3: Split data
            X_train, X_val, X_test, y_train, y_val, y_test = self.training_service.split_data(
                X, y, test_size, val_size
            )
            self.pipeline_state['X_train'] = X_train.to_dict('records')
            self.pipeline_state['X_val'] = X_val.to_dict('records')
            self.pipeline_state['X_test'] = X_test.to_dict('records')
            self.pipeline_state['y_train'] = y_train.to_dict()
            self.pipeline_state['y_val'] = y_val.to_dict()
            self.pipeline_state['y_test'] = y_test.to_dict()

            # Step 4: Train model
            model = self.training_service.train_model(X_train, y_train, n_jobs)
            model_b64 = self._serialize_model(model)
            self.pipeline_state['trained_model'] = model_b64

            # Step 5: Evaluate model
            results = self.evaluation_service.evaluate_model_comprehensive(
                model, X_val, y_val, X_test, y_test, df_processed
            )

            # Step 6: Save model
            model_path = self.saving_service.save_model(model, model_dir)

            # Update status
            self.pipeline_state['training_status'] = 'completed'
            self.pipeline_state['training_results'] = {
                'model_path': model_path,
                'evaluation_results': results,
                'data_info': {
                    'total_samples': len(df),
                    'train_samples': len(X_train),
                    'validation_samples': len(X_val),
                    'test_samples': len(X_test)
                }
            }

            logging.info("Complete training pipeline finished successfully")
            return True

        except Exception as e:
            logging.error(f"Error in complete training pipeline: {str(e)}")
            self.pipeline_state['training_status'] = 'failed'
            self.pipeline_state['training_error'] = str(e)
            return False

    def get_status_handler(self):
        """Get current pipeline status."""
        req_data = request.get_json() if request.is_json else None
        try:
            status_info = {
                'has_raw_data': 'raw_data' in self.pipeline_state,
                'has_processed_data': 'X_features' in self.pipeline_state,
                'has_split_data': 'X_train' in self.pipeline_state,
                'has_trained_model': 'trained_model' in self.pipeline_state
            }

            # Add training status if available
            if 'training_status' in self.pipeline_state:
                status_info['training_status'] = self.pipeline_state['training_status']

            # Add training results if completed
            if 'training_results' in self.pipeline_state:
                status_info['training_results'] = self.pipeline_state['training_results']

            # Add training error if failed
            if 'training_error' in self.pipeline_state:
                status_info['training_error'] = self.pipeline_state['training_error']

            resp = {
                'status': 'success',
                'pipeline_state': status_info
            }
            if self.request_data_loader:
                self.request_data_loader.log_api_activity(
                    endpoint='get_status_handler',
                    request_data=req_data,
                    response_data=resp,
                    status='success'
                )
            return jsonify(resp)
        except Exception as e:
            logging.error(f"Error getting status: {str(e)}")
            resp = {'status': 'error', 'message': str(e)}
            if self.request_data_loader:
                self.request_data_loader.log_api_activity(
                    endpoint='get_status_handler',
                    request_data=req_data,
                    response_data=resp,
                    status='error',
                    error_message=str(e)
                )
            return jsonify(resp), 500

    def get_training_status_handler(self):
        """Get detailed training status for background training."""
        req_data = request.get_json() if request.is_json else None
        try:
            if 'training_status' not in self.pipeline_state:
                resp = {
                    'status': 'success',
                    'message': 'No training in progress',
                    'training_status': 'not_started'
                }
                if self.request_data_loader:
                    self.request_data_loader.log_api_activity(
                        endpoint='get_training_status_handler',
                        request_data=req_data,
                        response_data=resp,
                        status='success'
                    )
                return jsonify(resp)

            training_status = self.pipeline_state['training_status']
            response_data = {
                'status': 'success',
                'training_status': training_status
            }

            if training_status == 'running':
                response_data['message'] = 'Training is currently running in background'
            elif training_status == 'completed':
                response_data['message'] = 'Training completed successfully'
                if 'training_results' in self.pipeline_state:
                    response_data['results'] = self.pipeline_state['training_results']
            elif training_status == 'failed':
                response_data['message'] = 'Training failed'
                if 'training_error' in self.pipeline_state:
                    response_data['error'] = self.pipeline_state['training_error']

            if self.request_data_loader:
                self.request_data_loader.log_api_activity(
                    endpoint='get_training_status_handler',
                    request_data=req_data,
                    response_data=response_data,
                    status='success'
                )
            return jsonify(response_data)
        except Exception as e:
            logging.error(f"Error getting training status: {str(e)}")
            resp = {'status': 'error', 'message': str(e)}
            if self.request_data_loader:
                self.request_data_loader.log_api_activity(
                    endpoint='get_training_status_handler',
                    request_data=req_data,
                    response_data=resp,
                    status='error',
                    error_message=str(e)
                )
            return jsonify(resp), 500

    def clear_state_handler(self):
        """Clear the pipeline state."""
        req_data = request.get_json() if request.is_json else None
        try:
            # Cancel any running training if exists
            if 'training_future' in self.pipeline_state:
                future = self.pipeline_state['training_future']
                if not future.done():
                    future.cancel()

            self.pipeline_state = {}
            resp = {'status': 'success', 'message': 'Pipeline state cleared'}
            if self.request_data_loader:
                self.request_data_loader.log_api_activity(
                    endpoint='clear_state_handler',
                    request_data=req_data,
                    response_data=resp,
                    status='success'
                )
            return jsonify(resp)
        except Exception as e:
            logging.error(f"Error clearing state: {str(e)}")
            resp = {'status': 'error', 'message': str(e)}
            if self.request_data_loader:
                self.request_data_loader.log_api_activity(
                    endpoint='clear_state_handler',
                    request_data=req_data,
                    response_data=resp,
                    status='error',
                    error_message=str(e)
                )
            return jsonify(resp), 500

    def predict_data_handler(self):
        """Make predictions using the trained model."""
        req_data = request.get_json() if request.is_json else None
        try:
            model = None
            model_source = None

            # Check if there's a trained model in the pipeline state
            if 'trained_model' in self.pipeline_state:
                model = self._deserialize_model(self.pipeline_state['trained_model'])
                model_source = 'pipeline'
            else:
                # Fallback: check if there's a model in storage using the new storage system
                try:
                    if self.model_saving_service.model_exists():
                        # Use relative path for loading (not the full URL)
                        model_path = self.model_saving_service.get_model_relative_path()
                        model = self.model_saving_service.load_model(model_path)
                        model_source = 'storage'
                        logging.info(f"Loaded model from storage: {model_path}")
                    else:
                        resp = {'status': 'error',
                                'message': 'No trained model found in pipeline or storage. Please train model first.'}
                        if self.request_data_loader:
                            self.request_data_loader.log_api_activity(
                                endpoint='predict_data_handler',
                                request_data=req_data,
                                response_data=resp,
                                status='error',
                                error_message=resp['message']
                            )
                        return jsonify(resp), 400
                except Exception as e:
                    logging.error(f"Error loading model from storage: {str(e)}")
                    resp = {'status': 'error', 'message': f'Error loading model from storage: {str(e)}'}
                    if self.request_data_loader:
                        self.request_data_loader.log_api_activity(
                            endpoint='predict_data_handler',
                            request_data=req_data,
                            response_data=resp,
                            status='error',
                            error_message=str(e)
                        )
                    return jsonify(resp), 500

            # Get prediction parameters from request
            data = req_data or {}
            ticker = data.get('ticker')

            if not ticker:
                resp = {'status': 'error',
                        'message': 'No ticker provided. Please include "ticker" field in request.'}
                if self.request_data_loader:
                    self.request_data_loader.log_api_activity(
                        endpoint='predict_data_handler',
                        request_data=req_data,
                        response_data=resp,
                        status='error',
                        error_message=resp['message']
                    )
                return jsonify(resp), 400

            # Fetch new data for the ticker using the data loader (following _b3_predict pattern)
            data_loader = self.data_loading_service.get_loader()
            new_data = data_loader.fetch(ticker=ticker)

            if new_data.empty:
                resp = {'status': 'error',
                        'message': f'No data found for ticker: {ticker}'}
                if self.request_data_loader:
                    self.request_data_loader.log_api_activity(
                        endpoint='predict_data_handler',
                        request_data=req_data,
                        response_data=resp,
                        status='error',
                        error_message=resp['message']
                    )
                return jsonify(resp), 400

            # Extract numeric features (excluding 'date') - following _b3_predict pattern
            features = [col for col in new_data.select_dtypes(include=['number']).columns if col != 'date']
            X_new = new_data[features]

            # Get expected features - either from pipeline state or fallback to predefined feature set
            if 'X_features' in self.pipeline_state:
                expected_features = list(pd.DataFrame(self.pipeline_state['X_features']).columns)
            else:
                # Fallback: use the predefined feature set from preprocessing service
                preprocessing_service = B3ModelPreprocessingService()
                expected_features = preprocessing_service._FEATURE_SET
                logging.info(f"Using fallback feature set for file-based model: {expected_features}")

            # Filter to only include features that exist in both datasets
            available_features = [f for f in expected_features if f in X_new.columns]
            missing_features = set(expected_features) - set(available_features)

            if missing_features:
                resp = {
                    'status': 'error',
                    'message': f'Missing required features in data for ticker {ticker}: {list(missing_features)}',
                    'required_features': expected_features,
                    'available_features': list(X_new.columns)
                }
                if self.request_data_loader:
                    self.request_data_loader.log_api_activity(
                        endpoint='predict_data_handler',
                        request_data=req_data,
                        response_data=resp,
                        status='error',
                        error_message=resp['message']
                    )
                return jsonify(resp), 400

            # Use only the available features for prediction
            X_prediction = X_new[available_features]

            # Make predictions
            predictions = model.predict(X_prediction)

            # Get prediction probabilities if available
            prediction_probs = None
            if hasattr(model, 'predict_proba'):
                prediction_probs = model.predict_proba(X_prediction).tolist()

            # Get feature importance if available
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_.tolist()

            logging.info(f"Predicted actions for ticker {ticker}: {predictions}")

            resp = {
                'status': 'success',
                'message': f'Predictions generated successfully for ticker: {ticker}',
                'ticker': ticker,
                'predictions': predictions.tolist(),
                'prediction_probabilities': prediction_probs,
                'feature_importance': feature_importance,
                'input_shape': X_prediction.shape,
                'features_used': available_features,
                'model_type': type(model).__name__,
                'model_source': model_source
            }
            if self.request_data_loader:
                self.request_data_loader.log_api_activity(
                    endpoint='predict_data_handler',
                    request_data=req_data,
                    response_data=resp,
                    status='success'
                )
            return jsonify(resp)
        except Exception as e:
            logging.error(f"Error making predictions: {str(e)}")
            resp = {'status': 'error', 'message': str(e)}
            if self.request_data_loader:
                self.request_data_loader.log_api_activity(
                    endpoint='predict_data_handler',
                    request_data=req_data,
                    response_data=resp,
                    status='error',
                    error_message=str(e)
                )
            return jsonify(resp), 500

    @staticmethod
    def _serialize_model(model):
        """Serialize model to base64 string for storage."""
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        model_bytes = buffer.getvalue()
        return base64.b64encode(model_bytes).decode('utf-8')

    @staticmethod
    def _deserialize_model(model_b64):
        """Deserialize model from base64 string."""
        model_bytes = base64.b64decode(model_b64)
        buffer = io.BytesIO(model_bytes)
        return joblib.load(buffer)
