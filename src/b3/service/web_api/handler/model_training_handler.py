import logging
import pandas as pd
import numpy as np

from concurrent.futures import ThreadPoolExecutor

from asset_model_data_storage.data_storage_service import DataStorageService
from flask import request, jsonify

from b3.service.pipeline.model.factory import ModelFactory
from b3.service.pipeline.model.manager import ModelManagerService
from b3.service.pipeline.model.utils import is_rf_model, is_lstm_model, normalize_model_type
from constants import HTTP_STATUS_INTERNAL_SERVER_ERROR, DEFAULT_N_JOBS, MODEL_STORAGE_KEY


class ModelTrainingHandler:
    def __init__(self, pipeline_state, log_api_activity, serialize_model, storage_service=None):
        self._pipeline_state = pipeline_state
        self._log_api_activity = log_api_activity
        self._serialize_model = serialize_model
        self._storage_service = storage_service or DataStorageService()
        self._model_manager_service = ModelManagerService(self._storage_service)
        self._executor = ThreadPoolExecutor(max_workers=1)

    def train_model_handler(self):
        req_data = request.get_json() if request.is_json else None
        try:
            if 'X_train' not in self._pipeline_state or 'y_train' not in self._pipeline_state:
                resp = {'status': 'error', 'message': 'No training data found. Please split data first.'}
                self._log_api_activity(
                    endpoint='train_model_handler',
                    request_data=req_data,
                    response_data=resp,
                    status='error',
                    error_message=resp['message']
                )
                return jsonify(resp), 400

            data = req_data or {}
            n_jobs = data.get('n_jobs', DEFAULT_N_JOBS)
            model_dir = data.get('model_dir', 'models')
            model_type = data.get('model_type', 'rf')
            model_type = normalize_model_type(model_type)

            # Start training in background
            future = self._executor.submit(self._run_training_pipeline, model_type, n_jobs, model_dir)
            self._pipeline_state['training_future'] = future
            self._pipeline_state['training_status'] = 'running'

            resp = {
                'status': 'success',
                'message': 'Model training started in background',
                'training_status': 'running',
                'model_type': model_type
            }
            self._log_api_activity(
                endpoint='train_model_handler',
                request_data=req_data,
                response_data=resp,
                status='success'
            )
            return jsonify(resp)
        except Exception as e:
            logging.error(f"Error starting model training: {str(e)}")
            resp = {'status': 'error', 'message': str(e)}
            self._log_api_activity(
                endpoint='train_model_handler',
                request_data=req_data,
                response_data=resp,
                status='error',
                error_message=str(e)
            )
            return jsonify(resp), HTTP_STATUS_INTERNAL_SERVER_ERROR

    def _run_training_pipeline(self, model_type: str, n_jobs: int, model_dir: str):
        try:
            # Create model instance using factory
            model = ModelFactory.get_model(model_type)

            # Handle different model types
            if is_rf_model(model_type):
                # Get training data from pipeline state for RF
                x_train = pd.DataFrame(self._pipeline_state['X_train'])
                y_train = pd.Series(self._pipeline_state['y_train'])

                # Prepare data if needed
                if hasattr(model, 'prepare_data'):
                    x_train, y_train = model.prepare_data(x_train, y_train)

                # Train model
                trained_model = model.train_model(x_train, y_train, n_jobs=n_jobs)
                train_size = len(x_train)

            elif is_lstm_model(model_type):
                # For LSTM, we need sequences - check if they're in pipeline_state
                # If not, we need to prepare them from the existing data
                if 'X_train_seq' in self._pipeline_state:
                    # Sequences already prepared
                    x_train = np.array(self._pipeline_state['X_train_seq'])
                    yA_train = pd.Series(self._pipeline_state['yA_train'])
                    yR_train = np.array(self._pipeline_state['yR_train'])
                else:
                    # Need to prepare sequences from existing data
                    # This requires processed data and features
                    if 'X_features' not in self._pipeline_state or 'y_targets' not in self._pipeline_state:
                        raise ValueError(
                            "For LSTM models, data must be prepared as sequences or X_features/y_targets must be available")

                    x_train = pd.DataFrame(self._pipeline_state['X_features'])

                    # Reconstruct y_train ensuring alignment with x_train (which has RangeIndex from records)
                    # y_targets dict keys might be original indices (strings from JSON), so we reset index
                    y_train = pd.Series(self._pipeline_state['y_targets'])
                    y_train = y_train.reset_index(drop=True)

                    df_processed = pd.DataFrame(self._pipeline_state.get('processed_data', []))

                    if df_processed.empty:
                        raise ValueError("For LSTM models, processed_data must be available in pipeline_state")

                    # Prepare sequences
                    x_train, yA_train, yR_train, _, _ = model.prepare_data(
                        x_train, y_train, df_processed
                    )

                # Train model
                trained_model = model.train_model(x_train, yA_train, yR_train, lookback=model.config.lookback,
                                                  n_features=x_train.shape[2])
                train_size = len(x_train)

            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            # Store model_type in pipeline_state
            self._pipeline_state['model_type'] = model_type

            # Store trained model in pipeline state
            if is_rf_model(model_type):
                # Serialize Random Forest models
                model_b64 = self._serialize_model(trained_model)
                self._pipeline_state[MODEL_STORAGE_KEY] = model_b64

            # Always save model to disk (especially important for LSTM)
            model_path = model.save_model(trained_model, model_dir)
            self._pipeline_state['model_path'] = model_path

            if not is_rf_model(model_type):
                # For LSTM models, store the model type
                self._pipeline_state['trained_model_type'] = model_type

            # Store training results
            self._pipeline_state['training_status'] = 'completed'
            self._pipeline_state['training_results'] = {
                'model_path': model_path,
                'model_type': model_type,
                'best_params': trained_model.get_params() if hasattr(trained_model, 'get_params') else {},
                'train_size': train_size
            }

            logging.info(f"Model training completed successfully for {model_type} model")
            return True
        except Exception as e:
            logging.error(f"Error in model training pipeline: {str(e)}")
            self._pipeline_state['training_status'] = 'failed'
            self._pipeline_state['training_error'] = str(e)
            return False
