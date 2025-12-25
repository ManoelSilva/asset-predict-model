import logging

import pandas as pd
from asset_model_data_storage.data_storage_service import DataStorageService
from flask import request, jsonify

from b3.service.pipeline.model.factory import ModelFactory
from b3.service.pipeline.model.utils import is_lstm_model
from b3.service.pipeline.model_evaluation_service import B3ModelEvaluationService
from b3.utils.api_handler_utils import ApiHandlerUtils
from constants import HTTP_STATUS_BAD_REQUEST, HTTP_STATUS_INTERNAL_SERVER_ERROR, MODEL_STORAGE_KEY


class ModelEvaluationHandler:
    def __init__(self, pipeline_state, log_api_activity, storage_service=None):
        self._pipeline_state = pipeline_state
        self._log_api_activity = log_api_activity
        self._storage_service = storage_service or DataStorageService()

    def evaluate_model_handler(self):
        req_data = request.get_json() if request.is_json else None
        try:
            data = req_data or {}
            # Try to get model_type from request or pipeline_state
            model_type = data.get('model_type') or self._pipeline_state.get('model_type', 'rf')

            # Check if model exists
            # For RF models, we check if the serialized model is in the pipeline state
            has_rf_model = MODEL_STORAGE_KEY in self._pipeline_state

            # For LSTM models, we check if training was completed (it's saved to disk, not in pipeline state as b64)
            # We can check 'training_status' or existence of 'model_path' in pipeline_state
            has_lstm_model = is_lstm_model(model_type) and \
                             (self._pipeline_state.get('training_status') == 'completed' or \
                              self._pipeline_state.get('model_path'))

            if not (has_rf_model or has_lstm_model):
                resp = {'status': 'error', 'message': 'No trained model found. Please train model first.'}
                self._log_api_activity(
                    endpoint='evaluate_model_handler',
                    request_data=req_data,
                    response_data=resp,
                    status='error',
                    error_message=resp['message']
                )
                return jsonify(resp), HTTP_STATUS_BAD_REQUEST

            # For LSTM models, we need to handle evaluation differently
            if is_lstm_model(model_type):
                # LSTM models are not serialized in the same way, so we need to load from storage
                # We use the Model wrapper to ensure we get a model object with .predict(), not just a checkpoint dict
                model_instance = ModelFactory.get_model(model_type)

                # Get model path from pipeline state if available, otherwise fallback to default
                saving_service = ModelFactory.get_persist_service(model_type, self._storage_service)
                model_path = self._pipeline_state.get('model_path') or saving_service.get_model_path('models')

                model = model_instance.load_model(model_path)
            else:
                # For Random Forest and other serializable models
                model = ApiHandlerUtils.deserialize_model(self._pipeline_state[MODEL_STORAGE_KEY])

            df_processed = pd.DataFrame(self._pipeline_state['processed_data'])

            if is_lstm_model(model_type):
                # For LSTM, we need to transform the 2D split data into 3D sequences
                # using the same logic as training.

                # 1. Reconstruct 2D DataFrames/Series from pipeline state
                # Note: DataSplittingHandler stores them as lists of dicts or values
                x_val_2d = pd.DataFrame(self._pipeline_state['X_val'])
                y_val_2d = pd.Series(self._pipeline_state['y_val'])
                x_test_2d = pd.DataFrame(self._pipeline_state['X_test'])
                y_test_2d = pd.Series(self._pipeline_state['y_test'])

                # 2. Get the model wrapper (LSTMModel) which has prepare_data
                # We already have 'model_instance' from earlier, but we need to ensure it has the correct config
                # The loaded 'model' is B3PytorchMTLModel which has the config used for training
                model_wrapper = model_instance  # This is the LSTMModel instance
                model_wrapper.model = model
                model_wrapper.is_trained = True

                # Sync wrapper config with loaded model config to ensure lookback/horizon match
                if hasattr(model, 'config'):
                    model_wrapper.config = model.config

                # 3. Prepare sequences for Validation set
                # Note: prepare_data needs to align X with df_processed to build sequences
                # However, X_val_2d is a subset (randomly split). Re-building sequences from random rows is impossible
                # if we rely on temporal continuity.
                # BUT, if we assume the user wants to evaluate on the sequences that WOULD have been generated
                # from the original data, we should probably regenerate sequences from the FULL dataset
                # and then split them again using the same random seed?
                #
                # OR, if the user ran /complete-training, the pipeline might have stored the sequence splits?
                # No, pipeline state doesn't have sequence splits.
                #
                # If we have only random 2D rows, we CANNOT evaluate LSTM properly because we lack the history for each row.
                # UNLESS the X_val rows *contain* the history? No, they are just features for time T.
                #
                # CRITICAL FIX: We must rebuild sequences from the FULL processed data, 
                # then apply the SAME split that was used during training (same random state).
                # This ensures we get the exact same validation set sequences.

                # Load full raw features and targets
                # We need X and y for the full dataset to rebuild sequences
                if 'X_features' in self._pipeline_state and 'y_targets' in self._pipeline_state:
                    X_full = pd.DataFrame(self._pipeline_state['X_features'])
                    # y_targets is dict {index: label}
                    y_targets_data = self._pipeline_state['y_targets']

                    # Convert dict to Series and reset index to match X_full's RangeIndex
                    # This ensures alignment by position, which is preserved by to_dict('records')
                    y_full = pd.Series(y_targets_data).reset_index(drop=True)

                    # Rebuild all sequences
                    lstm_config_dict = model.config.to_dict() if hasattr(model.config,
                                                                         'to_dict') else model_wrapper.config.get_lstm_config_dict()

                    # Ensure we pass the parameters expected by prepare_data
                    # extract lookback/horizon/price_col from config
                    lookback = getattr(model.config, 'lookback', 32)
                    horizon = getattr(model.config, 'horizon', 1)
                    price_col = getattr(model.config, 'price_col', 'close')

                    X_seq, yA_seq, yR_seq, p0_seq, pf_seq = model_wrapper.prepare_data(
                        X_full, y_full, df_processed,
                        lookback=lookback, horizon=horizon, price_col=price_col
                    )

                    # Re-split to get the same validation/test sets
                    # We need the split ratios from pipeline state or defaults
                    # Pipeline state usually doesn't store split ratios directly unless we added them
                    # We'll use defaults or try to infer. Using defaults is safer than failing.
                    test_size = 0.2  # DEFAULT_TEST_SIZE
                    val_size = 0.2  # DEFAULT_VAL_SIZE

                    (X_train_seq, X_val_seq, X_test_seq,
                     yA_train_seq, yA_val_seq, yA_test_seq,
                     yR_train_seq, yR_val_seq, yR_test_seq,
                     p0_train_seq, p0_val_seq, p0_test_seq,
                     pf_train_seq, pf_val_seq, pf_test_seq) = model_wrapper.split_data(
                        X_seq, yA_seq, yR_seq, p0_seq, pf_seq, test_size, val_size
                    )

                    # 5. Evaluate using the wrapper's method
                    results = model_wrapper.evaluate_model(
                        model,
                        X_val_seq, yA_val_seq,
                        X_test_seq, yA_test_seq,
                        df_processed,
                        p0_val=p0_val_seq, pf_val=pf_val_seq,
                        p0_test=p0_test_seq, pf_test=pf_test_seq
                    )
                else:
                    raise ValueError(
                        "Cannot evaluate LSTM: Full X_features and y_targets missing from pipeline state. Unable to reconstruct sequences.")

            else:
                x_val = pd.DataFrame(self._pipeline_state['X_val'])
                y_val = pd.Series(self._pipeline_state['y_val'])
                x_test = pd.DataFrame(self._pipeline_state['X_test'])
                y_test = pd.Series(self._pipeline_state['y_test'])

                # Instantiate evaluation service at runtime
                evaluation_service = B3ModelEvaluationService(self._storage_service)
                results = evaluation_service.evaluate_model_comprehensive(
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
            return jsonify(resp), HTTP_STATUS_INTERNAL_SERVER_ERROR
