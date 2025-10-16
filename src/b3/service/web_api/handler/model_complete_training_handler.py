import logging
from concurrent.futures import ThreadPoolExecutor

from flask import request, jsonify

from constants import HTTP_STATUS_INTERNAL_SERVER_ERROR, DEFAULT_TEST_SIZE, DEFAULT_VAL_SIZE, DEFAULT_N_JOBS, \
    MODEL_STORAGE_KEY


class CompleteTrainingHandler:
    def __init__(self, data_loading_service, preprocessing_service, pipeline_state, log_api_activity, serialize_model):
        self._data_loading_service = data_loading_service
        self._preprocessing_service = preprocessing_service
        self._pipeline_state = pipeline_state
        self._log_api_activity = log_api_activity
        self._serialize_model = serialize_model
        self._executor = ThreadPoolExecutor(max_workers=1)

    def complete_training_handler(self):
        req_data = request.get_json() if request.is_json else None
        try:
            data = req_data or {}
            model_dir = data.get('model_dir', 'models')
            n_jobs = data.get('n_jobs', DEFAULT_N_JOBS)
            test_size = data.get('test_size', DEFAULT_TEST_SIZE)
            val_size = data.get('val_size', DEFAULT_VAL_SIZE)
            model_type = data.get('model_type', 'rf')
            # LSTM-specific parameters
            lookback = data.get('lookback', 32)
            horizon = data.get('horizon', 1)
            epochs = data.get('epochs', 25)
            batch_size = data.get('batch_size', 128)
            lr = data.get('lr', 1e-3)
            units = data.get('units', 96)
            dropout = data.get('dropout', 0.2)
            price_col = data.get('price_col', 'close')

            future = self._executor.submit(self._run_complete_training_pipeline,
                                           model_dir, n_jobs, test_size, val_size, model_type,
                                           lookback, horizon, epochs, batch_size, lr, units, dropout, price_col)
            self._pipeline_state['training_future'] = future
            self._pipeline_state['training_status'] = 'running'
            resp = {
                'status': 'success',
                'message': 'Complete training pipeline started in background',
                'training_status': 'running',
                'model_type': model_type
            }
            self._log_api_activity(
                endpoint='train_complete_handler',
                request_data=req_data,
                response_data=resp,
                status='success'
            )
            return jsonify(resp)
        except Exception as e:
            logging.error(f"Error starting complete training pipeline: {str(e)}")
            resp = {'status': 'error', 'message': str(e)}
            self._log_api_activity(
                endpoint='train_complete_handler',
                request_data=req_data,
                response_data=resp,
                status='error',
                error_message=str(e)
            )
            return jsonify(resp), HTTP_STATUS_INTERNAL_SERVER_ERROR

    def _run_complete_training_pipeline(self, model_dir, n_jobs, test_size, val_size, model_type,
                                        lookback, horizon, epochs, batch_size, lr, units, dropout, price_col):
        try:
            # Use injected services and create storage service at runtime
            from b3.service.pipeline.model.factory import ModelFactory
            from asset_model_data_storage.data_storage_service import DataStorageService

            storage_service = DataStorageService()

            df = self._data_loading_service.load_data()
            self._pipeline_state['raw_data'] = df.to_dict('records')
            self._pipeline_state['data_shape'] = df.shape
            x, df_processed, y = self._preprocessing_service.preprocess_data(df)
            self._pipeline_state['X_features'] = x.to_dict('records')
            self._pipeline_state['y_targets'] = y.to_dict()
            self._pipeline_state['processed_data'] = df_processed.to_dict('records')

            # Use the B3Model class with factory pattern for training
            from b3.service.pipeline.model import B3Model
            b3_model = B3Model()

            # Train using the new factory pattern
            model_path = b3_model.train(
                model_dir=model_dir,
                n_jobs=n_jobs,
                test_size=test_size,
                val_size=val_size,
                model_type=model_type,
                lookback=lookback,
                horizon=horizon,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                units=units,
                dropout=dropout,
                price_col=price_col
            )

            # Store pipeline information in pipeline state
            self._pipeline_state['model_type'] = model_type
            self._pipeline_state['model_path'] = model_path

            # For backward compatibility, try to load and serialize the pipeline if it's a Random Forest
            if model_type in ["rf", "random_forest"]:
                try:
                    # Get persist service at runtime
                    persist_service = ModelFactory.get_persist_service(model_type, storage_service)
                    model = persist_service.load_model(persist_service.get_model_path(model_dir))
                    model_b64 = self._serialize_model(model)
                    self._pipeline_state[MODEL_STORAGE_KEY] = model_b64
                except Exception as e:
                    logging.warning(f"Could not serialize model for pipeline state: {str(e)}")

            self._pipeline_state['training_status'] = 'completed'
            self._pipeline_state['training_results'] = {
                'model_path': model_path,
                'model_type': model_type,
                'data_info': {
                    'total_samples': len(df),
                    'model_type': model_type
                }
            }
            logging.info(f"Complete training pipeline finished successfully for {model_type} model")
            return True
        except Exception as e:
            logging.error(f"Error in complete training pipeline: {str(e)}")
            self._pipeline_state['training_status'] = 'failed'
            self._pipeline_state['training_error'] = str(e)
            return False
