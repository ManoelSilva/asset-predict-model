import logging
from concurrent.futures import ThreadPoolExecutor

from asset_model_data_storage.data_storage_service import DataStorageService
from flask import request, jsonify

from b3.service.pipeline.model.training_config import TrainingConfig
from b3.service.pipeline.model.factory import ModelFactory
from b3.service.pipeline.model.manager import CompletePipelineManagerService
from b3.service.pipeline.model.utils import is_rf_model
from constants import HTTP_STATUS_INTERNAL_SERVER_ERROR, DEFAULT_TEST_SIZE, DEFAULT_VAL_SIZE, DEFAULT_N_JOBS, \
    MODEL_STORAGE_KEY


class CompleteTrainingHandler:
    def __init__(self, data_loading_service, preprocessing_service, pipeline_state, log_api_activity, serialize_model,
                 storage_service=None):
        self._data_loading_service = data_loading_service
        self._preprocessing_service = preprocessing_service
        self._pipeline_state = pipeline_state
        self._log_api_activity = log_api_activity
        self._serialize_model = serialize_model
        self._storage_service = storage_service or DataStorageService()
        self._manager_service = CompletePipelineManagerService(self._storage_service)
        self._executor = ThreadPoolExecutor(max_workers=1)

    def complete_training_handler(self):
        req_data = request.get_json() if request.is_json else None
        try:
            data = req_data or {}

            # Map API parameters to TrainingConfig
            config = TrainingConfig(
                model_dir=data.get('model_dir', 'models'),
                n_jobs=data.get('n_jobs', DEFAULT_N_JOBS),
                test_size=data.get('test_size', DEFAULT_TEST_SIZE),
                val_size=data.get('val_size', DEFAULT_VAL_SIZE),
                model_type=data.get('model_type', 'rf'),
                lookback=data.get('lookback', 32),
                horizon=data.get('horizon', 1),
                epochs=data.get('epochs', 25),
                batch_size=data.get('batch_size', 128),
                learning_rate=data.get('lr', 1e-3),
                units=data.get('units', 96),
                dropout=data.get('dropout', 0.2),
                price_col=data.get('price_col', 'close')
            )

            future = self._executor.submit(self._run_complete_training_pipeline, config)
            self._pipeline_state['training_future'] = future
            self._pipeline_state['training_status'] = 'running'
            resp = {
                'status': 'success',
                'message': 'Complete training pipeline started in background',
                'training_status': 'running',
                'model_type': config.model_type
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

    def _run_complete_training_pipeline(self, config: TrainingConfig):
        try:
            # Use injected services

            df = self._data_loading_service.load_data()
            self._pipeline_state['raw_data'] = df.to_dict('records')
            self._pipeline_state['data_shape'] = df.shape
            x, df_processed, y = self._preprocessing_service.preprocess_data(df)
            self._pipeline_state['X_features'] = x.to_dict('records')
            self._pipeline_state['y_targets'] = y.to_dict()
            self._pipeline_state['processed_data'] = df_processed.to_dict('records')

            model_path, evaluation_results = self._manager_service.run(
                config=config,
                df=df,
                X=x,
                y=y,
                df_processed=df_processed
            )

            # Store pipeline information in pipeline state
            self._pipeline_state['model_type'] = config.model_type
            self._pipeline_state['model_path'] = model_path

            if is_rf_model(config.model_type):
                try:
                    persist_service = ModelFactory.get_persist_service(config.model_type, self._storage_service)
                    model = persist_service.load_model(persist_service.get_model_path(config.model_dir))
                    model_b64 = self._serialize_model(model)
                    self._pipeline_state[MODEL_STORAGE_KEY] = model_b64
                except Exception as e:
                    logging.warning(f"Could not serialize model for pipeline state: {str(e)}")

            self._pipeline_state['training_status'] = 'completed'
            self._pipeline_state['training_results'] = {
                'model_path': model_path,
                'model_type': config.model_type,
                'evaluation': evaluation_results,
                'data_info': {
                    'total_samples': len(df),
                    'model_type': config.model_type
                }
            }
            logging.info(f"Complete training pipeline finished successfully for {config.model_type} model")
            return True
        except Exception as e:
            logging.error(f"Error in complete training pipeline: {str(e)}")
            self._pipeline_state['training_status'] = 'failed'
            self._pipeline_state['training_error'] = str(e)
            return False
