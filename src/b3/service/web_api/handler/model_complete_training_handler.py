import logging

from flask import request, jsonify
from concurrent.futures import ThreadPoolExecutor
from constants import HTTP_STATUS_INTERNAL_SERVER_ERROR, DEFAULT_TEST_SIZE, DEFAULT_VAL_SIZE, DEFAULT_N_JOBS, \
    MODEL_STORAGE_KEY


class CompleteTrainingHandler:
    def __init__(self, data_loading_service, preprocessing_service, training_service, evaluation_service,
                 saving_service, pipeline_state, log_api_activity, serialize_model):
        self.data_loading_service = data_loading_service
        self.preprocessing_service = preprocessing_service
        self.training_service = training_service
        self.evaluation_service = evaluation_service
        self.saving_service = saving_service
        self.pipeline_state = pipeline_state
        self._log_api_activity = log_api_activity
        self._serialize_model = serialize_model
        self.executor = ThreadPoolExecutor(max_workers=1)

    def complete_training_handler(self):
        req_data = request.get_json() if request.is_json else None
        try:
            data = req_data or {}
            model_dir = data.get('model_dir', 'models')
            n_jobs = data.get('n_jobs', DEFAULT_N_JOBS)
            test_size = data.get('test_size', DEFAULT_TEST_SIZE)
            val_size = data.get('val_size', DEFAULT_VAL_SIZE)
            future = self.executor.submit(self._run_complete_training_pipeline, model_dir, n_jobs, test_size, val_size)
            self.pipeline_state['training_future'] = future
            self.pipeline_state['training_status'] = 'running'
            resp = {
                'status': 'success',
                'message': 'Complete training pipeline started in background',
                'training_status': 'running'
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

    def _run_complete_training_pipeline(self, model_dir, n_jobs, test_size, val_size):
        try:
            df = self.data_loading_service.load_data()
            self.pipeline_state['raw_data'] = df.to_dict('records')
            self.pipeline_state['data_shape'] = df.shape
            x, df_processed, y = self.preprocessing_service.preprocess_data(df)
            self.pipeline_state['X_features'] = x.to_dict('records')
            self.pipeline_state['y_targets'] = y.to_dict()
            self.pipeline_state['processed_data'] = df_processed.to_dict('records')
            x_train, x_val, x_test, y_train, y_val, y_test = self.training_service.split_data(
                x, y, test_size, val_size
            )
            self.pipeline_state['X_train'] = x_train.to_dict('records')
            self.pipeline_state['X_val'] = x_val.to_dict('records')
            self.pipeline_state['X_test'] = x_test.to_dict('records')
            self.pipeline_state['y_train'] = y_train.to_dict()
            self.pipeline_state['y_val'] = y_val.to_dict()
            self.pipeline_state['y_test'] = y_test.to_dict()
            model = self.training_service.train_model(x_train, y_train, n_jobs)
            model_b64 = self._serialize_model(model)
            self.pipeline_state[MODEL_STORAGE_KEY] = model_b64
            results = self.evaluation_service.evaluate_model_comprehensive(
                model, x_val, y_val, x_test, y_test, df_processed
            )
            model_path = self.saving_service.save_model(model, model_dir)
            self.pipeline_state['training_status'] = 'completed'
            self.pipeline_state['training_results'] = {
                'model_path': model_path,
                'evaluation_results': results,
                'data_info': {
                    'total_samples': len(df),
                    'train_samples': len(x_train),
                    'validation_samples': len(x_val),
                    'test_samples': len(x_test)
                }
            }
            logging.info("Complete training pipeline finished successfully")
            return True
        except Exception as e:
            logging.error(f"Error in complete training pipeline: {str(e)}")
            self.pipeline_state['training_status'] = 'failed'
            self.pipeline_state['training_error'] = str(e)
            return False
