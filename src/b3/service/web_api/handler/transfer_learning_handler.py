import logging
import pandas as pd
from flask import request, jsonify

from b3.service.data.db.prediction.prediction_loader import PredictionDataLoader
from b3.service.pipeline.model.factory import ModelFactory
from b3.service.pipeline.model.lstm.lstm_model import LSTMModel
from b3.utils.mlflow_utils import MlFlowUtils
from asset_model_data_storage.data_storage_service import DataStorageService
from constants import FEATURE_SET


class TransferLearningHandler:
    """
    Handler for Transfer Learning (Fine-Tuning) of LSTM models.
    """

    def __init__(self, pipeline_state, log_api_activity, storage_service=None):
        self._pipeline_state = pipeline_state
        self._log_api_activity = log_api_activity
        self._storage_service = storage_service or DataStorageService()
        self._prediction_loader = PredictionDataLoader()

    def run_transfer_learning(self):
        """
        Run transfer learning pipeline:
        1. Load labeled data from b3_predictions (joined with ground truth)
        2. Load latest LSTM model
        3. Prepare data sequences
        4. Fine-tune model
        5. Evaluate and save
        """
        try:
            data = request.get_json() or {}
            epochs = data.get('epochs', 5)
            lr = data.get('learning_rate', 0.0001)
            ticker_filter = data.get('ticker_filter', [])

            logging.info(f"Starting transfer learning with epochs={epochs}, lr={lr}")

            # 1. Load Data
            df = self._prediction_loader.fetch_labeled_data_for_retraining(ticker_filter)
            if df.empty:
                return jsonify({'status': 'error', 'message': 'No labeled data found for transfer learning. Ensure predictions exist and have corresponding ground truth in b3_featured.'}), 400

            logging.info(f"Loaded {len(df)} rows for transfer learning")

            # 2. Load Model
            model_type = 'lstm'
            model_instance = ModelFactory.get_model(model_type)
            
            persist_service = ModelFactory.get_persist_service(model_type, self._storage_service)

            if not persist_service.model_exists():
                return jsonify({'status': 'error', 'message': 'No base LSTM model found to fine-tune.'}), 400

            model_path = persist_service.get_model_relative_path()
            # This sets model_instance.model to the loaded B3PytorchMTLModel
            model_instance.load_model(model_path)

            if not isinstance(model_instance, LSTMModel):
                return jsonify({'status': 'error', 'message': 'Model instance is not LSTMModel.'}), 500

            # 3. Prepare Data
            # Validate features
            valid_features = [f for f in FEATURE_SET if f in df.columns]
            if not valid_features:
                return jsonify({'status': 'error', 'message': 'Missing required features in data.'}), 400

            X = df[valid_features]
            y = df['target'] if 'target' in df.columns else pd.Series(index=df.index) # Targets needed for creating y sequences

            # Use config from the loaded model (wrapper) to ensure compatibility
            loaded_config = model_instance.model.config
            
            # Prepare sequences
            X_seq, yA_seq, yR_seq, p0_seq, pf_seq = model_instance.prepare_data(
                X, y, df_processed=df, 
                lookback=loaded_config.lookback, 
                horizon=loaded_config.horizon,
                price_col=loaded_config.price_col if hasattr(loaded_config, 'price_col') else 'close'
            )

            if len(X_seq) == 0:
                return jsonify({'status': 'error', 'message': 'Not enough data to build sequences (check lookback/horizon vs data size).'}), 400

            # 4. Fine Tune
            # Setup config for logging
            training_config = loaded_config
            training_config.epochs = epochs
            training_config.learning_rate = lr
            
            MlFlowUtils.start_run(training_config)
            
            # Fine tune
            model_instance.fine_tune_model(X_seq, yA_seq, yR_seq, epochs=epochs, learning_rate=lr)

            # 5. Evaluate
            # Split data for evaluation
            (X_train, X_val, X_test,
             yA_train, yA_val, yA_test,
             yR_train, yR_val, yR_test,
             p0_train, p0_val, p0_test,
             pf_train, pf_val, pf_test) = model_instance.split_data(
                X_seq, yA_seq, yR_seq, p0_seq, pf_seq,
                test_size=0.2, val_size=0.2
            )

            evaluation_results = model_instance.evaluate_model(
                model_instance.model, X_val, yA_val, X_test, yA_test, df,
                p0_val=p0_val, pf_val=pf_val, p0_test=p0_test, pf_test=pf_test,
                model_name='b3_lstm_mtl_transfer'
            )

            # 6. Save
            new_model_path = model_instance.save_model(model_instance.model, training_config.model_dir)

            # Log
            MlFlowUtils.log_result(new_model_path, evaluation_results)
            MlFlowUtils.end_run()

            resp = {
                'status': 'success',
                'message': 'Transfer learning completed successfully.',
                'metrics': evaluation_results,
                'model_path': new_model_path,
                'samples_used': len(X_seq)
            }
            
            self._log_api_activity(
                endpoint='run_transfer_learning',
                request_data=data,
                response_data=resp,
                status='success'
            )
            
            return jsonify(resp)

        except Exception as e:
            logging.error(f"Transfer learning failed: {str(e)}")
            resp = {'status': 'error', 'message': str(e)}
            self._log_api_activity(
                endpoint='run_transfer_learning',
                request_data=request.get_json() or {},
                response_data=resp,
                status='error'
            )
            return jsonify(resp), 500

