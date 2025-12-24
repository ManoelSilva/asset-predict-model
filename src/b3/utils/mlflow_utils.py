from datetime import datetime

import mlflow
from b3.service.pipeline.model.training_config import TrainingConfig
from b3.service.pipeline.model.utils import is_rf_model, is_lstm_model


class MlFlowUtils:
    @staticmethod
    def get_experiment_name(config: TrainingConfig) -> str:
        """
        Generate a deterministic experiment name based on the configuration.

        Pattern:
        Base: asset-prediction/{model_type}/{asset}/{horizon}
        RF:   .../features-{feature_set_version}
        LSTM: .../lb{lookback_window}
        """
        base_name = f"asset-prediction/{config.model_type}/{config.asset}/h{config.horizon}"

        if is_rf_model(config.model_type):
            return f"{base_name}/features-{config.feature_set_version}"
        elif is_lstm_model(config.model_type):
            return f"{base_name}/lb{config.lookback}"

        return base_name

    @staticmethod
    def start_run(config: TrainingConfig):
        experiment_name = MlFlowUtils.get_experiment_name(config)
        mlflow.set_experiment(experiment_name)

        active_run = mlflow.active_run()

        if active_run is None:
            mlflow.start_run()
            MlFlowUtils.log_param("train_start_date", datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
            MlFlowUtils.log_config_params(config)

    @staticmethod
    def log_config_params(config: TrainingConfig):
        mlflow.log_param("model_type", config.model_type)
        mlflow.log_param("asset", config.asset)
        mlflow.log_param("prediction_horizon", config.horizon)
        mlflow.log_param("feature_set_version", config.feature_set_version)

        if is_rf_model(config.model_type):
            mlflow.log_params(config.get_rf_config_dict())
        elif is_lstm_model(config.model_type):
            mlflow.log_params(config.get_lstm_config_dict())

    @staticmethod
    def log_param(param: str, value: str):
        mlflow.log_param(param, value)

    @staticmethod
    def log_result(model_path: str, evaluation_results: dict):
        # --- MLflow Logging Results ---

        # Log Artifacts
        mlflow.log_artifact(model_path)

        # Log Visualization if available
        vis_path = None
        if 'visualization_path' in evaluation_results:
            vis_path = evaluation_results['visualization_path']
        elif 'classification' in evaluation_results and isinstance(evaluation_results['classification'],
                                                                   dict) and 'visualization_path' in \
                evaluation_results['classification']:
            vis_path = evaluation_results['classification']['visualization_path']

        if vis_path:
            mlflow.log_artifact(vis_path)

        # Log Metrics
        # Flatten evaluation results
        if 'training_history' in evaluation_results:
            hist = evaluation_results['training_history']
            if 'loss' in hist and hist['loss']:
                mlflow.log_metric("final_train_loss", hist['loss'][-1])
            if 'val_loss' in hist and hist['val_loss']:
                mlflow.log_metric("final_val_loss", hist['val_loss'][-1])
                mlflow.log_metric("best_val_loss", min(hist['val_loss']))
                # Overfitting gap (last val loss - last train loss)
                if 'loss' in hist and hist['loss']:
                    mlflow.log_metric("overfitting_gap", hist['val_loss'][-1] - hist['loss'][-1])

        if 'regression' in evaluation_results:
            reg = evaluation_results['regression']
            if 'validation_regression' in reg:
                for k, v in reg['validation_regression'].items():
                    mlflow.log_metric(f"val_{k}", v)
            if 'test_regression' in reg:
                for k, v in reg['test_regression'].items():
                    mlflow.log_metric(f"test_{k}", v)

        # Classification metrics for RF or LSTM
        if 'classification' in evaluation_results:
            class_res = evaluation_results['classification']
        else:
            class_res = evaluation_results  # RF style

        if isinstance(class_res, dict):
            if 'validation' in class_res and isinstance(class_res['validation'], dict):
                # Log macro avg or weighted avg
                val_res = class_res['validation']
                if 'accuracy' in val_res:
                    mlflow.log_metric("val_accuracy", val_res['accuracy'])
                if 'macro avg' in val_res and isinstance(val_res['macro avg'], dict):
                    mlflow.log_metric("val_f1_macro", val_res['macro avg'].get('f1-score', 0))

            if 'test' in class_res and isinstance(class_res['test'], dict):
                test_res = class_res['test']
                if 'accuracy' in test_res:
                    mlflow.log_metric("test_accuracy", test_res['accuracy'])
                if 'macro avg' in test_res and isinstance(test_res['macro avg'], dict):
                    mlflow.log_metric("test_f1_macro", test_res['macro avg'].get('f1-score', 0))

    @staticmethod
    def end_run():
        MlFlowUtils.log_param("train_end_date", datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
        mlflow.end_run()
