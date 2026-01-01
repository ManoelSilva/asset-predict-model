from datetime import datetime
from dataclasses import fields

import mlflow
from b3.service.pipeline.model.training_config import TrainingConfig
from b3.service.pipeline.model.utils import is_rf_model, is_lstm_model


class MlFlowUtils:
    @staticmethod
    def _format_config_value(value) -> str:
        """
        Format a configuration value for use in experiment name.
        
        Args:
            value: Configuration value (str, int, float, etc.)
            
        Returns:
            Formatted string value safe for experiment names
        """
        if isinstance(value, float):
            # Convert float to string with fixed precision, remove trailing zeros
            # Replace dots with underscores for filesystem compatibility
            str_value = f"{value:.10f}".rstrip('0').rstrip('.')
            return str_value.replace('.', '_')
        return str(value)

    @staticmethod
    def get_experiment_name(config: TrainingConfig) -> str:
        """
        Generate a deterministic experiment name based on the configuration.
        """
        config_fields = fields(config)

        # Build experiment name
        name_parts = []
        for field in config_fields:
            field_name = field.name
            if field_name == 'model_dir':
                continue
            field_value = getattr(config, field_name)
            formatted_value = MlFlowUtils._format_config_value(field_value)
            name_parts.append(f"{field_name}_{formatted_value}")

        return "_".join(name_parts)

    @staticmethod
    def get_run_name(config: TrainingConfig) -> str:
        """
        Generate a deterministic run name based on the configuration.
        
        Returns:
            Run name string with all config values
        """
        return MlFlowUtils.get_experiment_name(config)

    @staticmethod
    def start_run(config: TrainingConfig):
        experiment_name = MlFlowUtils.get_experiment_name(config)
        mlflow.set_experiment(experiment_name)

        active_run = mlflow.active_run()

        if active_run is None:
            run_name = MlFlowUtils.get_run_name(config)
            mlflow.start_run(run_name=run_name)
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
            # Helper to log report metrics
            def log_report_metrics(report: dict, prefix: str):
                if 'accuracy' in report:
                    mlflow.log_metric(f"{prefix}_accuracy", report['accuracy'])

                # Global averages
                for avg_name in ['macro avg', 'weighted avg']:
                    if avg_name in report:
                        avg_dict = report[avg_name]
                        slug = avg_name.replace(' ', '_')
                        mlflow.log_metric(f"{prefix}_{slug}_f1", avg_dict.get('f1-score', 0))
                        mlflow.log_metric(f"{prefix}_{slug}_precision", avg_dict.get('precision', 0))
                        mlflow.log_metric(f"{prefix}_{slug}_recall", avg_dict.get('recall', 0))

                # Per-class metrics
                for label in ['buy', 'sell', 'hold']:
                    if label in report:
                        label_dict = report[label]
                        mlflow.log_metric(f"{prefix}_class_{label}_f1", label_dict.get('f1-score', 0))
                        mlflow.log_metric(f"{prefix}_class_{label}_precision", label_dict.get('precision', 0))
                        mlflow.log_metric(f"{prefix}_class_{label}_recall", label_dict.get('recall', 0))

            if 'validation' in class_res and isinstance(class_res['validation'], dict):
                log_report_metrics(class_res['validation'], "val")

            if 'test' in class_res and isinstance(class_res['test'], dict):
                log_report_metrics(class_res['test'], "test")

    @staticmethod
    def end_run():
        MlFlowUtils.log_param("train_end_date", datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
        mlflow.end_run()
