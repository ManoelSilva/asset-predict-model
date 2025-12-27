import logging
import os
import uuid
from datetime import datetime

import numpy as np
from asset_model_data_storage.data_storage_service import DataStorageService
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score, confusion_matrix

from b3.service.data.db.evaluation.data_loader import EvaluationDataLoader
from b3.service.pipeline.plotter import B3DashboardPlotter


class B3ModelEvaluationService:
    """
    Service responsible for evaluating trained B3 models and generating visualizations.
    """

    def __init__(self, storage_service: DataStorageService = None, evaluation_data_loader: EvaluationDataLoader = None):
        """
        Initialize the evaluation service.
        
        Args:
            storage_service: Data storage service instance (optional, creates default if not provided)
            evaluation_data_loader: Evaluation data loader instance (optional, creates default if not provided)
        """
        self._storage_service = storage_service or DataStorageService()
        self._evaluation_data_loader = evaluation_data_loader or EvaluationDataLoader()
        self._plotter = B3DashboardPlotter(storage_service)

    @staticmethod
    def _generate_evaluation_id(model_name: str, dataset_type: str) -> str:
        """
        Generate a unique evaluation ID.
        
        Args:
            model_name: Name of the model being evaluated
            dataset_type: Type of dataset ('validation' or 'test')
            
        Returns:
            str: Unique evaluation ID
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"{model_name}_{dataset_type}_{timestamp}_{unique_id}"

    def evaluate_model(self, model: BaseEstimator, X: DataFrame, y: Series, set_name: str = "Dataset",
                       model_name: str = "b3_model", persist_results: bool = True) -> dict:
        """
        Evaluate the classifier and return classification metrics.
        
        Args:
            model: Trained classifier
            X: Feature data
            y: Target labels
            set_name: Name of the dataset being evaluated (e.g., "Train", "Validation", "Test")
            model_name: Name of the model being evaluated
            persist_results: Whether to persist results to database
            
        Returns:
            dict: Classification report as dictionary
        """
        logging.info(f"Evaluating model on {set_name} set...")

        y_pred = model.predict(X)

        # Calculate Confusion Matrix
        cm = confusion_matrix(y, y_pred, labels=['buy', 'sell', 'hold'])
        logging.info(f"Confusion Matrix ({set_name}):\n{cm}")

        report = classification_report(
            y,
            y_pred,
            target_names=['buy', 'sell', 'hold'],
            output_dict=True,
            zero_division=0
        )

        # Add confusion matrix to report (as list for JSON serialization)
        report['confusion_matrix'] = cm.tolist()

        print(f"\n{classification_report(y, y_pred, target_names=['buy', 'sell', 'hold'])}")
        logging.info(f"Model evaluation completed on {set_name} set")

        # Persist results to database if requested
        if persist_results:
            try:
                # Determine dataset type for database storage
                dataset_type = set_name.lower() if set_name.lower() in ['validation', 'test'] else 'validation'

                # Generate unique evaluation ID
                evaluation_id = self._generate_evaluation_id(model_name, dataset_type)

                # Save evaluation results
                success = self._evaluation_data_loader.save_evaluation(
                    evaluation_id=evaluation_id,
                    model_name=model_name,
                    dataset_type=dataset_type,
                    evaluation_results=report
                )

                if success:
                    logging.info(f"Evaluation results saved to database with ID: {evaluation_id}")
                    report['evaluation_id'] = evaluation_id
                else:
                    logging.warning('Failed to save evaluation results to database')

            except Exception as e:
                logging.error(f"Error persisting evaluation results: {str(e)}")

        return report

    def generate_evaluation_visualization(self, df: DataFrame, save_dir: str = "b3/assets/evaluation") -> str:
        """
        Generate and save evaluation visualization using the configured storage service.
        
        Args:
            df: Full dataset for visualization
            save_dir: Directory to save the plot
            
        Returns:
            str: Path/URL to the saved plot file
        """
        logging.info("Generating evaluation visualization...")

        # Create directory if using local storage
        if self._storage_service.is_local_storage():
            os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"action_samples_by_ticker_{timestamp}.png"
        plot_path = os.path.join(save_dir, plot_filename).replace('\\', '/')

        self._plotter.plot_action_samples_by_ticker(df, save_path=plot_path)
        logging.info(f"Evaluation visualization saved: {plot_path}")

        return plot_path

    def evaluate_model_comprehensive(self, model: BaseEstimator, X_val: DataFrame, y_val: Series,
                                     X_test: DataFrame, y_test: Series, df: DataFrame,
                                     model_name: str = "b3_model", persist_results: bool = True) -> dict:
        """
        Perform comprehensive model evaluation on validation and test sets.
        
        Args:
            model: Trained classifier
            X_val: Validation features
            y_val: Validation targets
            X_test: Test features
            y_test: Test targets
            df: Full dataset for visualization
            model_name: Name of the model being evaluated
            persist_results: Whether to persist results to database
            
        Returns:
            dict: Complete evaluation results
        """
        # Generate visualization first
        visualization_path = self.generate_evaluation_visualization(df)

        # Evaluate validation set (classification)
        validation_results = self.evaluate_model(
            model, X_val, y_val, "Validation", model_name, persist_results
        )

        # Evaluate test set (classification)
        test_results = self.evaluate_model(
            model, X_test, y_test, "Test", model_name, persist_results
        )

        # If persisting results, update the database records with visualization path
        if persist_results:
            try:
                # Update validation record with visualization path
                if 'evaluation_id' in validation_results:
                    val_evaluation_id = validation_results['evaluation_id']
                    # Note: We would need to add an update method to EvaluationDataLoader
                    # For now, we'll log this information
                    logging.info(f"Validation evaluation ID: {val_evaluation_id}, Visualization: {visualization_path}")

                # Update test record with visualization path
                if 'evaluation_id' in test_results:
                    test_evaluation_id = test_results['evaluation_id']
                    logging.info(f"Test evaluation ID: {test_evaluation_id}, Visualization: {visualization_path}")

            except Exception as e:
                logging.error(f"Error updating evaluation records with visualization path: {str(e)}")

        return {
            'validation': validation_results,
            'test': test_results,
            'visualization_path': visualization_path
        }

    @staticmethod
    def evaluate_regression(y_true_prices: Series | np.ndarray, y_pred_prices: Series | np.ndarray) -> dict:
        """
        Evaluate regression predictions with MAE, RMSE, and MAPE.

        Args:
            y_true_prices: Ground-truth prices
            y_pred_prices: Predicted prices

        Returns:
            dict: { mae, rmse, mape }
        """
        y_true = np.asarray(y_true_prices, dtype=float)
        y_pred = np.asarray(y_pred_prices, dtype=float)
        mae = mean_absolute_error(y_true, y_pred)
        # Compute RMSE without relying on the 'squared' parameter for wider compatibility
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        # Avoid divide-by-zero in MAPE
        eps = 1e-8
        mape = np.mean(np.abs((y_true - y_pred) / (np.maximum(np.abs(y_true), eps)))) * 100.0

        r2 = r2_score(y_true, y_pred)

        logging.info(f"Regression metrics -> MAE: {mae:.6f}, RMSE: {rmse:.6f}, MAPE: {mape:.4f}%, R2: {r2:.4f}")
        return {"mae": float(mae), "rmse": float(rmse), "mape": float(mape), "r2": float(r2)}

    def update_evaluation_metrics(self, evaluation_id: str, additional_metrics: dict) -> bool:
        """
        Update evaluation metrics in database.
        
        Args:
            evaluation_id: ID of the evaluation to update
            additional_metrics: Dictionary of metrics to add
            
        Returns:
            bool: Success status
        """
        return self._evaluation_data_loader.update_metrics_json(evaluation_id, additional_metrics)
