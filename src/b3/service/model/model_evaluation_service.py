import logging
import os
from datetime import datetime
from pandas import DataFrame, Series
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator

from b3.service.model.plotter import B3DashboardPlotter
from b3.service.data.storage.data_storage_service import DataStorageService


class B3ModelEvaluationService:
    """
    Service responsible for evaluating trained B3 models and generating visualizations.
    """

    def __init__(self, storage_service: DataStorageService = None):
        """
        Initialize the evaluation service.
        
        Args:
            storage_service: Data storage service instance (optional, creates default if not provided)
        """
        self.storage_service = storage_service or DataStorageService()
        self.plotter = B3DashboardPlotter(storage_service)

    @staticmethod
    def evaluate_model(model: BaseEstimator, X: DataFrame, y: Series, set_name: str = "Dataset") -> dict:
        """
        Evaluate the classifier and return classification metrics.
        
        Args:
            model: Trained classifier
            X: Feature data
            y: Target labels
            set_name: Name of the dataset being evaluated (e.g., "Train", "Validation", "Test")
            
        Returns:
            dict: Classification report as dictionary
        """
        logging.info(f"Evaluating model on {set_name} set...")

        y_pred = model.predict(X)
        report = classification_report(y, y_pred, target_names=['buy', 'sell', 'hold'], output_dict=True)

        print(f"\n{classification_report(y, y_pred, target_names=['buy', 'sell', 'hold'])}")
        logging.info(f"Model evaluation completed on {set_name} set")

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
        if self.storage_service.is_local_storage():
            os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"action_samples_by_ticker_{timestamp}.png"
        plot_path = os.path.join(save_dir, plot_filename).replace('\\', '/')

        self.plotter.plot_action_samples_by_ticker(df, save_path=plot_path)
        logging.info(f"Evaluation visualization saved: {plot_path}")

        return plot_path

    def evaluate_model_comprehensive(self, model: BaseEstimator, X_val: DataFrame, y_val: Series,
                                     X_test: DataFrame, y_test: Series, df: DataFrame) -> dict:
        """
        Perform comprehensive model evaluation on validation and test sets.
        
        Args:
            model: Trained classifier
            X_val: Validation features
            y_val: Validation targets
            X_test: Test features
            y_test: Test targets
            df: Full dataset for visualization
            
        Returns:
            dict: Complete evaluation results
        """
        return {'validation': self.evaluate_model(model, X_val, y_val, "Validation"),
                'test': self.evaluate_model(model, X_test, y_test, "Test"),
                'visualization_path': self.generate_evaluation_visualization(df)}
