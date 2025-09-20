import logging
import os
import uuid
from datetime import datetime
from pandas import DataFrame, Series
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator

from b3.service.model.plotter import B3DashboardPlotter
from b3.service.data.storage.data_storage_service import DataStorageService
from b3.service.data.db.evaluation.data_loader import EvaluationDataLoader


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
        self.storage_service = storage_service or DataStorageService()
        self.evaluation_data_loader = evaluation_data_loader or EvaluationDataLoader()
        self.plotter = B3DashboardPlotter(storage_service)

    def _generate_evaluation_id(self, model_name: str, dataset_type: str) -> str:
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
        report = classification_report(y, y_pred, target_names=['buy', 'sell', 'hold'], output_dict=True)

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
                success = self.evaluation_data_loader.save_evaluation(
                    evaluation_id=evaluation_id,
                    model_name=model_name,
                    dataset_type=dataset_type,
                    evaluation_results=report
                )
                
                if success:
                    logging.info(f"Evaluation results saved to database with ID: {evaluation_id}")
                    report['evaluation_id'] = evaluation_id
                else:
                    logging.warning(f"Failed to save evaluation results to database")
                    
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
        if self.storage_service.is_local_storage():
            os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"action_samples_by_ticker_{timestamp}.png"
        plot_path = os.path.join(save_dir, plot_filename).replace('\\', '/')

        self.plotter.plot_action_samples_by_ticker(df, save_path=plot_path)
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
        
        # Evaluate validation set
        validation_results = self.evaluate_model(
            model, X_val, y_val, "Validation", model_name, persist_results
        )
        
        # Evaluate test set
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

    def get_evaluation_history(self, model_name: str = None) -> DataFrame:
        """
        Retrieve evaluation history from the database.
        
        Args:
            model_name: Optional model name to filter by
            
        Returns:
            DataFrame: Evaluation history data
        """
        if model_name:
            return self.evaluation_data_loader.fetch_by_model(model_name)
        else:
            return self.evaluation_data_loader.fetch_all()

    def get_evaluation_summary(self) -> DataFrame:
        """
        Get summary statistics of all evaluations.
        
        Returns:
            DataFrame: Summary statistics
        """
        return self.evaluation_data_loader.get_evaluation_summary()

    def close(self):
        """Close the evaluation data loader connection."""
        if hasattr(self, 'evaluation_data_loader') and self.evaluation_data_loader:
            self.evaluation_data_loader.close()
            logging.info("Evaluation data loader connection closed")
