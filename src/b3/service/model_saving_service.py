import logging
import os
import joblib
from sklearn.ensemble import RandomForestClassifier

# Constants
DEFAULT_MODEL_NAME = "b3_model.joblib"


class B3ModelSavingService:
    """
    Service responsible for saving and loading trained B3 models.
    """

    @staticmethod
    def save_model(model: RandomForestClassifier, model_dir: str = "models",
                   model_name: str = DEFAULT_MODEL_NAME) -> str:
        """
        Save a trained model to disk.
        
        Args:
            model: Trained model to save
            model_dir: Directory to save the model
            model_name: Name of the model file
            
        Returns:
            str: Path to the saved model file
        """
        logging.info(f"Saving model to {model_dir}...")

        # Prepare model directory
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, model_name)

        # Save model
        joblib.dump(model, model_path)
        logging.info(f"Model saved: {model_path}")

        return model_path

    @staticmethod
    def load_model(model_path: str) -> RandomForestClassifier:
        """
        Load a saved model from disk.
        
        Args:
            model_path: Path to the saved model file
            
        Returns:
            BaseEstimator: Loaded model
        """
        logging.info(f"Loading model from {model_path}...")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = joblib.load(model_path)
        logging.info("Model loaded successfully")

        return model

    @staticmethod
    def model_exists(model_dir: str = "models", model_name: str = DEFAULT_MODEL_NAME) -> bool:
        """
        Check if a model file exists.
        
        Args:
            model_dir: Directory where the model should be
            model_name: Name of the model file
            
        Returns:
            bool: True if model exists, False otherwise
        """
        model_path = os.path.join(model_dir, model_name)
        return os.path.exists(model_path)

    @staticmethod
    def get_model_path(model_dir: str = "models", model_name: str = DEFAULT_MODEL_NAME) -> str:
        """
        Get the full path to a model file.
        
        Args:
            model_dir: Directory where the model is located
            model_name: Name of the model file
            
        Returns:
            str: Full path to the model file
        """
        return os.path.join(model_dir, model_name)
