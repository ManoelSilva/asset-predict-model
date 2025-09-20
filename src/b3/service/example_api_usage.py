#!/usr/bin/env python3
"""
Example script demonstrating how to use the B3 Training API.
This script shows how to use individual endpoints and the complete training pipeline.
"""

import requests
import json
import time


class B3TrainingAPIClient:
    """Client for interacting with the B3 Training API."""

    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url

    def load_data(self):
        """Load B3 market data."""
        response = requests.post(f"{self.base_url}/api/b3/load-data")
        return response.json()

    def preprocess_data(self):
        """Preprocess loaded data."""
        response = requests.post(f"{self.base_url}/api/b3/preprocess-data")
        return response.json()

    def split_data(self, test_size=0.2, val_size=0.2):
        """Split preprocessed data."""
        data = {"test_size": test_size, "val_size": val_size}
        response = requests.post(f"{self.base_url}/api/b3/split-data", json=data)
        return response.json()

    def train_model(self, n_jobs=5):
        """Train the model."""
        data = {"n_jobs": n_jobs}
        response = requests.post(f"{self.base_url}/api/b3/train-model", json=data)
        return response.json()

    def evaluate_model(self):
        """Evaluate the trained model."""
        response = requests.post(f"{self.base_url}/api/b3/evaluate-model")
        return response.json()

    def save_model(self, model_dir="models", model_name="b3_model.joblib"):
        """Save the trained model."""
        data = {"model_dir": model_dir, "model_name": model_name}
        response = requests.post(f"{self.base_url}/api/b3/save-model", json=data)
        return response.json()

    def predict(self, ticker):
        """Make predictions using the trained model for a specific ticker."""
        data = {"ticker": ticker}
        response = requests.post(f"{self.base_url}/api/b3/predict", json=data)
        return response.json()

    def train_complete(self, model_dir="models", n_jobs=5, test_size=0.2, val_size=0.8):
        """Run the complete training pipeline."""
        data = {
            "model_dir": model_dir,
            "n_jobs": n_jobs,
            "test_size": test_size,
            "val_size": val_size
        }
        response = requests.post(f"{self.base_url}/api/b3/train-complete", json=data)
        return response.json()

    def get_status(self):
        """Get current pipeline status."""
        response = requests.get(f"{self.base_url}/api/b3/status")
        return response.json()

    def clear_state(self):
        """Clear the pipeline state."""
        response = requests.post(f"{self.base_url}/api/b3/clear-state")
        return response.json()


def main():
    """Main function demonstrating API usage."""
    client = B3TrainingAPIClient()

    print("=== B3 Training API Usage Example ===\n")

    # Check if API is running
    try:
        status = client.get_status()
        print("✓ API is running")
        print(f"Status: {json.dumps(status, indent=2)}\n")
    except requests.exceptions.ConnectionError:
        print("✗ API is not running. Please start the API first:")
        print("python src/b3/service/b3_model_api.py")
        return

    # Example 1: Complete training pipeline
    print("=== Example 1: Complete Training Pipeline ===")
    try:
        result = client.train_complete(n_jobs=2, test_size=0.2, val_size=0.2)
        print("✓ Complete training pipeline executed successfully")
        print(f"Model saved at: {result.get('model_path', 'N/A')}")
        print(f"Data info: {json.dumps(result.get('data_info', {}), indent=2)}")
    except Exception as e:
        print(f"✗ Error in complete training: {e}")

    print("\n" + "=" * 50 + "\n")

    # Example 2: Step-by-step training
    print("=== Example 2: Step-by-Step Training ===")

    # Clear state first
    client.clear_state()
    print("✓ Pipeline state cleared")

    try:
        # Step 1: Load data
        print("Step 1: Loading data...")
        result = client.load_data()
        if result['status'] == 'success':
            print(f"✓ Data loaded: {result['data_shape']} records")
        else:
            print(f"✗ Failed to load data: {result['message']}")
            return

        # Step 2: Preprocess data
        print("Step 2: Preprocessing data...")
        result = client.preprocess_data()
        if result['status'] == 'success':
            print(f"✓ Data preprocessed: {result['features_shape']} features")
            print(f"Target distribution: {result['target_distribution']}")
        else:
            print(f"✗ Failed to preprocess data: {result['message']}")
            return

        # Step 3: Split data
        print("Step 3: Splitting data...")
        result = client.split_data(test_size=0.2, val_size=0.2)
        if result['status'] == 'success':
            print(f"✓ Data split - Train: {result['train_size']}, "
                  f"Validation: {result['validation_size']}, "
                  f"Test: {result['test_size']}")
        else:
            print(f"✗ Failed to split data: {result['message']}")
            return

        # Step 4: Train model
        print("Step 4: Training model...")
        result = client.train_model(n_jobs=2)
        if result['status'] == 'success':
            print(f"✓ Model trained: {result['model_type']}")
        else:
            print(f"✗ Failed to train model: {result['message']}")
            return

        # Step 5: Evaluate model
        print("Step 5: Evaluating model...")
        result = client.evaluate_model()
        if result['status'] == 'success':
            print("✓ Model evaluated successfully")
            print("Evaluation results available in response")
        else:
            print(f"✗ Failed to evaluate model: {result['message']}")
            return

        # Step 6: Save model
        print("Step 6: Saving model...")
        result = client.save_model(model_dir="models", model_name="b3_model_step_by_step.joblib")
        if result['status'] == 'success':
            print(f"✓ Model saved: {result['model_path']}")
        else:
            print(f"✗ Failed to save model: {result['message']}")
            return

        print("\n✓ Step-by-step training completed successfully!")

    except Exception as e:
        print(f"✗ Error in step-by-step training: {e}")

    print("\n" + "=" * 50 + "\n")

    # Example 3: Make predictions
    print("=== Example 3: Making Predictions ===")
    try:
        # Example ticker for prediction
        example_ticker = "PETR4"

        print(f"Making prediction for ticker: {example_ticker}")
        result = client.predict(example_ticker)

        if result['status'] == 'success':
            print("✓ Prediction successful!")
            print(f"Ticker: {result['ticker']}")
            print(f"Predictions: {result['predictions']}")
            if result['prediction_probabilities']:
                print(f"Prediction probabilities: {result['prediction_probabilities']}")
            if result['feature_importance']:
                print(f"Feature importance: {result['feature_importance']}")
            print(f"Features used: {result['features_used']}")
            print(f"Model type: {result['model_type']}")
            print(f"Model source: {result['model_source']}")
        else:
            print(f"✗ Failed to make prediction: {result['message']}")
            if 'required_features' in result:
                print(f"Required features: {result['required_features']}")
            if 'available_features' in result:
                print(f"Available features: {result['available_features']}")

    except Exception as e:
        print(f"✗ Error making prediction: {e}")

    print("\n" + "=" * 50 + "\n")

    # Example 4: Check final status
    print("=== Example 4: Final Status ===")
    try:
        status = client.get_status()
        print("Final pipeline status:")
        print(json.dumps(status, indent=2))
    except Exception as e:
        print(f"✗ Error getting status: {e}")


if __name__ == "__main__":
    main()
