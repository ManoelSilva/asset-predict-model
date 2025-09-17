# B3 Model Training Services

This directory contains the refactored B3 model training system, now organized into separate service classes that can be used independently or through a Flask API.

## Architecture Overview

The training pipeline has been broken down into the following service classes:

### 1. B3DataLoadingService
- **Purpose**: Handles loading B3 market data
- **Key Methods**:
  - `load_data()`: Loads all B3 market data
  - `get_loader()`: Returns the underlying data loader instance

### 2. B3DataPreprocessingService
- **Purpose**: Handles data preprocessing, feature validation, and target label generation
- **Key Methods**:
  - `validate_features(df)`: Validates required features are present
  - `clean_data(X, df)`: Cleans data by removing infinite values and NaN
  - `generate_buy_signal_cluster(df)`: Creates buy signal clusters
  - `generate_sell_signal_cluster(df)`: Creates sell signal clusters
  - `get_target_labels(df)`: Creates target labels from signal clusters
  - `preprocess_data(df)`: Complete preprocessing pipeline

### 3. B3ModelTrainingService
- **Purpose**: Handles model training and data splitting
- **Key Methods**:
  - `split_data(X, y, test_size, val_size)`: Splits data into train/validation/test sets
  - `tune_hyperparameters(X_train, y_train, n_jobs)`: Tunes model hyperparameters
  - `train_model(X_train, y_train, n_jobs)`: Complete model training pipeline

### 4. B3ModelEvaluationService
- **Purpose**: Handles model evaluation and visualization
- **Key Methods**:
  - `evaluate_model(model, X, y, set_name)`: Evaluates model on a dataset
  - `generate_evaluation_visualization(df, save_dir)`: Creates evaluation plots
  - `evaluate_model_comprehensive(model, X_val, y_val, X_test, y_test, df)`: Complete evaluation

### 5. B3ModelSavingService
- **Purpose**: Handles model persistence
- **Key Methods**:
  - `save_model(model, model_dir, model_name)`: Saves model to disk
  - `load_model(model_path)`: Loads model from disk
  - `model_exists(model_dir, model_name)`: Checks if model exists
  - `get_model_path(model_dir, model_name)`: Gets model file path

## Flask API

The `B3TrainingAPI` class provides a Flask-based REST API that exposes all training stages as individual endpoints.

### API Endpoints

#### Individual Stage Endpoints

1. **POST /api/b3/load-data**
   - Loads B3 market data
   - Returns: Data shape and column information

2. **POST /api/b3/preprocess-data**
   - Preprocesses loaded data
   - Returns: Feature shapes and target distribution

3. **POST /api/b3/split-data**
   - Splits data into train/validation/test sets
   - Parameters: `test_size`, `val_size`
   - Returns: Split sizes

4. **POST /api/b3/train-model**
   - Trains the model with hyperparameter tuning
   - Parameters: `n_jobs`
   - Returns: Model type and best parameters

5. **POST /api/b3/evaluate-model**
   - Evaluates the trained model
   - Returns: Evaluation results and visualization path

6. **POST /api/b3/save-model**
   - Saves the trained model
   - Parameters: `model_dir`, `model_name`
   - Returns: Model file path

#### Complete Pipeline Endpoint

7. **POST /api/b3/train-complete**
   - Runs the complete training pipeline in one call
   - Parameters: `model_dir`, `n_jobs`, `test_size`, `val_size`
   - Returns: Complete results including model path and evaluation

#### Utility Endpoints

8. **GET /api/b3/status**
   - Returns current pipeline state

9. **POST /api/b3/clear-state**
   - Clears the pipeline state

## Usage Examples

### Using the Refactored B3Model Class

```python
from b3.service.model import B3Model

# Initialize model (now uses service classes internally)
model = B3Model()

# Train using the new modular approach
model.train(model_dir="models", n_jobs=5, test_size=0.2, val_size=0.2)

# Make predictions
predictions = model.predict(new_data, model_dir="models")
```

### Using Individual Services

```python
from b3.service.data_loading_service import B3DataLoadingService
from b3.service.data_preprocessing_service import B3DataPreprocessingService
from b3.service.model_training_service import B3ModelTrainingService

# Load data
data_service = B3DataLoadingService()
df = data_service.load_data()

# Preprocess data
preprocessing_service = B3DataPreprocessingService()
X, df_processed, y = preprocessing_service.preprocess_data(df)

# Train model
training_service = B3ModelTrainingService()
X_train, X_val, X_test, y_train, y_val, y_test = training_service.split_data(X, y)
model = training_service.train_model(X_train, y_train)
```

### Using the Flask API

```python
import requests

# Start the API server
# python src/b3/service/b3_training_api.py

# Use individual endpoints
response = requests.post("http://localhost:5000/api/b3/load-data")
response = requests.post("http://localhost:5000/api/b3/preprocess-data")
# ... continue with other steps

# Or use the complete pipeline
response = requests.post("http://localhost:5000/api/b3/train-complete", 
                        json={"n_jobs": 5, "test_size": 0.2, "val_size": 0.2})
```

## Running the API

To start the Flask API server:

```bash
cd src/b3/service
python b3_training_api.py
```

The API will be available at `http://localhost:5000`

## Example Client

See `example_api_usage.py` for a complete example of how to use the API client to interact with all endpoints.

## Benefits of the New Architecture

1. **Modularity**: Each training stage is now a separate, reusable service
2. **Flexibility**: Services can be used independently or in combination
3. **API Access**: Training stages can be accessed via REST API
4. **Testability**: Individual services can be tested in isolation
5. **Scalability**: Services can be deployed separately if needed
6. **Maintainability**: Clear separation of concerns makes code easier to maintain

## Migration Notes

- The `B3Model.train_and_save()` method has been renamed to `train()`
- The `B3Model` class now uses the new service classes internally
- All original functionality is preserved but now uses the modular architecture
- The `predict()` method now uses the `B3ModelSavingService` for loading models
