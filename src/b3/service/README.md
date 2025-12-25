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

## Execution Modes

The API supports two execution modes for model training:

### 1. Complete Training Pipeline (End-to-End)

Execute the entire training pipeline in a single API call. This is the recommended mode for production deployments.

**Endpoint**: `POST /api/b3/complete-pipeline`

**Supported Models**: Both `rf` (Random Forest) and `lstm` (LSTM Multi-Task Learning)

**Request Body for Random Forest**:
```json
{
  "model_type": "rf",
  "model_dir": "models",
  "n_jobs": 5,
  "test_size": 0.2,
  "val_size": 0.2
}
```

**Request Body for LSTM**:
```json
{
  "model_type": "lstm",
  "model_dir": "models",
  "lookback": 32,
  "horizon": 1,
  "epochs": 25,
  "batch_size": 128,
  "learning_rate": 0.001,
  "units": 96,
  "dropout": 0.2,
  "test_size": 0.2,
  "val_size": 0.2,
  "price_col": "close"
}
```

**Response**: 
- Returns immediately with status "running"
- Training executes in background
- Check status via `GET /api/b3/training-status`
- Results available in pipeline state when complete

**Pipeline Steps Executed**:
1. Load data from data source
2. Preprocess data (feature engineering, label generation)
3. Split data (train/validation/test)
4. Train model (with hyperparameter tuning for RF)
5. Evaluate model (classification and regression metrics)
6. Save model to storage

### 2. Independent Endpoints (Step-by-Step)

Execute each pipeline step independently for fine-grained control.

#### Step 1: Load Data
**Endpoint**: `POST /api/b3/load-data`

- **Purpose**: Load B3 market data from configured data source
- **Parameters**: None (uses default data source)
- **Returns**: 
  ```json
  {
    "status": "success",
    "message": "Data loaded successfully",
    "data_shape": [rows, columns],
    "columns": ["col1", "col2", ...]
  }
  ```
- **State**: Stores raw data in pipeline state

#### Step 2: Preprocess Data
**Endpoint**: `POST /api/b3/preprocess-data`

- **Purpose**: Feature engineering, validation, and target label generation
- **Requires**: Data must be loaded (Step 1)
- **Parameters**: None (uses preprocessing service defaults)
- **Returns**:
  ```json
  {
    "status": "success",
    "message": "Data preprocessed successfully",
    "features_shape": [rows, features],
    "targets_shape": [rows],
    "feature_columns": ["feature1", ...],
    "target_distribution": {"buy": count, "sell": count, ...}
  }
  ```
- **State**: Stores preprocessed features and targets

#### Step 3: Split Data
**Endpoint**: `POST /api/b3/split-data`

- **Purpose**: Split data into train/validation/test sets
- **Requires**: Preprocessed data (Step 2)
- **Parameters**:
  ```json
  {
    "model_type": "rf",  // or "lstm"
    "test_size": 0.2,
    "val_size": 0.2
  }
  ```
- **Returns**:
  ```json
  {
    "status": "success",
    "message": "Data split successfully",
    "train_size": 1000,
    "validation_size": 250,
    "test_size": 250
  }
  ```
- **Note**: Model type determines split strategy (RF uses standard split, LSTM handles sequences)
- **State**: Stores train/val/test splits

#### Step 4: Train Model
**Endpoint**: `POST /api/b3/train-model`

- **Purpose**: Train the selected model
- **Requires**: Split data (Step 3)
- **Parameters for Random Forest**:
  ```json
  {
    "model_type": "rf",
    "n_jobs": 5
  }
  ```
- **Parameters for LSTM**:
  ```json
  {
    "model_type": "lstm",
    "lookback": 32,
    "horizon": 1,
    "epochs": 25,
    "batch_size": 128,
    "learning_rate": 0.001,
    "units": 96,
    "dropout": 0.2
  }
  ```
- **Returns**: 
  ```json
  {
    "status": "success",
    "message": "Model training started in background",
    "training_status": "running",
    "model_type": "rf"
  }
  ```
- **Note**: Training executes in background; check status via `GET /api/b3/training-status`
- **State**: Stores trained model when complete

#### Step 5: Evaluate Model
**Endpoint**: `POST /api/b3/evaluate-model`

- **Purpose**: Evaluate trained model on validation and test sets
- **Requires**: Trained model (Step 4)
- **Parameters**: None (uses model and data from state)
- **Returns**:
  ```json
  {
    "status": "success",
    "evaluation": {
      "validation": {...},
      "test": {...}
    },
    "visualization_path": "path/to/plot.png"
  }
  ```
- **Metrics**: 
  - Random Forest: Classification metrics (accuracy, precision, recall, F1)
  - LSTM: Classification + Regression metrics (MAE, MSE, RMSE, R²)

#### Step 6: Save Model
**Endpoint**: `POST /api/b3/save-model`

- **Purpose**: Persist trained model to storage
- **Requires**: Trained model (Step 4)
- **Parameters**:
  ```json
  {
    "model_dir": "models",
    "model_name": "b3_model"  // optional
  }
  ```
- **Returns**:
  ```json
  {
    "status": "success",
    "model_path": "models/b3_model.joblib"  // or .pt for LSTM
  }
  ```

#### Utility Endpoints

**GET /api/b3/pipeline-status**
- Returns current pipeline state and intermediate results
- Useful for debugging and monitoring

**GET /api/b3/training-status**
- Returns training job status (running, completed, failed)
- Includes error messages if training failed

**POST /api/b3/clear-state**
- Clears all pipeline state
- Useful for starting fresh or debugging

## Flask API

The `B3ModelAPI` class provides a Flask-based REST API that exposes all training stages as individual endpoints or as a complete pipeline.

## Usage Examples

### Using the Refactored B3Model Class

```python
from b3.service.pipeline import B3Model

# Initialize pipeline (now uses service classes internally)
model = B3Model()

# Train using the new modular approach
model.run(model_dir="models", n_jobs=5, test_size=0.2, val_size=0.2)

# Make predictions
predictions = model.predict(new_data, model_dir="models")
```

### Using Individual Services

```python
from b3.service.data.db.b3_featured.data_loading_service import DataLoadingService
from b3.service.pipeline.model_preprocessing_service import PreprocessingService
from b3.service.pipeline.training.model_training_service import B3ModelTrainingService

# Load data
data_service = DataLoadingService()
df = data_service.load_data()

# Preprocess data
preprocessing_service = PreprocessingService()
X, df_processed, y = preprocessing_service.preprocess_data(df)

# Train pipeline
training_service = B3ModelTrainingService()
X_train, X_val, X_test, y_train, y_val, y_test = training_service.split_data(X, y)
model = training_service.train_model(X_train, y_train)
```

### Using the Flask API

#### Complete Pipeline Mode

```python
import requests

# Start the API server
# python src/web_api.py

# Complete pipeline for Random Forest
response = requests.post(
    "http://localhost:5000/api/b3/complete-pipeline",
    json={
        "model_type": "rf",
        "model_dir": "models",
        "n_jobs": 5,
        "test_size": 0.2,
        "val_size": 0.2
    }
)

# Complete pipeline for LSTM
response = requests.post(
    "http://localhost:5000/api/b3/complete-pipeline",
    json={
        "model_type": "lstm",
        "model_dir": "models",
        "lookback": 32,
        "horizon": 1,
        "epochs": 25,
        "batch_size": 128,
        "learning_rate": 0.001,
        "units": 96,
        "dropout": 0.2,
        "test_size": 0.2,
        "val_size": 0.2
    }
)

# Check training status
status = requests.get("http://localhost:5000/api/b3/training-status")
print(status.json())
```

#### Step-by-Step Mode

```python
import requests
import time

base_url = "http://localhost:5000/api/b3"

# Step 1: Load data
response = requests.post(f"{base_url}/load-data")
print(response.json())

# Step 2: Preprocess
response = requests.post(f"{base_url}/preprocess-data")
print(response.json())

# Step 3: Split data (specify model type)
response = requests.post(
    f"{base_url}/split-data",
    json={"model_type": "rf", "test_size": 0.2, "val_size": 0.2}
)
print(response.json())

# Step 4: Train model
response = requests.post(
    f"{base_url}/train-model",
    json={"model_type": "rf", "n_jobs": 5}
)
print(response.json())

# Wait for training to complete
while True:
    status = requests.get(f"{base_url}/training-status")
    status_data = status.json()
    if status_data.get("status") == "completed":
        break
    elif status_data.get("status") == "failed":
        print(f"Training failed: {status_data.get('error')}")
        break
    time.sleep(2)

# Step 5: Evaluate
response = requests.post(f"{base_url}/evaluate-model")
print(response.json())

# Step 6: Save
response = requests.post(
    f"{base_url}/save-model",
    json={"model_dir": "models"}
)
print(response.json())
```

## Running the API

To start the Flask API server:

```bash
cd src/b3_featured/service
python b3_model_api.py
```

The API will be available at `http://localhost:5000`

## Example Client


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

## Cross-References

- For CLI usage and overall project setup, see the main [README.md](../../../../README.md).
- For API endpoint details and OpenAPI/Swagger documentation, see [swagger.yml](../../config/web_api/swagger/swagger.yml) and the main README.

## Additional Notes
- All service classes can be used independently or via the REST API.
- The API server entry point is `src/b3/service/web_api/b3_model_api.py`.
- Endpoints and handler classes are defined in `src/b3/service/web_api/handler/`.




