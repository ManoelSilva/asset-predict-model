[Leia em português](README.pt-br.md)

# Asset Predict Model

This project provides tools and models for asset price prediction, including data loading, feature engineering, model training, and prediction for financial assets such as B3.

## Features
- Data loading and preprocessing for B3 asset type
- Feature engineering and label creation
- Model training (Random Forest, etc.)
- Prediction and plotting utilities

## Project Structure
- `src/` - Main source code
  - `b3/` - B3 asset tools
  - `models/` - Pretrained models
- `requirements.txt` - Python dependencies

## Getting Started
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Explore the code:**
   - Explore `src/b3/` for asset-specific modules.

## Requirements
- Python 3.12+
- See `requirements.txt` for all dependencies

## Architecture Overview

The project is built around a modular service architecture for asset prediction, with each stage of the pipeline implemented as a reusable service class. These can be used directly in Python or accessed via a REST API.

### Service Classes
- **B3DataLoadingService**: Loads B3 market data.
- **B3ModelPreprocessingService**: Handles data preprocessing, feature validation, and target label generation.
- **B3ModelTrainingService**: Handles model training and data splitting.
- **B3ModelEvaluationService**: Handles model evaluation and visualization.
- **B3ModelSavingService**: Handles model persistence (saving/loading models).

## Supported Models

The project supports two machine learning models for asset price prediction:

### 1. Random Forest (rf)
- **Type**: Classical Machine Learning (Ensemble Method)
- **Algorithm**: Random Forest Classifier (scikit-learn)
- **Use Case**: Classification of asset price direction (Buy/Sell/Hold)
- **Strengths**: 
  - Fast training and inference
  - Interpretable feature importance
  - Works well with tabular data
  - No sequence requirements
- **Output**: Action predictions (classification)
- **Storage Format**: Joblib serialized models (`.joblib`)
- **Best For**: Single-point predictions, quick iterations, interpretability

### 2. LSTM Multi-Task Learning (lstm/lstm_mtl)
- **Type**: Deep Learning (Recurrent Neural Network)
- **Framework**: PyTorch
- **Architecture**: Long Short-Term Memory (LSTM) with Multi-Task Learning
- **Use Case**: 
  - Action prediction (classification)
  - Return forecasting (regression)
- **Strengths**:
  - Captures temporal patterns in time series
  - Multi-task learning improves generalization
  - Can predict both actions and price returns
- **Requirements**: 
  - Historical sequences (lookback window)
  - Sequential data ordered by time
- **Output**: 
  - Action predictions (classification)
  - Return predictions (regression)
  - Price predictions (derived from returns)
- **Storage Format**: PyTorch state dictionaries (`.pt`)
- **Best For**: Time series analysis, capturing temporal dependencies, multi-task predictions

## Execution Modes

The project supports two execution modes for model training:

### 1. Complete Training Pipeline (End-to-End)

Execute the entire training pipeline in a single API call. This mode handles all steps automatically:
- Data loading
- Data preprocessing
- Data splitting
- Model training
- Model evaluation
- Model saving

**Endpoint**: `POST /api/b3/complete-pipeline`

**Example Request**:
```json
{
  "model_type": "rf",
  "model_dir": "models",
  "n_jobs": 5,
  "test_size": 0.2,
  "val_size": 0.2
}
```

**For LSTM**:
```json
{
  "model_type": "lstm",
  "model_dir": "models",
  "lookback": 32,
  "horizon": 2,
  "epochs": 50,
  "batch_size": 256,
  "learning_rate": 0.001,
  "units": 64,
  "dropout": 0.4,
  "loss_weight_action": 20.0,
  "loss_weight_return": 50.0,
  "early_stopping_patience": 20,
  "gradient_clip_norm": 1.0,
  "weight_decay": 1e-4,
  "focal_loss_gamma": 3.0,
  "test_size": 0.2,
  "val_size": 0.2
}
```

**Benefits**:
- Simple single-call execution
- Automatic state management
- Best for production deployments
- Handles all pipeline steps sequentially

### 2. Independent Endpoints (Step-by-Step)

Execute each pipeline step independently. This provides fine-grained control over each stage:

**Step 1: Load Data**
- **Endpoint**: `POST /api/b3/load-data`
- **Purpose**: Load B3 market data from data source
- **Returns**: Data shape and column information

**Step 2: Preprocess Data**
- **Endpoint**: `POST /api/b3/preprocess-data`
- **Purpose**: Feature engineering, validation, and target label generation
- **Requires**: Data must be loaded first
- **Returns**: Feature shapes and target distribution

**Step 3: Split Data**
- **Endpoint**: `POST /api/b3/split-data`
- **Purpose**: Split data into train/validation/test sets
- **Requires**: Preprocessed data
- **Parameters**: `model_type`, `test_size`, `val_size`
- **Returns**: Split sizes for each set

**Step 4: Train Model**
- **Endpoint**: `POST /api/b3/train-model`
- **Purpose**: Train the selected model (rf or lstm)
- **Requires**: Split data
- **Parameters**: `model_type`, `n_jobs` (for RF), LSTM-specific params
- **Returns**: Training status and model information

**Step 5: Evaluate Model**
- **Endpoint**: `POST /api/b3/evaluate-model`
- **Purpose**: Evaluate trained model on validation and test sets
- **Requires**: Trained model
- **Returns**: Evaluation metrics and visualization paths

**Step 6: Save Model**
- **Endpoint**: `POST /api/b3/save-model`
- **Purpose**: Persist trained model to storage
- **Requires**: Trained model
- **Parameters**: `model_dir`, `model_name`
- **Returns**: Model file path

**Benefits**:
- Fine-grained control over each step
- Ability to inspect intermediate results
- Useful for debugging and experimentation
- Can modify data between steps
- Supports custom workflows

**Example Workflow**:
```python
import requests

base_url = "http://localhost:5000/api/b3"

# Step 1: Load data
requests.post(f"{base_url}/load-data")

# Step 2: Preprocess
requests.post(f"{base_url}/preprocess-data")

# Step 3: Split (specify model type)
requests.post(f"{base_url}/split-data", json={"model_type": "rf", "test_size": 0.2, "val_size": 0.2})

# Step 4: Train
requests.post(f"{base_url}/train-model", json={"model_type": "rf", "n_jobs": 5})

# Step 5: Evaluate
requests.post(f"{base_url}/evaluate-model")

# Step 6: Save
requests.post(f"{base_url}/save-model", json={"model_dir": "models"})
```

## REST API (Flask)

A Flask-based REST API exposes all training and prediction stages as endpoints. The API is documented with OpenAPI/Swagger (see `/swagger` when running the server).

### Main Endpoints

#### Complete Pipeline
- `POST /api/b3/complete-pipeline`: Run the complete training pipeline (end-to-end)

#### Individual Steps
- `POST /api/b3/load-data`: Load B3 market data
- `POST /api/b3/preprocess-data`: Preprocess loaded data
- `POST /api/b3/split-data`: Split data into train/validation/test sets
- `POST /api/b3/train-model`: Train the model (rf or lstm)
- `POST /api/b3/evaluate-model`: Evaluate the trained model
- `POST /api/b3/save-model`: Save the trained model

#### Prediction & Status
- `POST /api/b3/predict`: Make predictions for a specific ticker (supports rf and lstm)
- `GET /api/b3/pipeline-status`: Get current pipeline status
- `POST /api/b3/clear-state`: Clear the pipeline state

### OpenAPI/Swagger
- The API is documented with Swagger UI, available at `/swagger` when the server is running.
- The OpenAPI spec is at `/swagger/swagger.yml`.

## Usage Examples

### Using the Python API

```python
from b3.service.pipeline import B3Model

model = B3Model()
model.train(model_dir="models", n_jobs=5, test_size=0.2, val_size=0.2)
predictions = model.predict(new_data, model_dir="models")
```

### Using Individual Services

```python
from b3.service.data.db.b3_featured.data_loading_service import DataLoadingService
from b3.service.pipeline.model_preprocessing_service import PreprocessingService
from b3.service.pipeline.training.model_training_service import B3ModelTrainingService

data_service = DataLoadingService()
df = data_service.load_data()

preprocessing_service = PreprocessingService()
X, df_processed, y = preprocessing_service.preprocess_data(df)

training_service = B3ModelTrainingService()
X_train, X_val, X_test, y_train, y_val, y_test = training_service.split_data(X, y)
model = training_service.train_model(X_train, y_train)
```

### Using the REST API
Start the API server:
```bash
python src/b3/service/web_api/b3_model_api.py
```

Example request (using `requests`):
```python
import requests
response = requests.post("http://localhost:5000/api/b3/load-data")
```

See the Swagger UI at [http://localhost:5000/swagger](http://localhost:5000/swagger) for full API documentation and to try endpoints interactively.

## Model Performance and Evaluation

### Model Metrics

#### Random Forest
- **Algorithm**: Random Forest Classifier
- **Target**: Asset price direction prediction (Buy/Sell/Hold)
- **Features**: Technical indicators, price movements, volume patterns
- **Evaluation**: Cross-validation with train/validation/test splits
- **Hyperparameter Tuning**: RandomizedSearchCV with stratified K-fold

#### LSTM Multi-Task Learning
- **Architecture**: PyTorch LSTM with multi-task learning
- **Tasks**: 
  - Classification: Action prediction (Buy/Sell/Hold)
  - Regression: Return forecasting
- **Features**: Sequential technical indicators and price data
- **Evaluation**: 
  - Classification metrics (accuracy, precision, recall, F1)
  - Regression metrics (MAE, MSE, RMSE, R²)
- **Training**: Adam optimizer with configurable learning rate

### Training Data Requirements
- **Source**: B3 historical data from asset-data-lake
- **Time Period**: Configurable (default: recent 2 years)
- **Minimum Records**: 1000+ per asset for reliable predictions
- **Update Frequency**: Retrain when new data is available

### Model Evaluation Results

#### LSTM Multi-Task Learning - Final Performance

After systematic optimization and improvements, the LSTM model achieved the following results:

**Classification Performance:**
- **Overall Accuracy**: 91.3% (improved from 59.1% - **+54% improvement**)
- **Macro F1 Score**: 0.411 (improved from 0.28 - **+47% improvement**)
- **Weighted F1 Score**: 0.946 (excellent)
- **Sell Class**: 
  - Precision: 99.7% (excellent)
  - Recall: 91.5% (excellent)
  - F1: 0.954 (improved from 0.74 - **+29% improvement**)
- **Buy Class**: 
  - Precision: 8.0% (improved from 1.0% - **+700% improvement**)
  - Recall: 65.2% (moderate)
  - F1: 0.143
- **Hold Class**: 
  - Precision: 7.5% (improved from 4.0% - **+88% improvement**)
  - Recall: 77.2% (good)
  - F1: 0.136

**Regression Performance:**
- **R² Score**: 0.9982 (99.82% variance explained - excellent)
- **Mean Absolute Error (MAE)**: 1.20 (validation), 1.25 (test)
- **Root Mean Squared Error (RMSE)**: 5.47 (validation), 5.74 (test)
- **Mean Absolute Percentage Error (MAPE)**: 8.98% (validation), 8.49% (test)

**Overfitting Reduction:**
- **Overfitting Gap**: 8,549 (reduced from 172,720 - **95% reduction**)
- **Training Loss**: 253.35 (reduced from 2,839.81 - **91% reduction**)
- **Validation Loss**: 8,802.48 (reduced from 175,560.08 - **95% reduction**)

**Key Improvements Achieved:**
1. ✅ **95% reduction in overfitting** through regularization techniques
2. ✅ **54% improvement in classification accuracy** (59% → 91.3%)
3. ✅ **99.8% R² score** for regression (excellent price prediction)
4. ✅ **700% improvement in Buy precision** (1.0% → 8.0%)
5. ✅ **Stable training** with early stopping and adaptive learning rate

**Technical Optimizations Applied:**
- Early stopping (patience=20) to prevent overfitting
- Gradient clipping (max_norm=1.0) for training stability
- L2 regularization (weight_decay=1e-4) to reduce model complexity
- Dropout regularization (0.4) to improve generalization
- Focal Loss (gamma=3.0) for class imbalance handling
- Loss weight balancing (action:return = 20:50) for multi-task learning
- Learning rate scheduling (ReduceLROnPlateau) for better convergence
- Oversampling (WeightedRandomSampler) for minority classes

#### Random Forest
- **Accuracy**: Varies by asset (typically 55-70%)
- **Precision/Recall**: Balanced for both classes
- **Feature Importance**: Price momentum and volatility indicators most significant
- **Validation**: Time-series cross-validation to prevent data leakage

## Environment Configuration

### Required Environment Variables

```bash
export MOTHERDUCK_TOKEN="your_motherduck_token_here"
export environment="AWS"  # or "LOCAL" for local development
export EC2_HOST="your_ec2_public_ip"  # for production deployment
```

### Local Development Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables**
   ```bash
   export MOTHERDUCK_TOKEN="your_token"
   export environment="LOCAL"
   ```

3. **Run the API server**
   ```bash
   python src/web_api.py
   ```

## Service Deployment

### Production Deployment (AWS EC2)

The service is designed to run as a systemd service on AWS EC2:

```bash
# Deploy using the provided script
sudo MOTHERDUCK_TOKEN=your_token EC2_HOST=your_ip bash deploy_asset_predict_model.sh
```

### Service Management

```bash
# Check service status
sudo systemctl status asset-predict-pipeline

# View logs
sudo journalctl -u asset-predict-pipeline -f

# Restart service
sudo systemctl restart asset-predict-pipeline
```

## Model Versioning and Persistence

### Model Storage
- **Format**: 
  - Random Forest: Joblib serialized models (`.joblib`)
  - LSTM: PyTorch state dictionaries (`.pt`)
- **Location**: `models/` directory
- **Naming**: `b3_model.joblib` or `b3_lstm_mtl.pt`
- **Backup**: Previous versions archived before updates

### Model Updates
- **Trigger**: Manual retraining or scheduled updates
- **Process**: Load new data → Retrain → Validate → Deploy
- **Rollback**: Previous model versions available for quick rollback

## API Integration

### Frontend Integration
The model API integrates with the Angular frontend:
- **Endpoint**: `POST /api/b3/predict`
- **Input**: `{"ticker": "PETR4", "model_type": "rf"}`
- **Output**: Prediction result with probabilities and feature importance

### Data Lake Integration
- **Data Source**: Connects to asset-data-lake for historical data
- **Real-time**: Uses latest available data for predictions
- **Caching**: Model predictions cached for performance

## Performance Considerations

### Model Inference
- **Latency**: < 100ms for single predictions
- **Throughput**: 100+ predictions/second
- **Memory**: ~500MB for loaded model
- **CPU**: Single-threaded inference

### Scalability
- **Horizontal**: Multiple instances behind load balancer
- **Vertical**: t3.large EC2 instance sufficient for moderate load
- **Caching**: Redis recommended for high-traffic scenarios

## Monitoring and Logging

### Logging
- **Level**: INFO for normal operations, ERROR for failures
- **Format**: Structured JSON logs
- **Retention**: 30 days (configurable)

## License
[MIT License](LICENSE)

---
[Leia em português](README.pt-br.md)
