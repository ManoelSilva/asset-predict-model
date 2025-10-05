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
  - `app.py` - Main application entry point
- `requirements.txt` - Python dependencies

## Getting Started
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Explore the code:**
   - Check `src/app.py` for the main application logic.
   - Explore `src/b3/` for asset-specific modules.

## Requirements
- Python 3.12+
- See `requirements.txt` for all dependencies

## Usage

You can use the application from the command line (CLI) for both training and prediction:

### Train the B3 Model

Train the model and specify the number of parallel jobs (CPU cores) to use for training:

```bash
python src/app.py train --n_jobs 8
```
- `train`: Run the training process.
- `--n_jobs 8`: (Optional) Number of parallel jobs for model training. Default is 5 if not specified.

### Predict Using the B3 Model

Make predictions for a specific ticker:

```bash
python src/app.py predict --ticker BTCI11
```
- `predict`: Run the prediction process.
- `--ticker BTCI11`: (Required) The ticker symbol you want to predict for.

### Help

To see all available commands and options:

```bash
python src/app.py --help
```

Or for a specific command:

```bash
python src/app.py train --help
python src/app.py predict --help
```

## Architecture Overview

The project is built around a modular service architecture for asset prediction, with each stage of the pipeline implemented as a reusable service class. These can be used directly in Python or accessed via a REST API.

### Service Classes
- **B3DataLoadingService**: Loads B3 market data.
- **B3ModelPreprocessingService**: Handles data preprocessing, feature validation, and target label generation.
- **B3ModelTrainingService**: Handles model training and data splitting.
- **B3ModelEvaluationService**: Handles model evaluation and visualization.
- **B3ModelSavingService**: Handles model persistence (saving/loading models).

## REST API (Flask)

A Flask-based REST API exposes all training and prediction stages as endpoints. The API is documented with OpenAPI/Swagger (see `/swagger` when running the server).

### Main Endpoints
- `POST /api/b3/load-data`: Load B3 market data
- `POST /api/b3/preprocess-data`: Preprocess loaded data
- `POST /api/b3/split-data`: Split data into train/validation/test sets
- `POST /api/b3/train-model`: Train the model (with hyperparameter tuning)
- `POST /api/b3/evaluate-model`: Evaluate the trained model
- `POST /api/b3/save-model`: Save the trained model
- `POST /api/b3/complete-training`: Run the complete training pipeline
- `POST /api/b3/predict`: Make predictions for a specific ticker
- `GET /api/b3/status`: Get current pipeline status
- `GET /api/b3/training-status`: Get current training status
- `POST /api/b3/clear-state`: Clear the pipeline state

### OpenAPI/Swagger
- The API is documented with Swagger UI, available at `/swagger` when the server is running.
- The OpenAPI spec is at `/swagger/swagger.yml`.

## Usage Examples

### Using the Python API
```python
from b3.service.model import B3Model

model = B3Model()
model.train(model_dir="models", n_jobs=5, test_size=0.2, val_size=0.2)
predictions = model.predict(new_data, model_dir="models")
```

### Using Individual Services
```python
from b3.service.data.db.b3_featured.data_loading_service import B3DataLoadingService
from b3.service.model.model_preprocessing_service import B3ModelPreprocessingService
from b3.service.model.model_training_service import B3ModelTrainingService

data_service = B3DataLoadingService()
df = data_service.load_data()

preprocessing_service = B3ModelPreprocessingService()
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
- **Algorithm**: Random Forest Classifier
- **Target**: Asset price direction prediction (up/down)
- **Features**: Technical indicators, price movements, volume patterns
- **Evaluation**: Cross-validation with train/validation/test splits

### Training Data Requirements
- **Source**: B3 historical data from asset-data-lake
- **Time Period**: Configurable (default: recent 2 years)
- **Minimum Records**: 1000+ per asset for reliable predictions
- **Update Frequency**: Retrain when new data is available

### Model Evaluation Results
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
sudo systemctl status asset-predict-model

# View logs
sudo journalctl -u asset-predict-model -f

# Restart service
sudo systemctl restart asset-predict-model
```

## Model Versioning and Persistence

### Model Storage
- **Format**: Joblib serialized models
- **Location**: `models/` directory
- **Naming**: `b3_model.joblib`
- **Backup**: Previous versions archived before updates

### Model Updates
- **Trigger**: Manual retraining or scheduled updates
- **Process**: Load new data → Retrain → Validate → Deploy
- **Rollback**: Previous model versions available for quick rollback

## API Integration

### Frontend Integration
The model API integrates with the Angular frontend:
- **Endpoint**: `POST /api/b3/predict`
- **Input**: `{"ticker": "PETR4"}`
- **Output**: Prediction result with confidence

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

### Health Checks
- **Endpoint**: `GET /api/b3/status`
- **Metrics**: Model loaded, data connection, prediction latency
- **Alerts**: Service down, prediction failures

### Logging
- **Level**: INFO for normal operations, ERROR for failures
- **Format**: Structured JSON logs
- **Retention**: 30 days (configurable)

## License
[MIT License](LICENSE)

---
[Leia em português](README.pt-br.md)
