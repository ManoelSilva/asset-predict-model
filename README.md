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

## License
MIT License

---
[Leia em português](README.pt-br.md)
