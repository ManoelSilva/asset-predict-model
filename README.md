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

## License
MIT License

---
[Leia em português](README.pt-br.md)
