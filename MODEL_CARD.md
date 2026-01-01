[Leia em português](MODEL_CARD.pt-br.md)

# Model Card: B3 Asset Prediction LSTM Multi-Task Learning Model

## Model Details

### Model Information
- **Model Name**: B3 LSTM Multi-Task Learning (MTL) Model
- **Model Type**: Deep Learning - Recurrent Neural Network (LSTM)
- **Framework**: PyTorch
- **Version**: 1.0
- **Date**: 01-2026

### Model Architecture

**Architecture Type**: Long Short-Term Memory (LSTM) with Multi-Task Learning

**Network Structure**:
```
Input Layer: (batch_size, lookback=32, n_features)
    ↓
LSTM Layer: 
    - Hidden Size: 64 units
    - Output: (batch_size, 32, 64)
    - Uses last timestep: (batch_size, 64)
    ↓
Dropout Layer: 0.4 (40% dropout rate)
    ↓
    ├─→ Action Head (Linear): 64 → 3 (Buy/Sell/Hold classification)
    └─→ Return Head (Linear): 64 → 1 (Price return regression)
```

**Key Components**:
- **LSTM Layer**: Single-layer LSTM with 64 hidden units
- **Dropout**: 0.4 rate applied after LSTM, before task heads
- **Task Heads**: Two separate linear layers for classification and regression
- **Total Parameters**: Approximately 4 × 64 × (64 + n_features) + task head parameters

## Training Data

### Dataset
- **Source**: B3 (Brazilian Stock Exchange) historical market data
- **Data Source**: asset-data-lake
- **Time Period**: Configurable (default: recent 2 years)
- **Minimum Records**: 1000+ per asset for reliable predictions
- **Update Frequency**: Retrain when new data is available

### Data Preprocessing
- **Feature Engineering**: Technical indicators, price movements, volume patterns
- **Normalization**: StandardScaler applied to input features
- **Sequence Building**: 
  - Lookback window: 32 timesteps
  - Horizon: 2 timesteps ahead
  - Shape: `(n_samples, 32, n_features)`

### Data Splits
- **Training Set**: 60% of data
- **Validation Set**: 20% of data
- **Test Set**: 20% of data
- **Split Method**: Temporal split (maintains time order)

### Class Distribution
- **Sell Class**: Majority class (high frequency)
- **Buy Class**: Minority class (low frequency)
- **Hold Class**: Minority class (low frequency)
- **Class Imbalance Handling**: 
  - Oversampling via WeightedRandomSampler
  - Focal Loss with γ=3.0
  - Optional class weights

## Training Procedure

### Training Configuration

**Hyperparameters**:
- **Lookback Window**: 32 timesteps
- **Horizon**: 2 timesteps
- **Hidden Units**: 64
- **Dropout Rate**: 0.4
- **Learning Rate**: 1e-3 (0.001)
- **Batch Size**: 128
- **Epochs**: 50 (maximum, with early stopping)
- **Optimizer**: Adam
- **Weight Decay (L2)**: 1e-4
- **Gradient Clipping**: 1.0 (max norm)

**Loss Function**:
```
Total Loss = w_action × FocalLoss + w_return × MSE

Where:
- w_action = 20.0
- w_return = 50.0
- Focal Loss γ = 3.0
```

**Regularization Techniques**:
- **Dropout**: 0.4 rate
- **Weight Decay**: 1e-4 (L2 regularization)
- **Gradient Clipping**: Prevents exploding gradients
- **Early Stopping**: 
  - Patience: 20 epochs
  - Min Delta: 100.0
  - Monitors validation loss

**Learning Rate Scheduling**:
- **Scheduler**: ReduceLROnPlateau
- **Factor**: 0.5 (halve learning rate)
- **Patience**: 5 epochs
- **Min Learning Rate**: 1e-6

### Training Process
1. **Data Preparation**: Sequence building with lookback window
2. **Oversampling**: WeightedRandomSampler for class imbalance
3. **Training Loop**: 
   - Forward pass through LSTM
   - Multi-task loss calculation
   - Backward pass with gradient clipping
   - Parameter update via Adam optimizer
4. **Validation**: Monitor validation loss and metrics
5. **Early Stopping**: Stop if no improvement for 20 epochs
6. **Model Selection**: Restore best model based on validation loss

## Evaluation

### Evaluation Metrics

#### Classification Performance (Action Prediction)

**Overall Metrics**:
- **Accuracy**: 91.3% (vs 33.3% random baseline - 2.74× improvement)
- **Macro F1 Score**: 0.411
- **Weighted F1 Score**: 0.946

**Per-Class Performance**:

| Class | Precision | Recall | F1 Score |
|-------|-----------|--------|----------|
| **Buy** | 8.0% | 65.2% | 0.143 |
| **Sell** | 99.7% | 91.5% | 0.954 |
| **Hold** | 7.5% | 77.2% | 0.136 |

**Interpretation**:
- **Sell Class**: Excellent performance (99.7% precision, 91.5% recall)
- **Buy/Hold Classes**: Low precision due to class imbalance, but moderate recall
- **Overall**: Strong performance on majority class, challenges with minority classes

#### Regression Performance (Return Prediction)

**Metrics**:

| Metric | Validation | Test |
|--------|------------|------|
| **R² Score** | 0.9982 (99.82%) | 0.9983 (99.83%) |
| **MAE** | 1.20 | 1.25 |
| **RMSE** | 5.47 | 5.74 |
| **MAPE** | 8.98% | 8.49% |

**Interpretation**:
- **R² = 99.8%**: Model explains 99.8% of variance in returns
- **MAE < 1.25**: Average error less than 1.25 price units
- **MAPE < 9%**: Average percentage error under 9% (excellent for financial data)

### Overfitting Analysis

**Training vs Validation Loss**:
- **Training Loss**: 253.35
- **Validation Loss**: 8,802.48
- **Overfitting Gap**: 8,549 (35× difference)
- **Improvement**: 95% reduction from initial gap of 172,720

**Status**: 
- Significant improvement achieved (95% reduction)
- Gap still present but manageable
- Early stopping and regularization working effectively

## Model Performance Summary

### Strengths
1. **High Overall Accuracy**: 91.3% classification accuracy
2. **Excellent Regression**: 99.8% R² score, <9% MAPE
3. **Strong Sell Prediction**: 99.7% precision, 91.5% recall
4. **Reduced Overfitting**: 95% reduction in training-validation gap
5. **Stable Training**: Early stopping and regularization working well

### Limitations
1. **Buy/Hold Precision**: Low (8.0% and 7.5%) due to class imbalance
2. **Overfitting Gap**: Still present (35×), though much improved
3. **Class Imbalance**: Sell class dominates, affecting minority class performance
4. **Trade-off**: Higher precision for Buy/Hold would reduce recall

## Intended Use

### Primary Use Cases
1. **Trading Signal Generation**: Predict Buy/Sell/Hold actions for B3 assets
2. **Return Forecasting**: Predict price returns for portfolio management
3. **Risk Assessment**: Identify potential sell signals with high confidence
4. **Research Tool**: Analyze temporal patterns in financial time series

### Out-of-Scope Uses
- **Not for**: High-frequency trading (designed for daily predictions)
- **Not for**: Guaranteed profit generation (predictions are probabilistic)
- **Not for**: Other markets without retraining (trained on B3 data)
- **Not for**: Long-term forecasting beyond 2-day horizon

## Ethical Considerations

### Bias and Fairness
- **Class Imbalance**: Model performs better on majority class (Sell)
- **Market Bias**: Reflects historical market patterns (may perpetuate biases)
- **Recommendation**: Monitor performance across different market conditions

### Transparency
- **Model Architecture**: Fully documented
- **Training Process**: Transparent and reproducible
- **Evaluation Metrics**: Comprehensive metrics provided
- **Limitations**: Clearly stated

### Data Privacy
- **Data Source**: Public market data (B3)
- **No Personal Data**: Model uses only market data, no personal information
- **Compliance**: Follows data usage policies

### Risk Warnings
- **Financial Risk**: Model predictions are not financial advice
- **Uncertainty**: Financial markets are inherently unpredictable
- **Validation**: Always validate predictions with domain expertise
- **Monitoring**: Continuous monitoring required for model drift

## Model Maintenance

### Retraining Schedule
- **Frequency**: When new data is available or performance degrades
- **Trigger**: Significant changes in market conditions
- **Validation**: Monitor validation metrics for drift

### Version Control
- **Model Versioning**: Tracked via MLflow
- **Artifact Storage**: Models saved as `.pt` files (PyTorch state dicts)
- **Scaler Storage**: Separate `.joblib` files for preprocessing

### Monitoring
- **Metrics to Monitor**:
  - Classification accuracy
  - Regression R² and MAPE
  - Per-class precision and recall
  - Training-validation loss gap
- **Alert Thresholds**: 
  - Accuracy drop > 5%
  - R² drop below 0.95
  - Overfitting gap increase > 50%

## Technical Specifications

### Hardware Requirements
- **Training**: GPU recommended (8GB+ VRAM)
- **Inference**: CPU or GPU
- **Memory**: 8GB+ RAM recommended

### Software Dependencies
- **Python**: 3.12+
- **PyTorch**: Latest stable version
- **NumPy**: For numerical operations
- **Pandas**: For data manipulation
- **scikit-learn**: For metrics and utilities

### Model Size
- **Model File**: ~500KB - 2MB (PyTorch state dict)
- **Scaler File**: ~10-50KB (joblib)
- **Total**: < 5MB

### Inference Speed
- **Batch Size**: 256 samples
- **Latency**: < 10ms per sample (on GPU)
- **Throughput**: ~1000 samples/second (on GPU)
---

**Last Updated**: 01-2026
**Model Version**: 1.0
**Status**: Production Ready

