from b3.service.pipeline.model.config import ModelConfig


class LSTMConfig(ModelConfig):
    """Configuration class for LSTM Multi-Task Learning pipeline."""

    def __init__(self, lookback: int = 32, horizon: int = 1, units: int = 64,
                 dropout: float = 0.3, learning_rate: float = 1e-3, epochs: int = 25,
                 batch_size: int = 128, loss_weight_action: float = 10.0,
                 loss_weight_return: float = 100.0, price_col: str = "close",
                 early_stopping_patience: int = 15, early_stopping_min_delta: float = 100.0,
                 gradient_clip_norm: float = 1.0, weight_decay: float = 1e-5,
                 lr_scheduler_patience: int = 5, lr_scheduler_factor: float = 0.5,
                 lr_scheduler_min_lr: float = 1e-6, use_class_weights: bool = False,
                 focal_loss_gamma: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.lookback = lookback
        self.horizon = horizon
        self.units = units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_weight_action = loss_weight_action
        self.loss_weight_return = loss_weight_return
        self.price_col = price_col
        # Early stopping parameters
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        # Gradient clipping
        self.gradient_clip_norm = gradient_clip_norm
        # L2 regularization
        self.weight_decay = weight_decay
        # Learning rate scheduler
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_min_lr = lr_scheduler_min_lr
        # Class imbalance handling
        self.use_class_weights = use_class_weights
        self.focal_loss_gamma = focal_loss_gamma
