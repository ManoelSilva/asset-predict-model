from b3.service.pipeline.model.config import ModelConfig


class LSTMConfig(ModelConfig):
    """Configuration class for LSTM Multi-Task Learning pipeline."""

    def __init__(self, lookback: int = 32, horizon: int = 1, units: int = 64,
                 dropout: float = 0.2, learning_rate: float = 1e-3, epochs: int = 25,
                 batch_size: int = 128, loss_weight_action: float = 1.0,
                 loss_weight_return: float = 1000.0, price_col: str = "close", **kwargs):
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
