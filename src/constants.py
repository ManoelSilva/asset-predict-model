# API configuration
DEFAULT_API_HOST = "0.0.0.0"
DEFAULT_API_PORT = 5000
SECONDARY_API_PORT = 5001
LOCALHOST_URL = "http://localhost:5000"

# HTTP status codes
HTTP_STATUS_OK = 200
HTTP_STATUS_BAD_REQUEST = 400
HTTP_STATUS_INTERNAL_SERVER_ERROR = 500

# Formatting
SEPARATOR_LINE = "=" * 50

# Serialization
MODEL_STORAGE_KEY = "trained_model"

# Default values for data splitting and training
DEFAULT_TEST_SIZE = 0.8
DEFAULT_VAL_SIZE = 0.2
DEFAULT_N_JOBS = 5

# Feature set for the B3 pipeline
FEATURE_SET = [
    'rolling_volatility_5',
    'moving_avg_10',
    'macd',
    'rsi_14',
    'volume_change',
    'avg_volume_10',
    'best_buy_sell_spread',
    'close_to_best_buy',
    'price_momentum_5',
    'high_breakout_20',
    'bollinger_upper',
    'stochastic_14'
]

# Model Random State
RANDOM_STATE = 42
