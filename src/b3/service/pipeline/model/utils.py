"""
Utility functions for model type handling and normalization.
Centralizes model type alias handling to eliminate duplication.
"""


def normalize_model_type(model_type: str) -> str:
    """
    Normalize model type aliases to canonical form.
    
    Args:
        model_type: Model type string (can be alias like "random_forest" or "lstm_mtl")
        
    Returns:
        Canonical model type string ("rf" or "lstm")
        
    Examples:
        >>> normalize_model_type("rf")
        'rf'
        >>> normalize_model_type("random_forest")
        'rf'
        >>> normalize_model_type("lstm")
        'lstm'
        >>> normalize_model_type("lstm_mtl")
        'lstm'
    """
    if model_type in ["rf", "random_forest"]:
        return "rf"
    elif model_type in ["lstm", "lstm_mtl"]:
        return "lstm"
    else:
        return model_type


def is_rf_model(model_type: str) -> bool:
    """
    Check if model type is Random Forest.
    
    Args:
        model_type: Model type string to check
        
    Returns:
        True if model type is Random Forest (including aliases), False otherwise
        
    Examples:
        >>> is_rf_model("rf")
        True
        >>> is_rf_model("random_forest")
        True
        >>> is_rf_model("lstm")
        False
    """
    return normalize_model_type(model_type) == "rf"


def is_lstm_model(model_type: str) -> bool:
    """
    Check if model type is LSTM.
    
    Args:
        model_type: Model type string to check
        
    Returns:
        True if model type is LSTM (including aliases), False otherwise
        
    Examples:
        >>> is_lstm_model("lstm")
        True
        >>> is_lstm_model("lstm_mtl")
        True
        >>> is_lstm_model("rf")
        False
    """
    return normalize_model_type(model_type) == "lstm"

