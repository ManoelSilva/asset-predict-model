import logging
from typing import Callable, Optional, Tuple, List, Any

import pandas as pd
from pandas import DataFrame

from constants import FEATURE_SET


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


def create_model_from_config(config):
    """
    Create a model instance from a TrainingConfig object.

    Args:
        config: TrainingConfig object containing model parameters

    Returns:
        BaseModel: Instantiated model
    """
    from b3.service.pipeline.model.factory import ModelFactory

    model_params = {}
    if is_rf_model(config.model_type):
        model_params = config.get_rf_config_dict()
    elif is_lstm_model(config.model_type):
        model_params = config.get_lstm_config_dict()

    return ModelFactory.get_model(config.model_type, **model_params)


def prepare_enrichment_dataframe(df: pd.DataFrame, max_samples: int = 5,
                                 ticker_validator: Optional[Callable[[pd.DataFrame, str], bool]] = None,
                                 check_limit: int = 20) -> (tuple[None, list[Any], None] |
                                                            tuple[DataFrame, list[Any] | Any, Any]):
    """
    Common function to prepare dataframe for prediction enrichment.
    Handles validation, ticker sampling, column initialization, and filtering.
    
    Args:
        df: Dataframe to enrich (modified in-place)
        max_samples: Maximum number of tickers to sample for predictions
        ticker_validator: Optional function to validate tickers (df, ticker) -> bool
        check_limit: Maximum number of tickers to check for validation
        
    Returns:
        Tuple of (original_df, sample_tickers, filtered_sub_df)
        Returns (None, [], None) if validation fails
    """
    if df is None or df.empty:
        return None, [], None

    if 'ticker' not in df.columns:
        return None, [], None

    all_tickers = df['ticker'].unique()

    # Sample tickers with optional validation
    if ticker_validator:
        valid_tickers = []
        check_limit = min(len(all_tickers), check_limit)

        for t in all_tickers[:check_limit]:
            if ticker_validator(df, t):
                valid_tickers.append(t)

        sample_tickers = valid_tickers[:max_samples]
        if not sample_tickers:
            sample_tickers = all_tickers[:max_samples]
    else:
        sample_tickers = all_tickers[:max_samples]

    logging.info(f"Generating predictions for visualization tickers: {sample_tickers}")

    # Initialize columns
    df['predicted_action'] = None

    # Filter dataframe for all sample tickers
    mask = df['ticker'].isin(sample_tickers)
    if not mask.any():
        return None, [], None

    sub_df = df[mask].copy()

    return df, sample_tickers, sub_df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Extract feature columns from dataframe based on FEATURE_SET.
    
    Args:
        df: Dataframe to extract features from
        
    Returns:
        List of feature column names that exist in the dataframe
    """
    feature_cols = [c for c in FEATURE_SET if c in df.columns]

    if not feature_cols:
        logging.warning(f"No valid features found for enrichment. Expected some of: {FEATURE_SET}")

    return feature_cols
