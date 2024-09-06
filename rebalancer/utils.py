import numpy as np
import pandas as pd
from typing import Dict, List

def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate returns from price data."""
    return prices.pct_change(fill_method=None)

def calculate_volatility(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """Calculate rolling volatility of returns."""
    return returns.rolling(window=window, min_periods=window // 2).std() * np.sqrt(252)

def calculate_momentum(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """Calculate rolling momentum of returns."""
    return returns.rolling(window=window, min_periods=window // 2).sum()

def clip_weights(weights: pd.Series, min_weight: float, max_weight: float) -> pd.Series:
    """Clip weights to specified range and renormalize."""
    clipped_weights = weights.clip(min_weight, max_weight)
    return clipped_weights / clipped_weights.sum()
