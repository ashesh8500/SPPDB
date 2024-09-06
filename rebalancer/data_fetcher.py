import yfinance as yf
import pandas as pd
from typing import List, Dict

def fetch_stock_data(tickers: List[str], period: str = '5y') -> pd.DataFrame:
    """Fetch historical stock data for given tickers."""
    data = yf.download(tickers, period=period)
    return data['Adj Close']

def fetch_index_holdings(index_ticker: str) -> Dict[str, float]:
    """Fetch top holdings of a given index ETF."""
    index = yf.Ticker(index_ticker)
    holdings = index.info.get('holdings', [])
    return {holding['symbol']: holding['holdingPercent'] for holding in holdings}

def fetch_market_data(market_index: str = '^GSPC', period: str = '5y') -> pd.DataFrame:
    """Fetch market index data for comparison."""
    market_data = yf.download(market_index, period=period)
    return market_data['Adj Close']
