import pandas as pd
import numpy as np
from typing import List, Dict
from datetime import datetime, timedelta

def calculate_attractiveness(prices: pd.DataFrame, n_years: int = 3, date_of_calculation: datetime = datetime.now()) -> pd.DataFrame:
    attractiveness_df = pd.DataFrame()
    for ticker in prices.columns:
        historical_prices = prices[ticker].loc[date_of_calculation-timedelta(days=n_years*365):]
        mean = historical_prices.mean()
        std = historical_prices.std()
        last_price = historical_prices.iloc[-1]
        attractiveness = (last_price - mean)/std
        attractiveness_df[ticker] = [attractiveness, last_price, mean, std]
    attractiveness_df = attractiveness_df.T
    attractiveness_df.columns = ['Attractiveness', 'Last Price', 'Mean', 'Std']
    return attractiveness_df.round(2)

def calculate_performance(prices: pd.DataFrame, portfolio: Dict[str, float], n_years: List[int] = [3, 4], date_of_calculation: datetime = datetime.now()) -> pd.DataFrame:
    df = pd.DataFrame(index=prices.columns)
    for ticker, dist in portfolio.items():
        df.loc[ticker, '% of portfolio'] = dist
    for years in n_years:
        historical_prices = prices.loc[date_of_calculation - timedelta(days=years*365):]
        for ticker in prices.columns:
            performance = (historical_prices[ticker].iloc[-1] - historical_prices[ticker].iloc[0]) / historical_prices[ticker].iloc[0]
            df.loc[ticker, f'{years}y'] = performance * 100
        df['contributed_growth'] = df['% of portfolio'] * df[f'{years}y']
        df.loc['portfolio', f'{years}y'] = df['contributed_growth'].sum()

    df.drop('contributed_growth', axis=1, inplace=True)
    return df.round(2)
