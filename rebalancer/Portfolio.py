from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import vectorbt as vbt
from .data_fetcher import fetch_stock_data, fetch_index_holdings, fetch_market_data
from .analysis import calculate_attractiveness, calculate_performance
from .optimizer import PortfolioOptimizer, plot_allocation

class Portfolio:
    def __init__(self, portfolio: Dict[str, float]):
        self.portfolio = portfolio
        self.symbols: List[str] = list(portfolio.keys())
        self.prices: pd.DataFrame = fetch_stock_data(self.symbols)
        self.dist = pd.DataFrame(self.present_distribution(), index=[pd.Timestamp(datetime.now().date())])
        self.cash_value = sum(map(lambda x: x[1] * self.portfolio[x[0]], self.prices.iloc[-1].items()))
        self.ndx_holdings = fetch_index_holdings('QQQ')
        self.pyopt = None
        self.optimizer = PortfolioOptimizer(self.portfolio)

    def present_distribution(self) -> Dict[str, float]:
        total = sum(self.portfolio.values())
        return {ticker: shares/total for ticker, shares in self.portfolio.items()}

    def get_attractiveness(self, n_years: int = 3, date_of_calculation: datetime = datetime.now()) -> pd.DataFrame:
        return calculate_attractiveness(self.prices, n_years, date_of_calculation)

    def get_performance(self, n_years: List[int] = [3, 4], date_of_calculation: datetime = datetime.now()) -> pd.DataFrame:
        return calculate_performance(self.prices, self.portfolio, n_years, date_of_calculation)

    def optimize_and_backtest(self, n_years: Optional[int] = None, num_tests: int = 1000, every_nth: int = 30):
        if n_years is None:
            n_years = len(self.prices)
        price = self.prices.loc[datetime.now().date() - timedelta(days=n_years*365):]
        self.pyopt = self.optimizer.run_simulation(price)
        return self.pyopt.stats()

    def plot_optimization(self):
        if self.pyopt is None:
            raise ValueError("Optimization has not been run yet. Please run optimize_and_backtest first.")
        return self.pyopt.plot()

    def plot_allocation(self):
        if self.pyopt is None:
            raise ValueError("Optimization has not been run yet. Please run optimize_and_backtest first.")
        return plot_allocation(self.pyopt, self.symbols)

    def get_orders_today(self, target_weights: Dict[str, float]) -> Dict[str, float]:
        current_weights = self.present_distribution()
        orders = {}
        for symbol in self.symbols:
            current_weight = current_weights.get(symbol, 0)
            target_weight = target_weights.get(symbol, 0)
            weight_diff = target_weight - current_weight
            if weight_diff != 0:
                orders[symbol] = weight_diff
        return orders

    def get_optimized_weights(self) -> pd.DataFrame:
        if self.pyopt is None:
            raise ValueError("Optimization has not been run yet. Please run optimize_and_backtest first.")
        return self.pyopt.asset_value(group_by=False).vbt / self.pyopt.value().vbt

    def get_market_comparison(self, market_index: str = '^GSPC') -> pd.DataFrame:
        market_data = fetch_market_data(market_index)
        portfolio_value = (self.prices * pd.Series(self.portfolio)).sum(axis=1)
        comparison = pd.concat([portfolio_value, market_data], axis=1)
        comparison.columns = ['Portfolio', 'Market']
        return comparison
