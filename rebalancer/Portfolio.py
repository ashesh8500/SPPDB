
# from matplotlib import pyplot as plt
# import pandas as pd
# from sympy import plot
# import yfinance
# from datetime import datetime, timedelta
# import seaborn as sns
# import plotly.graph_objects as go
# from rebalancer.optimizer import run_simulation, plot_allocation
# import os
# import numpy as np
# import pandas as pd
# import yfinance as yf
# from datetime import datetime
# from typing import List, Dict, Optional


# class Portfolio:

#     def __init__(self, portfolio: dict):
#         self.portfolio = portfolio
#         self.symbols: List[str] = list(portfolio.keys())
#         self.prices: pd.DataFrame = self.get_prices(portfolio)
#         self.dist = pd.DataFrame(self.present_distribution(), index=[datetime.now().date()])
#         self.cash_value = sum(map(lambda x: x[1] * self.portfolio[x[0]], self.prices.iloc[-1].items()))

#     def present_distribution(self):
#         total = sum(self.portfolio.values())
#         distribution = {ticker: shares/total for ticker, shares in self.portfolio.items()}
#         return distribution

#     def get_prices(self, portfolio: dict):
#         tickers = list(portfolio.keys())
#         data = yfinance.download(tickers, period='5y')
#         prices = data['Adj Close'].round(2)
#         return prices

#     def get_attractiveness(self, n_years=3, date_of_calculation=datetime.now()):
#         attractiveness_df = pd.DataFrame()
#         for ticker in self.symbols:
#             historical_prices: pd.DataFrame = self.prices[ticker].loc[date_of_calculation-timedelta(days=n_years*365):]
#             mean = historical_prices.mean()
#             std = historical_prices.std()
#             last_price = historical_prices.iloc[-1]
#             attractiveness = (last_price - mean)/std
#             attractiveness_df[ticker] = [attractiveness, last_price, mean, std]
#         attractiveness_df = attractiveness_df.T
#         attractiveness_df.columns = ['Attractiveness', 'Last Price', 'Mean', 'Std']
#         return attractiveness_df.round(2)

#     def plot_attractiveness(self):
#         self.get_attractiveness().pipe(
#             lambda df: sns.barplot(x=df.index, y=df['Attractiveness'], hue=df['Attractiveness'] > 0)
#         ).set(
#             title='Attractiveness of Stocks'
#         )
#         plt.show()

#     def get_performance(self, n_years: list = [3, 4], date_of_calculation=datetime.now()):
#         df = pd.DataFrame(index=self.symbols) #type: ignore
#         for ticker, dist in self.portfolio.items():
#             df.loc[ticker, '% of portfolio'] = dist
#         for years in n_years:
#             historical_prices: pd.DataFrame = self.prices.loc[date_of_calculation - timedelta(days=years*365):]
#             for ticker in self.symbols:
#                 performance = (historical_prices[ticker].iloc[-1] - historical_prices[ticker].iloc[0]) / historical_prices[ticker].iloc[0]
#                 df.loc[ticker, f'{years}y'] = performance * 100
#             df['contributed_growth'] = df['% of portfolio'] * df[f'{years}y']
#             df.loc['portfolio', f'{years}y'] = df['contributed_growth'].sum()

#         df.drop('contributed_growth', axis=1, inplace=True)
#         return df.round(2)


#     def optimize_and_backtest(self, n_years=-1, num_tests=1000, every_nth=30):
#         price = self.prices
#         self.pyopt = run_simulation(price, initial_weights=self.portfolio, num_tests=num_tests, every_nth=every_nth)
#         return self.pyopt.stats()

#     def plot_optimization(self):
#         return self.pyopt.plot()

#     def plot_allocation(self):
#         return plot_allocation(self.pyopt, self.symbols)







from matplotlib import pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import seaborn as sns
import plotly.graph_objects as go
from rebalancer.optimizer import run_simulation, plot_allocation
import os
import numpy as np
import yfinance as yf
from yahooquery import Ticker
from typing import List, Dict, Optional
import plotly.graph_objects as go
import pdb
def plot_qqq_performance(fig, qqq_data, **kwargs):
    fig.add_trace(go.Scatter(x=qqq_data.index, y=qqq_data['Close'], mode='lines', name='QQQ Performance'))

class Portfolio:

    def __init__(self, portfolio: dict):
        self.portfolio = portfolio
        self.symbols: List[str] = list(portfolio.keys())
        self.prices: pd.DataFrame = self.get_prices(portfolio)
        self.dist = pd.DataFrame(self.present_distribution(), index=[pd.Timestamp(datetime.now().date())])
        self.cash_value = sum(map(lambda x: x[1] * self.portfolio[x[0]], self.prices.iloc[-1].items()))
        self.ndx_holdings = self.get_ndx_holdings()

    def present_distribution(self):
        total = sum(self.portfolio.values())
        distribution = {ticker: shares/total for ticker, shares in self.portfolio.items()}
        return distribution

    def get_prices(self, portfolio: dict):
        tickers = list(portfolio.keys())
        data = Ticker(tickers).history(period='5y')
        prices = data['adjclose'].unstack(level=0).round(2)
        return prices

    def get_ndx_holdings(self):
        qqq = Ticker('QQQ')
        holdings = qqq.fund_top_holdings
        return holdings

    def get_attractiveness(self, n_years=3, date_of_calculation=datetime.now()):
        attractiveness_df = pd.DataFrame()
        for ticker in self.symbols:
            historical_prices: pd.DataFrame = self.prices[ticker].loc[date_of_calculation-timedelta(days=n_years*365):]
            mean = historical_prices.mean()
            std = historical_prices.std()
            last_price = historical_prices.iloc[-1]
            attractiveness = (last_price - mean)/std
            attractiveness_df[ticker] = [attractiveness, last_price, mean, std]
        attractiveness_df = attractiveness_df.T
        attractiveness_df.columns = ['Attractiveness', 'Last Price', 'Mean', 'Std']
        return attractiveness_df.round(2)

    def plot_attractiveness(self):
        self.get_attractiveness().pipe(
            lambda df: sns.barplot(x=df.index, y=df['Attractiveness'], hue=df['Attractiveness'] > 0)
        ).set(
            title='Attractiveness of Stocks'
        )
        plt.show()

    def get_performance(self, n_years: list = [3, 4], date_of_calculation=datetime.now()):
        df = pd.DataFrame(index=self.symbols) #type: ignore
        for ticker, dist in self.portfolio.items():
            df.loc[ticker, '% of portfolio'] = dist
        for years in n_years:
            historical_prices: pd.DataFrame = self.prices.loc[date_of_calculation - timedelta(days=years*365):]
            for ticker in self.symbols:
                performance = (historical_prices[ticker].iloc[-1] - historical_prices[ticker].iloc[0]) / historical_prices[ticker].iloc[0]
                df.loc[ticker, f'{years}y'] = performance * 100
            df['contributed_growth'] = df['% of portfolio'] * df[f'{years}y']
            df.loc['portfolio', f'{years}y'] = df['contributed_growth'].sum()

        df.drop('contributed_growth', axis=1, inplace=True)
        return df.round(2)

    def optimize_and_backtest(self, n_years=None, num_tests=1000, every_nth=30):
        if n_years is None:
            n_years = len(self.prices)
        price = self.prices.loc[datetime.now().date() - timedelta(days=n_years*365):]
        self.pyopt = run_simulation(price, initial_weights=self.portfolio, num_tests=num_tests, every_nth=every_nth)
        # pdb.set_trace()
        return self.pyopt.stats()

    def plot_optimization(self):
        return self.pyopt.plot()

    def plot_allocation(self):
        return plot_allocation(self.pyopt, self.symbols)

    def get_orders_today(self, target_weights: Dict[str, float]):
        current_weights = self.present_distribution()
        orders = {}
        for symbol in self.symbols:
            current_weight = current_weights.get(symbol, 0)
            target_weight = target_weights.get(symbol, 0)
            weight_diff = target_weight - current_weight
            if weight_diff != 0:
                orders[symbol] = weight_diff
        return orders

    def get_optimized_weights(self):
        if self.pyopt is None:
            raise ValueError("Optimization has not been run yet. Please run optimize_and_backtest first.")
        return self.pyopt.asset_value(group_by=False).vbt / self.pyopt.value().vbt
