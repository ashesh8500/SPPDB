# %%
import pandas as pd
import json

df = pd.read_csv('nasdaq_w.csv')

# %%
# df
portfolio = {row['ticker']: float(row['portfolio_weight'].replace("%","")) for index, row in df.iterrows()}
# %%
# Drop the ticker 'NVDA' and reweight the portfolio distribution
if 'NVDA' in portfolio:
    del portfolio['NVDA']
    portfolio.pop(list(portfolio.keys())[-1])
    total_weight = sum(portfolio.values())
    portfolio = {ticker: weight / total_weight for ticker, weight in portfolio.items()}
portfolio
# %%
with open('portfolio_c.json', 'w') as f:
    json.dump(portfolio, f)
# %%
from typing import Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# %config InlineBackend.figure_format = 'retina'
import yfinance as yfinance
import pandas as pd
from datetime import datetime, timedelta

class Portfolio:
    """
    A class representing a stock portfolio.

    Attributes:
        portfolio (dict): A dictionary of stock symbols and their respective quantities.
        symbols (list): A list of stock symbols in the portfolio.
        prices (pd.DataFrame): A DataFrame of historical prices of the stocks in the portfolio.
        dist (pd.DataFrame): A DataFrame of the current distribution of the portfolio.
        cash_value (float): The total cash value of the portfolio.
    """

    def __init__(self, portfolio: dict):
        """
        Initializes the portfolio with the given stocks and quantities.

        Args:
            portfolio (dict): A dictionary of stock symbols and their respective quantities.
        """
        self.portfolio = portfolio
        self.symbols = list(portfolio.keys())
        self.prices: pd.DataFrame = self.get_prices(portfolio)
        self.dist = pd.DataFrame(self.present_distribution(), index=[datetime.now().date()])
        self.cash_value = sum(map(lambda x: x[1] * self.portfolio[x[0]], self.prices.iloc[-1].items()))
        self.liquid = 0
    def present_distribution(self):
        """
        Calculates the current distribution of the portfolio based on the stock quantities.

        Returns:
            dict: A dictionary of stock symbols and their respective distribution percentages.
        """
        total = sum(self.portfolio.values())
        distribution = {ticker: shares/total for ticker, shares in self.portfolio.items()}
        return distribution

    def get_prices(self, portfolio: Optional[Dict[str, int]]=None):
        """
        Retrieves the historical prices of the stocks in the portfolio.

        Args:
            portfolio (dict): A dictionary of stock symbols and their respective quantities.

        Returns:
            pd.DataFrame: A DataFrame of historical prices of the stocks in the portfolio.
        """
        if portfolio is None:
            portfolio = self.portfolio
        tickers = list(portfolio.keys())
        data = yfinance.download(tickers, period='5y')
        prices = data['Adj Close'].round(2)
        return prices

    def get_attractiveness(self, n_years=3):
        """
        Calculates the attractiveness of each stock in the portfolio based on its historical prices.

        Args:
            n_years (int, optional): The number of years to consider for the attractiveness calculation. Defaults to 3.

        Returns:
            pd.DataFrame: A DataFrame of attractiveness scores for each stock in the portfolio.
        """
        attractiveness_df = pd.DataFrame()
        for ticker in self.symbols:
            historical_prices: pd.DataFrame = self.prices[ticker].loc[datetime.today()-timedelta(days=n_years*365):]
            mean = historical_prices.mean()
            std = historical_prices.std()
            last_price = historical_prices.iloc[-1]
            attractiveness = -(last_price - mean)/std
            attractiveness_df[ticker] = [attractiveness, last_price, mean, std]
        attractiveness_df = attractiveness_df.T
        attractiveness_df.columns = ['Attractiveness', 'Last Price', 'Mean', 'Std']
        return attractiveness_df.round(2)

    def plot_attractiveness(self):
        """
        Plots the attractiveness of each stock in the portfolio.
        """
        attractiveness_df = self.get_attractiveness()
        plt.figure(figsize=(10,6))
        sns.barplot(x=attractiveness_df.index, y=attractiveness_df['Attractiveness'], hue=attractiveness_df['Attractiveness'] > 0, palette=['red' if i < 0 else 'green' for i in attractiveness_df['Attractiveness']])
        plt.title('Attractiveness of Stocks')
        plt.xlabel('Stocks')
        plt.ylabel('Attractiveness')
        plt.show()

# %%
class SimulTrader:
    def __init__(self, portfolio: Portfolio, frequency='quarterly'):
        self.portfolio = portfolio
        self.frequency = frequency
        self.decisions = dict()

    def trade(self, ticker, volume: Optional[float] = None, price: Optional[float] = None):
        if volume is None and price is None:
            raise ValueError('Either volume or price must be provided')
        if price is not None:
            volume = price / self.portfolio.prices[ticker].iloc[-1]
        self.decisions[ticker] = volume
        self.portfolio.portfolio[ticker] += volume
        trade_value = volume * self.portfolio.prices[ticker].iloc[-1]
        self.portfolio.liquid -= trade_value
        return {"ticker": ticker, "volume": volume, "trade_value": trade_value}
    def calculate_trade_performance(self):
        prior_price = self.portfolio.prices.sum(axis=1).iloc[-1]
        self.attractiveness = self.portfolio.get_attractiveness().sort_values(by='Attractiveness', ascending=False)
        most_attractive = self.attractiveness.index[0]
        least_attractive = self.attractiveness.index[-1]

        if self.attractiveness.loc[most_attractive, 'Attractiveness'] > 0:
            trade_result = self.trade(most_attractive, volume=-1)
            self.trade(least_attractive, price=trade_result['trade_value'])

        current_price = self.portfolio.prices.sum(axis=1).iloc[-1]
        performace = (current_price - prior_price) / prior_price
        print(f'Prior Price: {prior_price}, Current Price: {current_price}, Performance: {performace}')
        return performace



# %%
portfolio = Portfolio(portfolio = {'TSLA':2, 'MSFT':3, 'SPOT':5, 'SNAP':10})
# %%
portfolio.present_distribution()
# %%
trader = SimulTrader(portfolio)
print(portfolio.portfolio)
trader.calculate_trade_performance()
# %% [markdown]
# ### Attractiveness weights
# 1. Thresholds of weighing that must be eventually learned by the model
# 2. weight based transaction on stocks apply to all stocks
# 3.

# %%
portfolio.get_attractiveness(n_years=3).sort_values(by='Attractiveness', ascending=False).reset_index(drop=True)

# %%
portfolio.plot_attractiveness()

# %%
import altair as alt
import numpy as np
df = portfolio.get_attractiveness(n_years=1)
df['Stock'] = df.index  # Add this line to specify the field name correctly
df['Color'] = df['Attractiveness'].apply(lambda x: 'green' if x > 0 else 'red')
# Define color scale for gradientation
color_scale = alt.Scale(
    domain=[df['Attractiveness'].min(), df['Attractiveness'].max()],
    range=['#FF0000', '#00FF00'],
    type='linear',
    interpolate='rgb',
    zero=False
)

# Create Vega-Lite chart
chart = alt.Chart(df).mark_bar().encode(
    x='Stock',
    y='Attractiveness',
    color=alt.Color('Attractiveness', scale=color_scale, sort=alt.SortField(field='Attractiveness', order='ascending')),
    tooltip=['Stock', 'Attractiveness']
).properties(
    title='Your Portfolio Attractiveness Plot'
)
chart

# %%


# %% [markdown]
# # Current Portfolio
#

# %%
import pandas as pd

pf = pd.read_csv('nasdaq weights adjustment.csv', index_col='ticker')
pf.drop('Unnamed: 0', axis=1, inplace=True)
pf.loc[:,['nd_weights', 'portfolio_weight']]=pf.loc[:,['nd_weights', 'portfolio_weight']].apply(
    lambda x: x.str.replace('%', '').astype(float)
)
pf = pf.loc[:,['company_name','nd_weights', 'portfolio_weight']].dropna()
pf

# %%
pf.loc[:, 'portfolio_weight'].sum()

# %%
from rebalancer.Portfolio import Portfolio
portfolio = Portfolio(portfolio = pf['portfolio_weight'].to_dict())

# %%
import pandas as pd
import numpy as np
from scipy.optimize import minimize

def optimize_portfolio(returns, weights=None):
    if weights is None:
        num_assets = len(returns.columns)
        weights = np.array([1.0 / num_assets] * num_assets)

    def neg_returns(weights):
        return -returns.agg('prod', axis=0).dot(weights)

    def volatility(weights):
        return np.sqrt(returns.agg('prod', axis=0).dot(weights) ** 2)

    def mean_variance(weights):
        return -returns.mean().dot(weights) / volatility(weights)

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(len(returns.columns)))

    result = minimize(mean_variance, weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

pf_returns = pf['portfolio_weight'].to_frame('returns').apply(lambda x: np.log(1+x))

pf_optimized = optimize_portfolio(pf_returns)

pf_optimized = pd.DataFrame(pf_optimized, index=pf_returns.index, columns=['optimized_weights'])
pf_optimized
