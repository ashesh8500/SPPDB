
from matplotlib import pyplot as plt
import pandas as pd
from pyparsing import col
import yfinance
from datetime import datetime, timedelta
import seaborn as sns


class Portfolio:

    def __init__(self, portfolio: dict):
        self.portfolio = portfolio
        self.symbols = list(portfolio.keys())
        self.prices: pd.DataFrame = self.get_prices(portfolio)
        self.dist = pd.DataFrame(self.present_distribution(), index=[datetime.now().date()]).map('{:.2%}'.format)
        self.cash_value = sum(map(lambda x: x[1] * self.portfolio[x[0]], self.prices.iloc[-1].items()))

    def present_distribution(self):
        total = sum(self.portfolio.values())
        distribution = {ticker: shares/total for ticker, shares in self.portfolio.items()}
        return distribution

    def get_prices(self, portfolio: dict):
        tickers = list(portfolio.keys())
        data = yfinance.download(tickers, period='5y')
        prices = data['Adj Close'].round(2)
        return prices

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
        df = pd.DataFrame(index=self.symbols)
        for ticker, dist in self.portfolio.items():
            df.loc[ticker, '% of portfolio'] = dist
        for years in n_years:
            historical_prices = self.prices.loc[date_of_calculation - timedelta(days=years*365):]
            for ticker in self.symbols:
                performance = (historical_prices[ticker].iloc[-1] - historical_prices[ticker].iloc[0]) / historical_prices[ticker].iloc[0]
                df.loc[ticker, f'{years}y'] = performance * 100
            df['contributed_growth'] = df['% of portfolio'] * df[f'{years}y']
            df.loc['portfolio', f'{years}y'] = df['contributed_growth'].sum()
            
        df.drop('contributed_growth', axis=1, inplace=True)
        return df.round(2)
