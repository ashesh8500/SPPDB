import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yfinance
import pandas as pd
from datetime import datetime, timedelta
import altair as alt
import numpy as np

class Portfolio:
    
    def __init__(self, portfolio: dict):
        self.portfolio = portfolio
        self.symbols = list(portfolio.keys())
        self.prices: pd.DataFrame = self.get_prices(portfolio)
        self.dist = pd.DataFrame(self.present_distribution(), index=[datetime.now().date()])
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
    
    def get_attractiveness(self, n_years=3):
        attractiveness_df = pd.DataFrame()
        for ticker in self.symbols:
            historical_prices: pd.DataFrame = self.prices[ticker].loc[datetime.today()-timedelta(days=n_years*365):]
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
        
        
        
def main():
    st.title('Portfolio Analysis')
    st.write('Welcome to the Portfolio Analysis App')
    st.write('This app will help you analyze your portfolio and make informed decisions')
    
    portfolio = {
        'AAPL': 10,
        'MSFT': 20,
        'GOOGL': 5,
        'AMZN': 3,
        'META': 7,
        'TSLA': 5,
        'NFLX': 10
    }
    
    my_portfolio = Portfolio(portfolio)
    st.write('Your Portfolio')
    st.write(my_portfolio.portfolio)
    st.write('Your Portfolio Value')
    st.write(my_portfolio.cash_value)
    st.write('Your Portfolio Distribution')
    st.write(my_portfolio.dist)
    st.write('Your Portfolio Attractiveness')
    df = my_portfolio.get_attractiveness()
    df.index.name = 'Ticker'
    st.dataframe(df)
    df['Stock'] = df.index
    st.write('Your Portfolio Attractiveness Plot')
    # Calculate color based on attractiveness
    df['Color'] = df['Attractiveness'].apply(lambda x: 'green' if x > 0 else 'red')

    # Define color scale for gradientation
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

    # Display the chart
    st.vega_lite_chart(chart.to_dict(), use_container_width=True)

if __name__ == '__main__':
    main()
