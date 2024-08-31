import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from rebalancer.Portfolio import Portfolio
import json
from pdb import Pdb
pdb = Pdb()

def main():
    st.title('Prototype')
    st.write('Dashboard overview of your portfolio and its attractiveness')

    with open("portfolio.json", "r") as f:
        portfolio_2 = json.load(f)
    tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "TATAMOTORS.NS", "SUNPHARMA.NS", "SBIN.NS", "TATAPOWER.NS", "VEDL.NS"]
    init_weights = np.ones(len(tickers)) / len(tickers)
    portfolio = dict(zip(tickers, init_weights))

    my_portfolio = Portfolio(portfolio)
    st.write('Your Portfolio')
    st.dataframe(pd.DataFrame(my_portfolio.portfolio, index=['Shares']))
    st.write('Your Portfolio Value')
    st.write(f'${"{:,.2f}".format(my_portfolio.cash_value)}')
    st.write('Your Portfolio Distribution')
    st.write(my_portfolio.dist)
    st.write('Your Portfolio Attractiveness')
    st.slider('Select the number of years to calculate attractiveness', min_value=1, max_value=5, value=3, key='years')
    st.date_input('Select the date to calculate attractiveness', value=datetime.now(), key='date')
    df = my_portfolio.get_attractiveness(n_years=int(st.session_state.years), date_of_calculation=st.session_state.date)
    df.index.name = 'Ticker'
    st.dataframe(df)
    st.write('Your Portfolio Performance')
    st.date_input('Select the date to calculate performance', value=datetime.now(), key='performance_date')
    st.dataframe(my_portfolio.get_performance(n_years=[3, 4], date_of_calculation=st.session_state.performance_date))
    df['Stock'] = df.index

    # Calculate color based on attractiveness
    df['Color'] = df['Attractiveness'].apply(lambda x: 'red' if x > 0 else 'green')

    # Create Plotly chart for Attractiveness
    df_sorted = df.sort_values(by='Attractiveness', ascending=False)
    chart = px.bar(df_sorted, x='Stock', y='Attractiveness', color='Attractiveness',
                   color_continuous_scale=px.colors.diverging.RdYlGn[::-1], title='Portfolio Attractiveness Plot')
    st.plotly_chart(chart, use_container_width=True)

    # plotting the weighted attractiveness scores
    df["dist"] = my_portfolio.dist.iloc[0]
    df["Weighted Attractiveness"] = df["Attractiveness"] * df["dist"]
    df_sorted_weighted = df.sort_values(by='Weighted Attractiveness', ascending=False)
    weighted_chart = px.bar(df_sorted_weighted, x='Stock', y='Weighted Attractiveness', color='Weighted Attractiveness',
                            color_continuous_scale=px.colors.diverging.RdYlGn[::-1], title='Portfolio Weighted Attractiveness Plot')
    st.plotly_chart(weighted_chart, use_container_width=True)

    # plotting the 3 year performance of the portfolio
    st.write('Portfolio Performance')
    prices_df = my_portfolio.prices.copy()
    prices_df.index.name = 'Date'  # Ensure the index is named 'Date'
    performance_df = prices_df.reset_index().melt('Date', var_name='Stock', value_name='Price')
    performance_chart = px.line(performance_df, x='Date', y='Price', color='Stock', title='Portfolio Performance')
    st.plotly_chart(performance_chart, use_container_width=True)

    # Optimization and Backtesting
    st.write('Optimize and Backtest Portfolio')
    history_len = st.number_input('History Length (years)', min_value=-1, max_value=10, value=1)
    num_tests = st.number_input('Number of Tests', min_value=100, max_value=2000, value=2000)
    every_nth = st.number_input('Rebalance Frequency (days)', min_value=1, max_value=365, value=30)

    # if st.button('Run Optimization'):
    optimized_portfolio = my_portfolio.optimize_and_backtest(num_tests=int(num_tests), n_years=int(history_len), every_nth=int(every_nth))
    st.write('Optimization Results')
    st.dataframe(optimized_portfolio)

    st.write('Portfolio Allocation Over Time')
    allocation_fig = my_portfolio.plot_allocation()
    st.plotly_chart(allocation_fig, use_container_width=True)

    st.write('Portfolio Performance Over Time')
    optimization_fig = my_portfolio.plot_optimization()
    st.plotly_chart(optimization_fig, use_container_width=True)

    st.write('Orders')
    st.dataframe(my_portfolio.pyopt.orders.records_readable)
    st.write('Order Counts')
    st.dataframe(my_portfolio.pyopt.orders.records_readable\
        .loc[:,['Column', 'Side']]\
        .groupby('Column')\
        .value_counts().unstack(fill_value=0))
    st.write('Positions')
    st.dataframe(my_portfolio.pyopt.trades.records_readable)




if __name__ == '__main__':
    main()
