import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yfinance
import pandas as pd
from datetime import date, datetime, timedelta
from rebalancer.Portfolio import Portfolio
import altair as alt
import json

def main():
    st.title('Prototype')
    st.write('Dashboard overview of your portfolio and its attractiveness')

    portfolio_2 = {
        "META": 0.0934,
        "AMZN": 0.0978,
        "NFLX": 0.0369,
        "GOOG": 0.0429,
        "AAPL": 0.154077980787342,
    }
    with open("portfolio.json", "r") as f:
        portfolio_2 = json.load(f)

    my_portfolio = Portfolio(portfolio_2)
    st.write('Your Portfolio')
    st.dataframe(pd.DataFrame(my_portfolio.portfolio, index=['Shares']))
    st.write('Your Portfolio Value')
    st.write(f'${"{:,.2f}".format(my_portfolio.cash_value)}')
    st.write('Your Portfolio Distribution')
    st.write(my_portfolio.dist)
    st.write('Your Portfolio Attractiveness')
    st.slider('Select the number of years to calculate attractiveness', min_value=1, max_value=5, value=3, key='years')
    st.date_input('Select the date to calculate attractiveness', value=datetime.now(), key='date')
    df = my_portfolio.get_attractiveness(n_years=st.session_state.years, date_of_calculation=st.session_state.date)
    df.index.name = 'Ticker'
    st.dataframe(df)
    st.write('Your Portfolio Performance')
    st.date_input('Select the date to calculate performance', value=datetime.now(), key='performance_date')
    st.dataframe(my_portfolio.get_performance(n_years=[3, 4], date_of_calculation=st.session_state.performance_date))
    df['Stock'] = df.index

    # Calculate color based on attractiveness
    df['Color'] = df['Attractiveness'].apply(lambda x: 'green' if x > 0 else 'red')
    # Define color scale for gradientation
    color_scale = alt.Scale(
        domain=[df['Attractiveness'].min(), df['Attractiveness'].max()], # type: ignore
        range=['#00FF00', '#FF0000'],
        type='linear',
        interpolate='rgb',
        zero=False
    )

    # Create Vega-Lite chart
    df_sorted = df.sort_values(by='Attractiveness', ascending=False)
    chart = alt.Chart(df_sorted).mark_bar().encode(
        x=alt.X('Stock', sort='-y'),
        y='Attractiveness',
        color=alt.Color('Attractiveness', scale=color_scale, sort=alt.SortField(field='Attractiveness', order='ascending')), # type: ignore
        tooltip=['Stock', 'Attractiveness']
    ).properties(
        title='Portfolio Attractiveness Plot'
    )

    # Display the chart
    st.altair_chart(chart, use_container_width=True)

    # plotting the weighted attractiveness scores
    # Create a new column with the weighted attractiveness

    df['Weighted Attractiveness'] = df['Attractiveness'] * my_portfolio.dist
    weighted_chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Stock', sort='-y'),
        y='Weighted Attractiveness',
        color=alt.Color('Weighted Attractiveness', scale=color_scale, sort=alt.SortField(field='Weighted Attractiveness', order='ascending')), # type: ignore
        tooltip=['Stock', 'Weighted Attractiveness']
    ).properties(
        title='Portfolio Weighted Attractiveness Plot'
    )

    # Display the chart
    st.altair_chart(weighted_chart, use_container_width=True)

    # plotting the 3 year performance of the portfolio
    st.write('Portfolio Performance')
    performance_chart = alt.Chart(my_portfolio.prices.reset_index().melt('Date', var_name='Stock', value_name='Price')).mark_line().encode(
        x='Date',
        y='Price',
        color='Stock'
    ).properties(
        title='Portfolio Performance'
    )
    st.altair_chart(performance_chart, use_container_width=True)

    Optimization and Backtesting
    st.write('Optimize and Backtest Portfolio')
    history_len = st.number_input('History Length (years)', min_value=-1, max_value=10, value=-1)
    num_tests = st.number_input('Number of Tests', min_value=100, max_value=2000, value=2000)
    every_nth = st.number_input('Rebalance Frequency (days)', min_value=1, max_value=365, value=30)

    if st.button('Run Optimization'):
        optimized_portfolio = my_portfolio.optimize_and_backtest(num_tests=num_tests, n_years=history_len, every_nth=every_nth)
        st.write('Optimization Results')
        st.dataframe(optimized_portfolio)

        st.write('Portfolio Allocation Over Time')
        allocation_fig = my_portfolio.plot_allocation()
        st.plotly_chart(allocation_fig, use_container_width=True)

        st.write('Portfolio Performance Over Time')
        optimization_fig = my_portfolio.plot_optimization()
        st.plotly_chart(optimization_fig, use_container_width=True)


if __name__ == '__main__':
    main()
