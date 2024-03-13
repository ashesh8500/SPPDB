import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yfinance
import pandas as pd
from datetime import date, datetime, timedelta
import altair as alt
from Portfolio import Portfolio

def main():
    st.title('Protoype')
    st.write('Dashboard overview of your portfolio and its attractiveness')
    # pf = pd.read_csv('nasdaq weights adjustment.csv', index_col='ticker')
    # pf.drop('Unnamed: 0', axis=1, inplace=True)
    # pf.loc[:,['nd_weights', 'portfolio_weight']] =pf.loc[:,['nd_weights', 'portfolio_weight']].apply(lambda x: x.str.replace('%', '').astype(float)/100)
    # pf = pf.loc[:,['company_name','nd_weights', 'portfolio_weight']].dropna()
    portfolio = {
        'AAPL': 10,
        'MSFT': 20,
        'GOOGL': 5,
        'AMZN': 3,
        'META': 7,
        'TSLA': 5,
        'NFLX': 10
    }
    portfolio_2 = {
        "MSFT": 0.165567903559993,
        "AAPL": 0.154077980787342,
        "NVDA": 0.1047,
        "AMZN": 0.0978,
        "META": 0.0934,
        "AVGO": 0.085,
        "TSLA": 0.053099999999999994,
        "COST": 0.0467,
        "GOOGL": 0.0441,
        "GOOG": 0.0429,
        "AMD": 0.0403,
        "NFLX": 0.0369,
        "ADBE": 0.0354,
    }

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
    # st.write('Your Portfolio Attractiveness Plot')
    # Calculate color based on attractiveness
    df['Color'] = df['Attractiveness'].apply(lambda x: 'green' if x > 0 else 'red')
    # Define color scale for gradientation
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
    chart = alt.Chart(df).mark_bar().encode(
        x='Stock',
        y='Attractiveness',
        color=alt.Color('Attractiveness', scale=color_scale, sort=alt.SortField(field='Attractiveness', order='ascending')), # type: ignore
        tooltip=['Stock', 'Attractiveness']
    ).properties(
        title='Portfolio Attractiveness Plot'
    )

    # Display the chart
    st.altair_chart(chart, use_container_width=True)

    # plotting the 3 year performace of the portfolio
    st.write('Portfolio Performance')
    performance_chart = alt.Chart(my_portfolio.prices.reset_index().melt('Date', var_name='Stock', value_name='Price')).mark_line().encode(
        x='Date',
        y='Price',
        color='Stock'
    ).properties(
        title='Portfolio Performance'
    )
    st.altair_chart(performance_chart, use_container_width=True)

if __name__ == '__main__':
    main()
