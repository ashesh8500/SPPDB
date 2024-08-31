import json
import pandas as pd
import numpy as np
from datetime import datetime
from rebalancer.Portfolio import Portfolio
# with open("portfolio.json", "r") as f:
#     portfolio = json.load(f)

init_weights = np.ones(4) / 4
portfolio = dict(zip(["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"], init_weights))


# Initialize Portfolio
my_portfolio = Portfolio(portfolio)

# Display Portfolio
print('Your Portfolio:')
print(pd.DataFrame(my_portfolio.portfolio, index=pd.Index(['Shares'])))

# Display Portfolio Value
print('Your Portfolio Value:')
print(f'${"{:,.2f}".format(my_portfolio.cash_value)}')

# Display Portfolio Distribution
print('Your Portfolio Distribution:')
print(my_portfolio.dist)

# Calculate and Display Portfolio Attractiveness
# n_years = 3
# date_of_calculation = datetime.now()
# df = my_portfolio.get_attractiveness(n_years=n_years, date_of_calculation=date_of_calculation)
# df.index.name = 'Ticker'
# print('Your Portfolio Attractiveness:')
# print(df)

# # Calculate and Display Portfolio Performance
# performance_date = datetime.now().date()
# performance_df = my_portfolio.get_performance(n_years=[3, 4], date_of_calculation=performance_date)
# print('Your Portfolio Performance:')
# print(performance_df)

# Optimization and Backtesting
history_len = 1
num_tests = 2000
every_nth = 30

print('Running Optimization...')
optimized_portfolio = my_portfolio.optimize_and_backtest(num_tests=num_tests, n_years=history_len, every_nth=every_nth)
print('Optimization Results:')
print(optimized_portfolio)



my_portfolio.pyopt.orders.records_readable.loc[:,['Column', 'Side']].groupby('Column').value_counts().unstack(fill_value=0)



def main():
    # Load portfolio from JSON file
    with open("portfolio.json", "r") as f:
        portfolio_2 = json.load(f)

    # Initialize Portfolio
    my_portfolio = Portfolio(portfolio_2)

    # Display Portfolio
    print('Your Portfolio:')
    print(pd.DataFrame(my_portfolio.portfolio, index=pd.Index(['Shares'])))

    # Display Portfolio Value
    print('Your Portfolio Value:')
    print(f'${"{:,.2f}".format(my_portfolio.cash_value)}')

    # Display Portfolio Distribution
    print('Your Portfolio Distribution:')
    print(my_portfolio.dist)

    # Calculate and Display Portfolio Attractiveness
    # n_years = 3
    # date_of_calculation = datetime.now()
    # df = my_portfolio.get_attractiveness(n_years=n_years, date_of_calculation=date_of_calculation)
    # df.index.name = 'Ticker'
    # print('Your Portfolio Attractiveness:')
    # print(df)

    # # Calculate and Display Portfolio Performance
    # performance_date = datetime.now().date()
    # performance_df = my_portfolio.get_performance(n_years=[3, 4], date_of_calculation=performance_date)
    # print('Your Portfolio Performance:')
    # print(performance_df)

    # Optimization and Backtesting
    history_len = 1
    num_tests = 2000
    every_nth = 30

    print('Running Optimization...')
    optimized_portfolio = my_portfolio.optimize_and_backtest(num_tests=num_tests, n_years=history_len, every_nth=every_nth)
    print('Optimization Results:')
    print(optimized_portfolio)

    # Plotting (Optional, requires graphical backend)
    # allocation_fig = my_portfolio.plot_allocation()
    # optimization_fig = my_portfolio.plot_optimization()

if __name__ == '__main__':
    main()
