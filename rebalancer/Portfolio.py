from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
import vectorbt as vbt
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from numba import njit

class Portfolio:
    def __init__(self, portfolio: Dict[str, float]):
        self.portfolio = portfolio
        self.symbols: List[str] = list(portfolio.keys())
        self.prices: pd.DataFrame = self.fetch_stock_data(self.symbols)
        self.dist = pd.DataFrame(self.present_distribution(), index=[pd.Timestamp(datetime.now().date())])
        self.cash_value = sum(map(lambda x: x[1] * self.portfolio[x[0]], self.prices.iloc[-1].items()))
        self.ndx_holdings = self.fetch_index_holdings('QQQ')
        self.pyopt = None
        self.optimizer = PortfolioOptimizer(self.portfolio)
        self.weight_change_log = []
    def fetch_stock_data(self, tickers: List[str], period: str = '5y') -> pd.DataFrame:
        """Fetch historical stock data for given tickers."""
        data = yf.download(tickers, period=period)
        return data['Adj Close']

    def fetch_index_holdings(self, index_ticker: str) -> Dict[str, float]:
        """Fetch top holdings of a given index ETF."""
        index = yf.Ticker(index_ticker)
        holdings = index.info.get('holdings', [])
        return {holding['symbol']: holding['holdingPercent'] for holding in holdings}

    def fetch_market_data(self, market_index: str = '^GSPC', period: str = '5y') -> pd.DataFrame:
        """Fetch market index data for comparison."""
        market_data = yf.download(market_index, period=period)
        return market_data['Adj Close']

    def present_distribution(self) -> Dict[str, float]:
        total = sum(self.portfolio.values())
        return {ticker: shares/total for ticker, shares in self.portfolio.items()}

    def get_attractiveness(self, n_years: int = 3, date_of_calculation: datetime = datetime.now()) -> pd.DataFrame:
        return self.calculate_attractiveness(self.prices, n_years, date_of_calculation)

    def get_performance(self, n_years: List[int] = [3, 4], date_of_calculation: datetime = datetime.now()) -> pd.DataFrame:
        return self.calculate_performance(self.prices, self.portfolio, n_years, date_of_calculation)

    def get_orders_with_reasons(self) -> pd.DataFrame:
        if self.pyopt is None:
            raise ValueError("Optimization has not been run yet. Please run optimize_and_backtest first.")

        orders = self.pyopt.orders.records_readable
        weight_changes = pd.DataFrame(self.weight_change_log)

        # Merge orders with weight changes
        combined = pd.merge(orders, weight_changes, left_index=True, right_on='date', how='left')

        # Format the output
        combined['reason'] = combined.apply(lambda row: row['reasons'].get(row['asset'], 'No change') if isinstance(row['reasons'], dict) else 'No change', axis=1)
        combined['weight_change'] = combined.apply(lambda row: row['weight_change'].get(row['asset'], 0) if isinstance(row['weight_change'], dict) else 0, axis=1)

        return combined[['asset', 'size', 'weight_change', 'reason']]
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
        return self.plot_allocation_helper(self.pyopt, self.symbols)

    def get_orders_today(self) -> pd.DataFrame:
        current_prices = self.prices.iloc[-1]
        current_value = sum(shares * current_prices[ticker] for ticker, shares in self.portfolio.items())
        current_weights = {ticker: (shares * current_prices[ticker]) / current_value
                           for ticker, shares in self.portfolio.items()}

        optimizer = PortfolioOptimizer(current_weights)
        new_weights = optimizer.calculate_new_weights(self.prices)

        orders = []
        for ticker in self.portfolio.keys():
            current_weight = current_weights.get(ticker, 0)
            new_weight = new_weights.get(ticker, 0)
            weight_change = new_weight - current_weight

            if abs(weight_change) > 0.01:  # Only suggest orders for significant changes
                current_shares = self.portfolio.get(ticker, 0)
                new_shares = (new_weight * current_value) / current_prices[ticker]
                order_size = new_shares - current_shares

                orders.append({
                    'asset': ticker,
                    'current_weight': current_weight,
                    'new_weight': new_weight,
                    'weight_change': weight_change,
                    'order_size': order_size,
                    'current_price': current_prices[ticker]
                })

        return pd.DataFrame(orders)



    def get_optimized_weights(self) -> pd.DataFrame:
        if self.pyopt is None:
            raise ValueError("Optimization has not been run yet. Please run optimize_and_backtest first.")
        return self.pyopt.asset_value(group_by=False).vbt / self.pyopt.value().vbt

    def get_market_comparison(self, market_index: str = '^GSPC') -> pd.DataFrame:
        market_data = self.fetch_market_data(market_index)
        portfolio_value = (self.prices * pd.Series(self.portfolio)).sum(axis=1)
        comparison = pd.concat([portfolio_value, market_data], axis=1)
        comparison.columns = ['Portfolio', 'Market']
        return comparison

    @staticmethod
    def plot_allocation_helper(portfolio, symbols):
        rb_asset_value = portfolio.asset_value(group_by=False)
        rb_value = portfolio.value()
        rb_idxs = np.flatnonzero((portfolio.asset_flow() != 0).any(axis=1))
        rb_dates = portfolio.wrapper.index[rb_idxs]

        fig = (rb_asset_value.vbt / rb_value).vbt.plot(
            trace_names=symbols,
            trace_kwargs=dict(stackgroup='one')
        )

        for rb_date in rb_dates:
            fig.add_shape(dict(
                xref='x', yref='paper',
                x0=rb_date, x1=rb_date,
                y0=0, y1=1,
                line_color=fig.layout.template.layout.plot_bgcolor
            ))

        return fig

    @staticmethod
    def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """Calculate returns from price data."""
        return prices.pct_change(fill_method=None)

    @staticmethod
    def calculate_volatility(returns: pd.DataFrame, window: int) -> pd.DataFrame:
        """Calculate rolling volatility of returns."""
        return returns.rolling(window=window, min_periods=window // 2).std() * np.sqrt(252)

    @staticmethod
    def calculate_momentum(returns: pd.DataFrame, window: int) -> pd.DataFrame:
        """Calculate rolling momentum of returns."""
        return returns.rolling(window=window, min_periods=window // 2).sum()

    @staticmethod
    def clip_weights(weights: pd.Series, min_weight: float, max_weight: float) -> pd.Series:
        """Clip weights to specified range and renormalize."""
        clipped_weights = weights.clip(min_weight, max_weight)
        return clipped_weights / clipped_weights.sum()

    @staticmethod
    def plot_attractiveness(attractiveness_df: pd.DataFrame):
        sns.barplot(x=attractiveness_df.index, y=attractiveness_df['Attractiveness'], hue=attractiveness_df['Attractiveness'] > 0)
        plt.title('Attractiveness of Stocks')
        plt.show()

    @staticmethod
    def plot_qqq_performance(fig: go.Figure, qqq_data: pd.DataFrame):
        fig.add_trace(go.Scatter(x=qqq_data.index, y=qqq_data['Close'], mode='lines', name='QQQ Performance'))
        return fig

    @staticmethod
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

    @staticmethod
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

class PortfolioOptimizer:
    def __init__(self, initial_weights: Dict[str, float], num_tests: int = 2000, history_len: int = -1, every_nth: int = 30):
        self.initial_weights = initial_weights
        self.symbols = list(initial_weights.keys())
        self.num_tests = num_tests
        self.history_len = history_len
        self.every_nth = every_nth

    def calculate_new_weights(self, prices: pd.DataFrame) -> Dict[str, float]:
        price_df = prices[self.symbols].iloc[-252:]  # Use last year of data

        returns = Portfolio.calculate_returns(price_df)
        vol = Portfolio.calculate_volatility(returns, 252)
        vol_change = vol.pct_change().iloc[-1]

        momentum = Portfolio.calculate_momentum(returns, 252)
        momentum_change = momentum.pct_change().iloc[-1]

        attractiveness = self.calculate_attractiveness(vol_change, momentum_change)
        new_weights = self.adjust_weights(pd.Series(self.initial_weights), attractiveness)
        new_weights = Portfolio.clip_weights(new_weights, min_weight=0.05, max_weight=0.4)

        return new_weights.to_dict()

    def run_simulation(self, price: pd.DataFrame) -> vbt.Portfolio:
        vbao_srb_sharpe = np.full(price.shape[0], np.nan)
        returns = Portfolio.calculate_returns(price)
        history_len = 365 * self.history_len if self.history_len != -1 else -1

        vbao_srb_pf = vbt.Portfolio.from_order_func(
            price,
            self.order_func_nb,
            pre_sim_func_nb=self.pre_sim_func_nb,
            pre_sim_args=(self.every_nth,),
            pre_segment_func_nb=self.pre_segment_func_nb,
            pre_segment_args=(self.vbao_find_weights, history_len, self.num_tests, vbao_srb_sharpe, self.initial_weights),
            cash_sharing=True,
            group_by=True,
            use_numba=False
        )
        return vbao_srb_pf

    @staticmethod
    @njit
    def pre_sim_func_nb(c, every_nth):
        c.segment_mask[:, :] = False
        c.segment_mask[every_nth::every_nth, :] = True
        return ()

    @staticmethod
    def pre_segment_func_nb(sc, find_weights_nb, history_len, num_tests, srb_sharpe, initial_weights):
        if history_len == -1:
            close = sc.close[:sc.i, sc.from_col:sc.to_col]
        else:
            if sc.i - history_len <= 0:
                raise ValueError("Insufficient data for analysis")
            close = sc.close[sc.i - history_len:sc.i, sc.from_col:sc.to_col]

        best_sharpe_ratio, weights = find_weights_nb(sc, close, initial_weights, num_tests)
        srb_sharpe[sc.i] = best_sharpe_ratio

        for k in range(sc.group_len):
            col = sc.from_col + k
            sc.last_val_price[col] = sc.close[sc.i, col]
        vbt.portfolio.nb.sort_call_seq_nb(sc, weights, vbt.portfolio.enums.SizeType.TargetPercent, vbt.portfolio.enums.Direction.LongOnly, np.empty(sc.group_len, dtype=np.float_))

        return (weights,)

    @staticmethod
    @njit
    def order_func_nb(c, weights):
        col_i = c.call_seq_now[c.call_idx]
        return vbt.portfolio.nb.order_nb(
            weights[col_i],
            c.close[c.i, c.col],
            size_type=vbt.portfolio.enums.SizeType.TargetPercent
        )

    def vbao_find_weights(self, sc, price, initial_weights: Dict[str, float], num_tests) -> Tuple[float, np.ndarray]:
        current_weights = self.get_current_weights(sc, initial_weights)
        price_df = pd.DataFrame(price, columns=list(self.symbols))
        window = min(len(price_df) - 1, 252 if self.history_len == -1 else self.history_len)

        returns = Portfolio.calculate_returns(price_df)
        vol = Portfolio.calculate_volatility(returns, window)
        vol_change = vol.pct_change().iloc[-1]

        momentum = Portfolio.calculate_momentum(returns, min(252, len(returns)))
        momentum_change = momentum.pct_change().iloc[-1]

        attractiveness = self.calculate_attractiveness(vol_change, momentum_change)
        new_weights = self.adjust_weights(current_weights, attractiveness)
        new_weights = Portfolio.clip_weights(new_weights, min_weight=0.05, max_weight=0.4)

        return 1.0, new_weights.to_numpy()

    def get_current_weights(self, sc, initial_weights: Dict[str, float]) -> pd.Series:
        if sc.i <= self.every_nth:
            return pd.Series(initial_weights)
        current_value = sum(sc.last_position * sc.last_value)
        return pd.Series(sc.last_position * sc.last_value / current_value, index=initial_weights.keys())

    @staticmethod
    def calculate_attractiveness(vol_change: pd.Series, momentum_change: pd.Series) -> pd.Series:
        vol_score = 1 / (1 + vol_change)
        momentum_score = 1 + momentum_change
        return pd.Series(vol_score.values * momentum_score.values, index=vol_change.index)

    @staticmethod
    def adjust_weights(current_weights: pd.Series, attractiveness: pd.Series) -> pd.Series:
        weight_change = attractiveness / attractiveness.sum() - current_weights
        return current_weights + 0.2 * weight_change
