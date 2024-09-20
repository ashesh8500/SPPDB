import numpy as np
import pandas as pd
import vectorbt as vbt
from numba import njit
from typing import Dict, Tuple
from .exceptions import InsufficientDataError, InvalidWeightError
from .utils import calculate_returns, calculate_volatility, calculate_momentum, clip_weights

# VectorBT settings
vbt.settings['array_wrapper']['freq'] = 'days'
vbt.settings['returns']['year_freq'] = '252 days'
vbt.settings['portfolio']['seed'] = 42
vbt.settings['portfolio']['stats']['incl_unrealized'] = True

class PortfolioOptimizer:

    def __init__(self, initial_weights: Dict[str, float], num_tests: int = 2000, history_len: int = -1, every_nth: int = 30):
        self.initial_weights = initial_weights
        self.symbols = list(initial_weights.keys())
        self.num_tests = num_tests
        self.history_len = history_len
        self.every_nth = every_nth

    def run_simulation(self, price: pd.DataFrame) -> vbt.Portfolio:
        vbao_srb_sharpe = np.full(price.shape[0], np.nan)
        returns = calculate_returns(price)
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
                raise InsufficientDataError("Insufficient data for analysis")
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

        returns = calculate_returns(price_df)
        vol = calculate_volatility(returns, window)
        vol_change = vol.pct_change().iloc[-1]

        momentum = calculate_momentum(returns, min(252, len(returns)))
        momentum_change = momentum.pct_change().iloc[-1]

        attractiveness = self.calculate_attractiveness(vol_change, momentum_change)
        new_weights = self.adjust_weights(current_weights, attractiveness)
        new_weights = clip_weights(new_weights, min_weight=0.05, max_weight=0.4)

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

def plot_allocation(portfolio, symbols):
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
