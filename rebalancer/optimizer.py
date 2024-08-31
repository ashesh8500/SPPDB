import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import pytz
from numba import njit
import pdb
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import base_optimizer
import vectorbt as vbt
from vectorbt.generic.nb import nanmean_nb
from vectorbt.portfolio.nb import order_nb, sort_call_seq_nb
from vectorbt.portfolio.enums import SizeType, Direction


vbt.settings.array_wrapper['freq'] = 'days'
vbt.settings.returns['year_freq'] = '252 days'
vbt.settings.portfolio['seed'] = 42
vbt.settings.portfolio.stats['incl_unrealized'] = True


def plot_allocation(rb_pf, symbols):
    # Plot weights development of the portfolio
    rb_asset_value = rb_pf.asset_value(group_by=False)
    rb_value = rb_pf.value()
    rb_idxs = np.flatnonzero((rb_pf.asset_flow() != 0).any(axis=1))
    rb_dates = rb_pf.wrapper.index[rb_idxs]
    fig = (rb_asset_value.vbt / rb_value).vbt.plot(
        trace_names=symbols,
        trace_kwargs=dict(
            stackgroup='one'
        )
    )
    for rb_date in rb_dates:
        fig.add_shape(
            dict(
                xref='x',
                yref='paper',
                x0=rb_date,
                x1=rb_date,
                y0=0,
                y1=1,
                line_color=fig.layout.template.layout.plot_bgcolor
            )
        )
    return fig

@njit
def pre_sim_func_nb(c, every_nth):
    # Define rebalancing days
    c.segment_mask[:, :] = False
    c.segment_mask[every_nth::every_nth, :] = True
    return ()


@njit
def pre_segment_func_nb(sc, find_weights_nb, history_len, num_tests, srb_sharpe, initial_weights):
    if history_len == -1:
        # Look back at the entire time period
        close = sc.close[:sc.i, sc.from_col:sc.to_col]
    else:
        # Look back at a fixed time period
        pdb.set_trace()
        if sc.i - history_len <= 0:
            return (np.full(sc.group_len, np.nan),)  # insufficient data
        close = sc.close[sc.i - history_len:sc.i, sc.from_col:sc.to_col]


    # Find optimal weights
    best_sharpe_ratio, weights = find_weights_nb(sc, close, initial_weights, num_tests)
    srb_sharpe[sc.i] = best_sharpe_ratio

    # Update valuation price and reorder orders
    size_type = SizeType.TargetPercent
    direction = Direction.LongOnly
    order_value_out = np.empty(sc.group_len, dtype=np.float_)
    for k in range(sc.group_len):
        col = sc.from_col + k
        sc.last_val_price[col] = sc.close[sc.i, col]
    sort_call_seq_nb(sc, weights, size_type, direction, order_value_out)

    return (weights,)

@njit
def order_func_nb(c, weights):
    col_i = c.call_seq_now[c.call_idx]
    return order_nb(
        weights[col_i],
        c.close[c.i, c.col],
        size_type=SizeType.TargetPercent
    )



pdb = pdb.Pdb()

def run_simulation(price, initial_weights: dict, num_tests=2000, history_len=-1, every_nth=30):
    symbols = list(initial_weights.keys())
    # Run simulation with a custom order function
    def vbao_find_weights(sc, price, initial_weights: dict, num_tests):
        # Get current weights
        if sc.i <= every_nth:
            current_weights = pd.Series(initial_weights, index=symbols)
        else:
            # pdb.set_trace()
            current_value = sum(sc.last_position * sc.last_value)
            current_weights = pd.Series((sc.last_position * current_value) / sum(sc.last_position * current_value), index=symbols)

        price = pd.DataFrame(price, columns=symbols)
        window = min(len(price) - 1, 252 if history_len == -1 else history_len)

        # Calculate returns
        returns = price.pct_change().dropna()

        # Calculate volatility
        vol = returns.rolling(window=window, min_periods=window // 2).std() * np.sqrt(252)
        vol_change = vol.pct_change().iloc[-1]

        # Calculate momentum (12-month rolling returns)
        momentum = returns.rolling(window=min(252, len(returns)), min_periods=window // 2).sum()
        momentum_change = momentum.pct_change().iloc[-1]


        # Calculate attractiveness score
        vol_score = 1 / (1 + vol_change)
        momentum_score = 1 + momentum_change
        attractiveness = vol_score * momentum_score #* current_weights

        # Adjust weights based on attractiveness
        weight_change = attractiveness / attractiveness.sum() - current_weights
        new_weights = current_weights + 0.2 * weight_change  # Adjust by 20% of the difference

        # Apply concentration limits
        max_weight = 0.4
        min_weight = 0.05
        new_weights = new_weights.clip(min_weight, max_weight)
        new_weights /= new_weights.sum()  # Renormalize after clipping
        # pdb.set_trace()
        return 1, new_weights.values  # Return a dummy Sharpe ratio and the weights

    vbao_srb_sharpe = np.full(price.shape[0], np.nan)
    returns = price.pct_change()
    history_len = 365*history_len if history_len != -1 else -1
    vbao_srb_pf = vbt.Portfolio.from_order_func(
        price,
        order_func_nb,
        pre_sim_func_nb=pre_sim_func_nb,
        pre_sim_args=(every_nth,),
        pre_segment_func_nb=pre_segment_func_nb.py_func,
        pre_segment_args=(vbao_find_weights, history_len, num_tests, vbao_srb_sharpe, initial_weights),
        cash_sharing=True,
        group_by=True,
        use_numba=False
    )
    return vbao_srb_pf
