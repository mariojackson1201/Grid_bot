from chatgpt2 import run_backtest
import pandas as pd

# load your candles
candles = pd.read_csv('eurusd_1h.csv', parse_dates=['time'])

results = run_backtest(
    candles,
    grid_start_price=None,      # default: first candle close
    n_levels_each_side=8,       # try different values
    distance_pips=10.0,         # pip spacing between levels
    up_max=None,                # default: max(high) in data
    below_max=None              # default: min(low) in data
)

trades_df = results['trades_df']
print('Net pips:', results['net_pips'])
print(trades_df.head())
