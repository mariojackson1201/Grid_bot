"""
Grid Trading Strategy Simulation

This module implements a grid trading strategy for forex pairs.
Grid trading places buy and sell orders at regular price intervals (grids),
profiting from price oscillations in ranging markets.
"""

import pandas as pd
from dateutil.parser import parse
import utils
import instrument
import grid_result


class GridLevel:
    """Represents a single grid level with its associated orders"""
    def __init__(self, price, position_size):
        self.price = price
        self.position_size = position_size
        self.position = 0  # Current position: 0 (none), 1 (long), -1 (short)
        self.entry_price = None
        self.trades = []

    def __repr__(self):
        return f"GridLevel(price={self.price:.5f}, position={self.position})"


class GridStrategy:
    """
    Grid Trading Strategy

    Parameters:
    - grid_size: Distance between grid levels (in pips)
    - num_levels: Number of grid levels above and below starting price
    - position_size: Size of position at each grid level
    - start_price: Center price for the grid (if None, uses first price)
    """

    def __init__(self, grid_size_pips, num_levels, position_size=1, start_price=None):
        self.grid_size_pips = grid_size_pips
        self.num_levels = num_levels
        self.position_size = position_size
        self.start_price = start_price
        self.grid_levels = []
        self.trades = []

    def initialize_grid(self, start_price, pip_location):
        """Initialize grid levels around the starting price"""
        self.start_price = start_price
        self.pip_location = pip_location
        self.grid_size = self.grid_size_pips * pip_location

        # Create grid levels above and below start price
        for i in range(-self.num_levels, self.num_levels + 1):
            price = start_price + (i * self.grid_size)
            self.grid_levels.append(GridLevel(price, self.position_size))

        self.grid_levels.sort(key=lambda x: x.price)

    def get_relevant_levels(self, price):
        """Get the grid levels immediately below and above current price"""
        below = None
        above = None

        for level in self.grid_levels:
            if level.price <= price:
                below = level
            elif level.price > price and above is None:
                above = level
                break

        return below, above

    def process_tick(self, timestamp, price, pip_location):
        """Process a price tick and execute grid strategy logic"""
        below_level, above_level = self.get_relevant_levels(price)

        # Check if we crossed a grid level
        if below_level and below_level.position == 0:
            # Price crossed below this level - open long position
            self.open_position(below_level, timestamp, price, 1, pip_location)
            self.open_position(above_level, timestamp, price, -1, pip_location)

        if above_level and above_level.position == 0:
            # Price crossed above this level - open short position
            self.open_position(above_level, timestamp, price, -1, pip_location)
            self.open_position(below_level, timestamp, price, 1, pip_location)

        # Check for profit taking on existing positions
        for level in self.grid_levels:
            if level.position != 0:
                self.check_close_position(level, timestamp, price, pip_location)

    def open_position(self, level, timestamp, current_price, direction, pip_location):
        """Open a position at a grid level"""
        level.position = direction
        level.entry_price = level.price
        level.entry_time = timestamp

    def check_close_position(self, level, timestamp, current_price, pip_location):
        """Check if position should be closed (reached adjacent grid level)"""
        if level.position == 1:  # Long position
            # Close if price reaches next grid level up
            next_level_price = level.price + self.grid_size
            if current_price >= next_level_price:
                self.close_position(level, timestamp, current_price, pip_location)

        elif level.position == -1:  # Short position
            # Close if price reaches next grid level down
            next_level_price = level.price - self.grid_size
            if current_price <= next_level_price:
                self.close_position(level, timestamp, current_price, pip_location)

    def close_position(self, level, timestamp, close_price, pip_location):
        """Close a position and record the trade"""
        if level.position == 0:
            return

        # Calculate profit in pips
        if level.position == 1:  # Long
            profit_pips = (close_price - level.entry_price) / pip_location
        else:  # Short
            profit_pips = (level.entry_price - close_price) / pip_location

        trade = {
            'entry_time': level.entry_time,
            'exit_time': timestamp,
            'entry_price': level.entry_price,
            'exit_price': close_price,
            'direction': level.position,
            'grid_level': level.price,
            'profit_pips': profit_pips,
            'position_size': level.position_size
        }

        self.trades.append(trade)

        # Reset level
        level.position = 0
        level.entry_price = None


def evaluate_pair(i_pair, grid_size_pips, num_levels, price_data):
    """
    Evaluate grid trading strategy on a currency pair

    Parameters:
    - i_pair: Instrument object containing pair information
    - grid_size_pips: Size of each grid level in pips
    - num_levels: Number of grid levels above/below center
    - price_data: DataFrame with price history

    Returns:
    - GridResult object with trade details and statistics
    """

    if len(price_data) == 0:
        return None

    # Initialize strategy with first price
    first_price = price_data.iloc[0]['mid_c']
    strategy = GridStrategy(grid_size_pips, num_levels, position_size=1, start_price=first_price)
    strategy.initialize_grid(first_price, i_pair.pipLocation)

    # Process each price tick
    for idx, row in price_data.iterrows():
        timestamp = parse(row['time']) if isinstance(row['time'], str) else row['time']
        strategy.process_tick(timestamp, row['mid_c'], i_pair.pipLocation)

    # Create trades DataFrame
    if len(strategy.trades) > 0:
        df_trades = pd.DataFrame(strategy.trades)
        df_trades['PAIR'] = i_pair.name
        df_trades['GRID_SIZE'] = grid_size_pips
        df_trades['NUM_LEVELS'] = num_levels
        df_trades['GAIN'] = df_trades['profit_pips'] * df_trades['position_size']

        # Calculate duration
        df_trades['DURATION'] = (df_trades['exit_time'] - df_trades['entry_time']).dt.total_seconds() / 3600
    else:
        df_trades = pd.DataFrame()

    return grid_result.GridResult(
        df_trades=df_trades,
        pairname=i_pair.name,
        params={'grid_size_pips': grid_size_pips, 'num_levels': num_levels}
    )


def get_price_data(pairname, granularity):
    """Load historical price data for a currency pair"""
    df = pd.read_pickle(utils.get_his_data_filename(pairname, granularity))

    non_cols = ['time', 'volume']
    mod_cols = [x for x in df.columns if x not in non_cols]
    df[mod_cols] = df[mod_cols].apply(pd.to_numeric)

    return df[['time', 'mid_c']].copy()


def process_results(results):
    """Process and save results to pickle file"""
    results_list = [r.result_ob() for r in results if r is not None]

    if len(results_list) == 0:
        print("No results to process")
        return

    final_df = pd.DataFrame.from_dict(results_list)
    final_df.to_pickle('grid_test_res.pkl')
    print(f"Results shape: {final_df.shape}, Total trades: {final_df.num_trades.sum()}")
    print(f"Total profit (pips): {final_df.total_gain.sum():.2f}")


def store_trades(results):
    """Store all trades from all results into a single pickle file"""
    all_trade_df_list = [r.df_trades for r in results if r is not None and len(r.df_trades) > 0]

    if len(all_trade_df_list) == 0:
        print("No trades to store")
        return

    all_trade_df = pd.concat(all_trade_df_list)
    all_trade_df.to_pickle("grid_all_trades.pkl")
    print(f"Stored {len(all_trade_df)} trades")


def run():
    """
    Run grid strategy backtesting on multiple currency pairs

    Tests different grid configurations:
    - Grid sizes: 10, 20, 30, 50 pips
    - Number of levels: 5, 10, 15, 20
    """
    currencies = "EUR,USD"
    granularity = "M1"

    # Grid parameters to test
    grid_sizes = [5, 10]  # pips
    num_levels_list = [5, 10, 15, 20]  # number of levels above/below center

    test_pairs = instrument.Instrument.get_pairs_from_string(currencies)

    results = []
    for pairname in test_pairs:
        print(f"Processing {pairname}...")
        i_pair = instrument.Instrument.get_instruments_dict()[pairname]
        price_data = get_price_data(pairname, granularity)

        for grid_size in grid_sizes:
            for num_levels in num_levels_list:
                result = evaluate_pair(i_pair, grid_size, num_levels, price_data)
                if result is not None:
                    results.append(result)
                    print(f"  Grid {grid_size} pips, {num_levels} levels: "
                          f"{result.df_trades.shape[0] if len(result.df_trades) > 0 else 0} trades, "
                          f"{result.df_trades['GAIN'].sum() if len(result.df_trades) > 0 else 0:.2f} pips")

    process_results(results)
    store_trades(results)


if __name__ == "__main__":
    run()
