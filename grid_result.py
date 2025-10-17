"""
Grid Trading Results Processing

This module handles the processing and analysis of grid trading strategy results.
"""

import pandas as pd


class GridResult:
    """
    Container for grid trading strategy results

    Attributes:
    - df_trades: DataFrame containing all trades executed
    - pairname: Currency pair name
    - params: Dictionary of strategy parameters (grid_size_pips, num_levels)
    """

    def __init__(self, df_trades, pairname, params):
        self.pairname = pairname
        self.df_trades = df_trades
        self.params = params

    def result_ob(self):
        """
        Create a summary object of the strategy results

        Returns:
        - Dictionary with key performance metrics
        """
        if len(self.df_trades) == 0:
            d = {
                'pair': self.pairname,
                'num_trades': 0,
                'total_gain': 0.0,
                'mean_gain': 0.0,
                'min_gain': 0.0,
                'max_gain': 0.0,
                'win_rate': 0.0,
                'avg_duration': 0.0,
                'max_drawdown': 0.0,
                'profit_factor': 0.0,
                'long_trades': 0,
                'short_trades': 0
            }
        else:
            # Calculate winning trades
            winning_trades = self.df_trades[self.df_trades.GAIN > 0]
            losing_trades = self.df_trades[self.df_trades.GAIN < 0]

            win_rate = len(winning_trades) / len(self.df_trades) * 100 if len(self.df_trades) > 0 else 0

            # Calculate profit factor (gross profit / gross loss)
            gross_profit = winning_trades.GAIN.sum() if len(winning_trades) > 0 else 0
            gross_loss = abs(losing_trades.GAIN.sum()) if len(losing_trades) > 0 else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

            # Calculate drawdown
            cumulative_gains = self.df_trades.GAIN.cumsum()
            running_max = cumulative_gains.expanding().max()
            drawdown = running_max - cumulative_gains
            max_drawdown = drawdown.max() if len(drawdown) > 0 else 0

            # Count long vs short trades
            long_trades = len(self.df_trades[self.df_trades.direction == 1])
            short_trades = len(self.df_trades[self.df_trades.direction == -1])

            d = {
                'pair': self.pairname,
                'num_trades': self.df_trades.shape[0],
                'total_gain': self.df_trades.GAIN.sum(),
                'mean_gain': self.df_trades.GAIN.mean(),
                'min_gain': self.df_trades.GAIN.min(),
                'max_gain': self.df_trades.GAIN.max(),
                'win_rate': win_rate,
                'avg_duration': self.df_trades.DURATION.mean() if 'DURATION' in self.df_trades.columns else 0,
                'max_drawdown': max_drawdown,
                'profit_factor': profit_factor,
                'long_trades': long_trades,
                'short_trades': short_trades
            }

        # Add strategy parameters to result
        for k, v in self.params.items():
            d[k] = v

        return d

    def print_summary(self):
        """Print a formatted summary of the results"""
        result = self.result_ob()

        print(f"\n{'='*60}")
        print(f"Grid Trading Strategy Results: {self.pairname}")
        print(f"{'='*60}")
        print(f"Grid Size: {result['grid_size_pips']} pips")
        print(f"Number of Levels: {result['num_levels']}")
        print(f"\nPerformance Metrics:")
        print(f"  Total Trades: {result['num_trades']}")
        print(f"  Total Gain: {result['total_gain']:.2f} pips")
        print(f"  Average Gain: {result['mean_gain']:.2f} pips")
        print(f"  Win Rate: {result['win_rate']:.2f}%")
        print(f"  Profit Factor: {result['profit_factor']:.2f}")
        print(f"  Max Drawdown: {result['max_drawdown']:.2f} pips")
        print(f"\nTrade Distribution:")
        print(f"  Long Trades: {result['long_trades']}")
        print(f"  Short Trades: {result['short_trades']}")
        print(f"  Avg Duration: {result['avg_duration']:.2f} hours")
        print(f"{'='*60}\n")

    def get_trades_df(self):
        """Return the trades DataFrame"""
        return self.df_trades

    def get_winning_trades(self):
        """Return only winning trades"""
        if len(self.df_trades) == 0:
            return pd.DataFrame()
        return self.df_trades[self.df_trades.GAIN > 0].copy()

    def get_losing_trades(self):
        """Return only losing trades"""
        if len(self.df_trades) == 0:
            return pd.DataFrame()
        return self.df_trades[self.df_trades.GAIN < 0].copy()

    def get_cumulative_returns(self):
        """Calculate cumulative returns over time"""
        if len(self.df_trades) == 0:
            return pd.Series()
        return self.df_trades.GAIN.cumsum()


def load_results(filename='grid_test_res.pkl'):
    """
    Load grid trading results from pickle file

    Parameters:
    - filename: Path to pickle file

    Returns:
    - DataFrame with results
    """
    return pd.read_pickle(filename)


def load_all_trades(filename='grid_all_trades.pkl'):
    """
    Load all trades from pickle file

    Parameters:
    - filename: Path to pickle file

    Returns:
    - DataFrame with all trades
    """
    return pd.read_pickle(filename)


def analyze_results(results_df):
    """
    Analyze results across all pairs and configurations

    Parameters:
    - results_df: DataFrame with results from multiple runs

    Returns:
    - Dictionary with analysis insights
    """
    if len(results_df) == 0:
        return {}

    analysis = {
        'total_trades': results_df.num_trades.sum(),
        'total_profit': results_df.total_gain.sum(),
        'avg_profit_per_config': results_df.total_gain.mean(),
        'best_pair': results_df.loc[results_df.total_gain.idxmax()]['pair'],
        'best_profit': results_df.total_gain.max(),
        'worst_pair': results_df.loc[results_df.total_gain.idxmin()]['pair'],
        'worst_profit': results_df.total_gain.min(),
        'avg_win_rate': results_df.win_rate.mean(),
        'avg_profit_factor': results_df.profit_factor.mean()
    }

    return analysis


def get_best_configurations(results_df, top_n=10):
    """
    Get the best performing configurations

    Parameters:
    - results_df: DataFrame with results
    - top_n: Number of top configurations to return

    Returns:
    - DataFrame sorted by total_gain
    """
    if len(results_df) == 0:
        return pd.DataFrame()

    return results_df.nlargest(top_n, 'total_gain')


if __name__ == "__main__":
    # Example usage
    try:
        results = load_results()
        print(f"Loaded {len(results)} results")

        analysis = analyze_results(results)
        print("\nOverall Analysis:")
        for key, value in analysis.items():
            print(f"  {key}: {value}")

        print("\nTop 5 Configurations:")
        print(get_best_configurations(results, top_n=5))

    except FileNotFoundError:
        print("No results file found. Run grid_sim.py first to generate results.")
