# Forex Grid Trading Strategy - Backtesting Framework

A comprehensive grid trading strategy implementation for forex backtesting, built as an extension to a forex backtesting framework.

## Overview

This repository implements a **grid trading strategy** for forex pairs using historical data from the OANDA API. Grid trading is a systematic approach that places buy and sell orders at predetermined price intervals, profiting from price oscillations in ranging markets.

## Features

- **Flexible Grid Configuration**: Customize grid size (pips) and number of levels
- **Multi-Pair Backtesting**: Test strategies across multiple currency pairs simultaneously
- **Comprehensive Analysis**: Win rate, profit factor, drawdown, and more
- **Visualization Tools**: Jupyter notebook with interactive charts and heatmaps
- **Results Export**: Pickle files for further analysis

## Project Structure

```
├── grid_sim.py                    # Main grid strategy simulation
├── grid_result.py                 # Results processing and analysis
├── grid_strategy_example.ipynb    # Jupyter notebook with examples
├── GRID_STRATEGY_README.md        # Detailed documentation
├── ma_sim.py                      # Moving average strategy
├── inside_bar_sim.py              # Inside bar strategy
├── instrument.py                  # Instrument/pair management
├── utils.py                       # Utility functions
├── defs.py                        # API configuration
└── oanda_api.py                   # OANDA API integration
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/forex-grid-trading.git
cd forex-grid-trading

# Install dependencies
pip install pandas matplotlib jupyter
```

### Basic Usage

```python
import grid_sim
import grid_result
import instrument

# Load currency pair
pairname = "EUR_USD"
i_pair = instrument.Instrument.get_instrument_by_name(pairname)

# Load price data
price_data = grid_sim.get_price_data(pairname, "H1")

# Run grid strategy
result = grid_sim.evaluate_pair(i_pair, grid_size_pips=20, num_levels=10, price_data=price_data)

# View results
result.print_summary()
```

### Run Full Backtest

```bash
python grid_sim.py
```

This will test multiple grid configurations across all available currency pairs and save results to:
- `grid_test_res.pkl` - Summary results
- `grid_all_trades.pkl` - All individual trades

### Interactive Analysis

```bash
jupyter notebook grid_strategy_example.ipynb
```

## Grid Trading Strategy

### How It Works

1. **Initialize Grid**: Create price levels at regular intervals around a center price
2. **Enter Positions**:
   - Open long when price crosses below a grid level
   - Open short when price crosses above a grid level
3. **Exit Positions**: Close for profit when price reaches the next grid level

### Parameters

- **Grid Size**: Distance between grid levels (in pips)
  - Smaller grids (10-20 pips): More frequent trades
  - Larger grids (30-50 pips): Fewer, larger trades

- **Number of Levels**: Grid levels above/below center
  - Fewer levels (5-10): Lower risk, smaller range
  - More levels (15-20): Higher exposure, wider range

### Best Use Cases

Grid trading works best in:
- Ranging/sideways markets
- Low to medium volatility environments
- Mean-reverting currency pairs
- Markets with clear support/resistance

## Performance Metrics

The framework tracks:
- Total profit/loss (pips)
- Number of trades
- Win rate (%)
- Profit factor (gross profit / gross loss)
- Maximum drawdown
- Average trade duration
- Long vs short trade distribution

## Example Results

```
Grid Trading Strategy Results: EUR_USD
Grid Size: 20 pips
Number of Levels: 10

Performance Metrics:
  Total Trades: 145
  Total Gain: 342.50 pips
  Average Gain: 2.36 pips
  Win Rate: 58.62%
  Profit Factor: 1.45
  Max Drawdown: 87.30 pips
```

## Data Requirements

You'll need historical price data in pickle format:
- Stored in `his_data/` directory
- Format: `{PAIR}_{GRANULARITY}.pkl`
- Example: `EUR_USD_H1.pkl`

Also required:
- `instruments.pkl` - Currency pair metadata

## Comparison with Other Strategies

This framework includes multiple strategies for comparison:
- **Moving Average Strategy** (`ma_sim.py`) - Trend following
- **Inside Bar Strategy** (`inside_bar_sim.py`) - Breakout trading
- **Grid Strategy** (`grid_sim.py`) - Range trading

## Documentation

For detailed documentation, see:
- [GRID_STRATEGY_README.md](GRID_STRATEGY_README.md) - Complete grid strategy guide
- [grid_strategy_example.ipynb](grid_strategy_example.ipynb) - Interactive examples

## Advanced Features

- Multiple simultaneous positions at different grid levels
- Automatic profit taking at adjacent levels
- Symmetrical grid structure (balanced long/short exposure)
- Batch processing of multiple configurations
- Cross-strategy performance comparison

## Configuration

### Default Test Settings

```python
# Currency pairs
currencies = "GBP,EUR,USD,CAD,JPY,NZD,CHF"

# Grid parameters
grid_sizes = [10, 20, 30, 50]  # pips
num_levels_list = [5, 10, 15, 20]

# Timeframe
granularity = "H1"  # Hourly bars
```

## Customization

Edit `grid_sim.py` to customize:
- Grid sizes to test
- Number of levels
- Currency pairs
- Data granularity
- Position sizing

## Contributing

This is an educational project for backtesting trading strategies. Contributions are welcome!

## Disclaimer

This software is for educational and research purposes only. It is not financial advice. Trading forex carries significant risk. Past performance does not guarantee future results.

## License

This project builds upon the original backtesting framework by [jcjstuga2](https://github.com/jcjstuga2/FinalYearProject-Backtesting).

## Acknowledgments

- Original framework: FinalYearProject-Backtesting
- Data provider: OANDA API
- Built with: Python, Pandas, Matplotlib, Jupyter

## Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: Ensure you have proper OANDA API credentials configured in `defs.py` before fetching new historical data.