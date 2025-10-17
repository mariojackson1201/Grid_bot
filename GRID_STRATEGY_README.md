# Grid Trading Strategy Implementation

This implementation adds a grid trading strategy to the forex backtesting framework.

## Overview

Grid trading is a systematic trading strategy that places buy and sell orders at predetermined price intervals (called "grids"). The strategy profits from price oscillations in ranging markets by:
- Opening long positions when price crosses below grid levels
- Opening short positions when price crosses above grid levels
- Closing positions for profit when price reaches the next grid level

## Files Added

### 1. `grid_sim.py`
Main simulation module that implements the grid trading strategy.

**Key Classes:**
- `GridLevel`: Represents a single price level in the grid
- `GridStrategy`: Implements the core grid trading logic

**Key Functions:**
- `evaluate_pair()`: Runs grid strategy on a currency pair
- `get_price_data()`: Loads historical price data
- `run()`: Executes full backtest across multiple pairs and configurations

### 2. `grid_result.py`
Results processing and analysis module.

**Key Class:**
- `GridResult`: Container for strategy results with analysis methods

**Key Functions:**
- `load_results()`: Load saved results from pickle
- `analyze_results()`: Perform cross-configuration analysis
- `get_best_configurations()`: Find optimal strategy parameters

### 3. `grid_strategy_example.ipynb`
Jupyter notebook with complete examples and visualizations.

**Sections:**
1. Single pair testing
2. Trade analysis (winning/losing trades)
3. Cumulative returns visualization
4. Multi-configuration testing
5. Performance heatmaps
6. Full backtest execution
7. Strategy comparison

## Usage

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
grid_size_pips = 20  # Distance between grid levels
num_levels = 10      # Levels above/below center
result = grid_sim.evaluate_pair(i_pair, grid_size_pips, num_levels, price_data)

# View results
result.print_summary()
```

### Running Full Backtest

```python
# Run on all currency pairs with multiple configurations
grid_sim.run()

# Results saved to:
# - grid_test_res.pkl (summary results)
# - grid_all_trades.pkl (all individual trades)
```

### Analyzing Results

```python
import grid_result

# Load results
results_df = grid_result.load_results('grid_test_res.pkl')
all_trades = grid_result.load_all_trades('grid_all_trades.pkl')

# Analyze
analysis = grid_result.analyze_results(results_df)
best_configs = grid_result.get_best_configurations(results_df, top_n=10)

print(analysis)
print(best_configs)
```

## Strategy Parameters

### Grid Size (pips)
- **Smaller grids (10-20 pips)**: More frequent trades, suitable for low volatility
- **Larger grids (30-50 pips)**: Fewer trades, better for higher volatility
- **Recommendation**: Test multiple sizes to match the pair's typical movement

### Number of Levels
- **Fewer levels (5-10)**: Lower risk, covers smaller price range
- **More levels (15-20)**: Higher exposure, covers wider range
- **Recommendation**: Balance between coverage and risk tolerance

### Default Test Configurations
The `run()` function tests:
- Grid sizes: 10, 20, 30, 50 pips
- Number of levels: 5, 10, 15, 20
- Currency pairs: All combinations from GBP, EUR, USD, CAD, JPY, NZD, CHF

## Performance Metrics

The implementation tracks:
- **Total Gain**: Cumulative profit in pips
- **Number of Trades**: Total positions opened
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss
- **Max Drawdown**: Largest peak-to-trough decline
- **Average Duration**: Mean time trades stay open
- **Long vs Short**: Distribution of trade directions

## When Grid Trading Works Best

### Favorable Conditions
- **Ranging Markets**: Clear support/resistance with sideways movement
- **Mean Reverting Pairs**: Currencies that oscillate around a mean
- **Low Volatility**: Stable, predictable price movement
- **Pairs with patterns**: Historical tendency to range

### Unfavorable Conditions
- **Strong Trends**: Sustained directional movement
- **High Volatility**: Unpredictable price swings
- **Breakout Periods**: Price escaping established ranges
- **News Events**: Sudden market shocks

## Integration with Existing Framework

The grid strategy follows the same patterns as existing strategies:

### Similar to MA Strategy (`ma_sim.py`)
- Uses `evaluate_pair()` function structure
- Returns result objects with standard metrics
- Stores results in pickle files
- Batch processes multiple pairs

### Key Differences
- **State Management**: Tracks multiple open positions simultaneously
- **Entry/Exit Logic**: Price level-based rather than indicator crossovers
- **Position Sizing**: Fixed size at each grid level
- **Profit Taking**: Automatic at next grid level

## Example Workflow

1. **Explore single pair**:
   ```bash
   jupyter notebook grid_strategy_example.ipynb
   ```

2. **Run full backtest**:
   ```bash
   python grid_sim.py
   ```

3. **Analyze results**:
   ```bash
   python grid_result.py
   ```

4. **Compare with other strategies**:
   - Load both `grid_test_res.pkl` and `ma_test_res.pkl`
   - Compare metrics across strategies

## Customization

### Modify Grid Parameters
Edit `grid_sim.py`, line ~270:
```python
grid_sizes = [10, 20, 30, 50]  # Your custom sizes
num_levels_list = [5, 10, 15, 20]  # Your custom levels
```

### Change Currency Pairs
Edit `grid_sim.py`, line ~268:
```python
currencies = "GBP,EUR,USD"  # Your custom currencies
```

### Adjust Granularity
Edit `grid_sim.py`, line ~269:
```python
granularity = "H4"  # H1, H4, D, etc.
```

## Advanced Features

### Multiple Position Management
The strategy can hold multiple positions simultaneously at different grid levels, allowing it to capture multiple price swings.

### Automatic Profit Taking
Positions automatically close when price reaches the adjacent grid level, locking in profits without manual intervention.

### Symmetrical Grid
The grid is centered around the starting price with equal levels above and below, providing balanced exposure to both directions.

## Troubleshooting

### No Results Generated
- Ensure historical data files exist in `his_data/` directory
- Check that `instruments.pkl` is present
- Verify currency pairs are available

### Performance Issues
- Reduce number of grid configurations tested
- Use smaller date ranges
- Test fewer currency pairs initially

### Unexpected Results
- Verify grid parameters are reasonable for the pair's volatility
- Check that price data is clean (no gaps or errors)
- Review the example notebook for proper usage patterns

## Further Development

Potential enhancements:
1. **Dynamic Grid Sizing**: Adjust grid size based on volatility (ATR)
2. **Trend Filter**: Disable grid in strong trending markets
3. **Risk Management**: Add stop loss beyond grid boundaries
4. **Position Limits**: Cap total number of open positions
5. **Partial Profit Taking**: Close portions at multiple targets
6. **Time-based Exits**: Close positions after duration threshold

## References

- Original framework: Moving Average Strategy (`ma_sim.py`)
- Result processing: `ma_result.py`
- Data utilities: `utils.py`, `instrument.py`
