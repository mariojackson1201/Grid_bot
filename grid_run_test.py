import grid_sim
import grid_result
import instrument

# Load currency pair
pairname = "EUR_USD"
i_pair = instrument.Instrument.get_instrument_by_name(pairname)

# Load price data
price_data = grid_sim.get_price_data(pairname, "H1")

# Run grid strategy
grid_size_pips = 10  # Distance between grid levels
num_levels = 100      # Levels above/below center
result = grid_sim.evaluate_pair(i_pair, grid_size_pips, num_levels, price_data)

# View results
#result.print_summary()


grid_sim.run()

# Load results
results_df = grid_result.load_results('grid_test_res.pkl')
all_trades = grid_result.load_all_trades('grid_all_trades.pkl')

# Analyze
analysis = grid_result.analyze_results(results_df)
best_configs = grid_result.get_best_configurations(results_df, top_n=10)

print(analysis)
print(best_configs)