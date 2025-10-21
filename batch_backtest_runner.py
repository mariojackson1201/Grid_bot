
"""
Batch Backtest Runner for Grid Strategy
---------------------------------------
Features:
- Runs multiple parameterized tests (instrument, date range, granularity, levels, distance pips).
- Supports two data sources per test:
    1) Live OANDA fetch via your existing module (/mnt/data/chatgpt2.py)
    2) Local CSV of candles (columns: time, open, high, low, close, complete?, volume?) [at minimum: time, high, low, close]
- Summaries: total trades, wins, losses, win rate, total pips, avg pips/trade, max drawdown (pips), start/end dates.
- Artifacts per test: trades CSV, equity-curve PNG.
- Comparison plots: total pips and win rate across tests.
- Master summary CSV of all tests.

Requirements:
- matplotlib, pandas, numpy
- Your existing backtester code at /mnt/data/chatgpt2.py with:
    - fetch_oanda_candles(token, instrument, granularity, start=..., end=...)
    - GridBacktester(candles, levels, distance_pips, up_max=None, bellow_max=None).run(), results_df(), summary()

Usage:
    python batch_backtest_runner.py --token YOUR_OANDA_TOKEN
    # or, to run from local CSVs only (no token needed):
    python batch_backtest_runner.py --from-csv

Edit the TESTS list below or pass a JSON file with --tests path/to/tests.json.
"""

import argparse
import importlib.util
import os
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------- Helpers: dynamic import ---------------------------
def load_backtester_module(module_path: str):
    spec = importlib.util.spec_from_file_location("user_backtester", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module at {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# ----------------------- Equity & Drawdown --------------------------------
def build_equity_curve_pips(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by close_time with columns:
      - pips: trade pips
      - equity_pips: cumulative pips
      - peak: running max
      - drawdown: equity - peak (<= 0)
    """
    if trades_df.empty:
        return pd.DataFrame(columns=["close_time","pips","equity_pips","peak","drawdown"]).set_index("close_time")

    df = trades_df.copy()
    df = df.sort_values("close_time")
    df["pips"] = df["pips"].astype(float)
    df["equity_pips"] = df["pips"].cumsum()
    df["peak"] = df["equity_pips"].cummax()
    df["drawdown"] = df["equity_pips"] - df["peak"]
    df = df[["close_time", "pips", "equity_pips", "peak", "drawdown"]]
    df = df.set_index("close_time")
    return df

def max_drawdown_pips(equity_df: pd.DataFrame) -> float:
    if equity_df.empty:
        return 0.0
    return float(equity_df["drawdown"].min())  # will be <= 0

# ----------------------- Test definition ----------------------------------
@dataclass
class TestConfig:
    name: str
    instrument: Optional[str] = None   # e.g., "EUR_USD"
    granularity: Optional[str] = None  # e.g., "M1", "H1"
    start: Optional[str] = None        # RFC3339 "YYYY-MM-DDTHH:MM:SSZ"
    end: Optional[str] = None
    levels: int = 10
    distance_pips: float = 10.0
    up_max: Optional[float] = None
    bellow_max: Optional[float] = None
    candles_csv: Optional[str] = None  # if provided, use this instead of fetching

# ----------------------- Default test grid --------------------------------
TESTS: List[TestConfig] = [
    # Edit these or provide a JSON via --tests
    TestConfig(
        name="EURUSD_10pips",
        instrument="EUR_USD",
        granularity="M1",
        start="2021-05-24T00:00:00Z",
        end="2022-09-27T00:00:00Z",
        levels=2000,
        distance_pips=10.0,
    ),
    TestConfig(
        name="EURUSD_longer",
        instrument="EUR_USD",
        granularity="M1",
        start="2021-05-24T00:00:00Z",
        end="2023-02-02T00:00:00Z",
        levels=2000,
        distance_pips=10.0,
    ),
    # Example using a local CSV instead of fetching:
    # TestConfig(
    #     name="FromCSV_Demo",
    #     candles_csv="/path/to/eurusd_m1_sample.csv",
    #     levels=12,
    #     distance_pips=8.0,
    # ),
]

# ----------------------- Core batch runner --------------------------------
def run_single_test(backtester_mod, cfg: TestConfig, token: Optional[str], outdir: str) -> Dict:
    # Obtain candles
    if cfg.candles_csv:
        candles = pd.read_csv(cfg.candles_csv)
        # Ensure necessary columns exist and time type is datetime
        required_any = {"time", "close"}
        if not required_any.issubset(set(candles.columns)):
            raise ValueError(f"CSV {cfg.candles_csv} must include at least columns: {required_any}")
        candles["time"] = pd.to_datetime(candles["time"], utc=True, errors="coerce")
        # if high/low missing, approximate using close; this will under-estimate fills
        for col in ["high","low","open","complete","volume"]:
            if col not in candles.columns:
                if col in ["high","low","open"]:
                    candles[col] = candles["close"]
                elif col == "complete":
                    candles[col] = True
                elif col == "volume":
                    candles[col] = 0
        candles = candles.sort_values("time").reset_index(drop=True)
        src = f"csv:{os.path.basename(cfg.candles_csv)}"
    else:
        if not token:
            raise RuntimeError(f"Test '{cfg.name}' requires --token or candles_csv.")
        candles = backtester_mod.fetch_oanda_candles(
            token=token,
            instrument=cfg.instrument,
            granularity=cfg.granularity,
            start=cfg.start,
            end=cfg.end,
            show_progress=True,
        )
        src = f"oanda:{cfg.instrument}:{cfg.granularity}"

    # Run backtest
    bt = backtester_mod.GridBacktester(
        candles=candles,
        levels=cfg.levels,
        distance_pips=cfg.distance_pips,
        up_max=cfg.up_max,
        bellow_max=cfg.bellow_max,
    )
    bt.run()
    trades = bt.results_df()
    summ = bt.summary()

    # Equity & drawdown
    equity = build_equity_curve_pips(trades)
    mdd = max_drawdown_pips(equity)

    # Persist artifacts
    os.makedirs(outdir, exist_ok=True)
    trades_path = os.path.join(outdir, f"{cfg.name}_trades.csv")
    equity_path = os.path.join(outdir, f"{cfg.name}_equity.png")

    trades.to_csv(trades_path, index=False)

    # Plot equity curve
    plt.figure(figsize=(10, 5))
    if not equity.empty:
        equity["equity_pips"].plot(title=f"Equity (pips) — {cfg.name}")
        plt.xlabel("Time")
        plt.ylabel("Cumulative Pips")
    else:
        plt.title(f"Equity (pips) — {cfg.name} (no trades)")
    plt.tight_layout()
    plt.savefig(equity_path, dpi=150)
    plt.close()

    # Return enriched summary
    enriched = {
        "name": cfg.name,
        "source": src,
        "instrument": cfg.instrument or "csv",
        "granularity": cfg.granularity or "csv",
        "start": cfg.start or (str(candles.time.min()) if "time" in candles else ""),
        "end": cfg.end or (str(candles.time.max()) if "time" in candles else ""),
        "levels": cfg.levels,
        "distance_pips": cfg.distance_pips,
        "total_trades": summ.get("total_trades", 0),
        "wins": summ.get("wins", 0),
        "losses": summ.get("losses", 0),
        "win_rate": summ.get("win_rate", 0.0),
        "total_pips": summ.get("total_pips", 0.0),
        "avg_pips_per_trade": round((summ.get("total_pips", 0.0) / max(1, summ.get("total_trades", 0))), 3),
        "max_drawdown_pips": mdd,  # typically <= 0
        "trades_csv": trades_path,
        "equity_png": equity_path,
    }
    return enriched

def run_batch(backtester_mod, tests: List[TestConfig], token: Optional[str], outdir: str) -> pd.DataFrame:
    rows = []
    for cfg in tests:
        print(f"\n=== Running: {cfg.name} ===")
        rows.append(run_single_test(backtester_mod, cfg, token, outdir))
    df = pd.DataFrame(rows)
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(os.path.join(outdir, "summary.csv"), index=False)
    return df

# ----------------------- Comparison charts --------------------------------
def plot_bar(df: pd.DataFrame, column: str, out_png: str, title: str, ylabel: str):
    plt.figure(figsize=(10,5))
    x = np.arange(len(df))
    vals = df[column].values
    labels = df["name"].values
    plt.bar(x, vals)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

# ----------------------- CLI ---------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--token", type=str, default="28ea3ce61c08a436f559a2aa4f49cab2-c6732cd18d2ec0bbe4f8579adb722602", help="OANDA API token (needed unless you only use --from-csv tests).")
    p.add_argument("--module", type=str, default="./chatgpt2.py", help="Path to your existing backtester module.")
    p.add_argument("--tests", type=str, default=None, help="Path to a JSON file with a list of tests (same keys as TestConfig).")
    p.add_argument("--outdir", type=str, default="./batch_results", help="Output directory for results.")
    p.add_argument("--from-csv", action="store_true", help="Ignore --token and only run tests that specify candles_csv.")
    return p.parse_args()

def main():
    args = parse_args()

    # Load tests
    if args.tests:
        with open(args.tests, "r") as f:
            test_dicts = pd.read_json(f).to_dict(orient="records")
        tests = [TestConfig(**d) for d in test_dicts]
    else:
        tests = TESTS

    if args.from_csv:
        args.token = None  # force CSV-only

    # Import backtester/source module
    backtester_mod = load_backtester_module(args.module)

    # Run batch
    summary_df = run_batch(backtester_mod, tests, token=args.token, outdir=args.outdir)

    # Comparison plots
    if not summary_df.empty:
        totals_png = os.path.join(args.outdir, "comparison_total_pips.png")
        winrate_png = os.path.join(args.outdir, "comparison_win_rate.png")
        dd_png = os.path.join(args.outdir, "comparison_max_drawdown.png")

        plot_bar(summary_df, "total_pips", totals_png, "Total Pips by Test", "Total Pips")
        plot_bar(summary_df, "win_rate", winrate_png, "Win Rate (%) by Test", "Win Rate (%)")
        plot_bar(summary_df, "max_drawdown_pips", dd_png, "Max Drawdown (pips) by Test", "Max Drawdown (pips)")

        print(f"\nArtifacts written to: {args.outdir}")
        print(f" - Summary CSV: {os.path.join(args.outdir, 'summary.csv')}")
        print(f" - Total pips chart: {totals_png}")
        print(f" - Win rate chart: {winrate_png}")
        print(f" - Max drawdown chart: {dd_png}")
    else:
        print("No results produced. Check your test configs.")

if __name__ == "__main__":
    main()
