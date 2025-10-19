"""
hedging_grid_backtester_dynamic.py
Dynamic hedging grid backtester — reads local CSV file for EUR/USD.
"""

import pandas as pd
from typing import List, Dict, Optional

PIP = 0.0001  # EUR/USD pip size


def load_candles_csv(path: str) -> pd.DataFrame:
    """Load CSV containing columns: time, open, high, low, close"""
    df = pd.read_csv(path, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


class GridBacktester:
    def __init__(
        self,
        candles: pd.DataFrame,
        start_price: Optional[float] = None,
        levels: int = 100,
        distance_pips: float = 10.0,
        up_max: Optional[float] = None,
        bellow_max: Optional[float] = None,
    ):
        self.candles = candles.copy().reset_index(drop=True)
        self.start_price = start_price if start_price else float(self.candles.loc[0, "close"])
        self.levels = int(levels)
        self.distance = float(distance_pips) * PIP
        self.up_max = up_max
        self.bellow_max = bellow_max
        self.level_prices = self._build_levels()
        self.pending_orders = self._init_pending_orders()
        self.trades = []

    def _build_levels(self) -> List[float]:
        prices = []
        for i in range(-self.levels, self.levels + 1):
            prices.append(round(self.start_price + i * self.distance, 6))
        return sorted(list(set(prices)))

    def _init_pending_orders(self) -> List[Dict]:
        orders = []
        for idx, lp in enumerate(self.level_prices):
            higher = next((p for p in self.level_prices if p > lp), None)
            lower = next((p for p in reversed(self.level_prices) if p < lp), None)

            buy_sl = self.bellow_max if self.bellow_max is not None else min(self.level_prices)
            sell_sl = self.up_max if self.up_max is not None else max(self.level_prices)

            orders.append({
                "id": f"buy_{idx}",
                "side": "buy",
                "level_price": lp,
                "tp_price": higher,
                "sl_price": buy_sl,
                "status": "pending",
            })
            orders.append({
                "id": f"sell_{idx}",
                "side": "sell",
                "level_price": lp,
                "tp_price": lower,
                "sl_price": sell_sl,
                "status": "pending",
            })
        return orders
    

    def _add_adjacent_orders_if_missing(self, level: float):
        """Add new pending orders for adjacent levels if missing."""
        level_idx = self.level_prices.index(level)
        adjacent_levels = []
        if level_idx > 0:
            adjacent_levels.append(self.level_prices[level_idx - 1])
        if level_idx < len(self.level_prices) - 1:
            adjacent_levels.append(self.level_prices[level_idx + 1])

        for adj_level in adjacent_levels:
            # Check if *any* order (pending or filled) already exists at this level
            existing = [
                o for o in self.pending_orders
                if o["level_price"] == adj_level
            ]
            if existing:
                continue  # Skip if already defined

            # Only create if it’s a known grid level (avoid runaway growth)
            if adj_level not in self.level_prices:
                continue

            higher = next((p for p in self.level_prices if p > adj_level), None)
            lower = next((p for p in reversed(self.level_prices) if p < adj_level), None)
            buy_sl = self.bellow_max if self.bellow_max is not None else min(self.level_prices)
            sell_sl = self.up_max if self.up_max is not None else max(self.level_prices)

            self.pending_orders.append({
                "id": f"buy_new_{adj_level}",
                "side": "buy",
                "level_price": adj_level,
                "tp_price": higher,
                "sl_price": buy_sl,
                "status": "pending",
            })
            self.pending_orders.append({
                "id": f"sell_new_{adj_level}",
                "side": "sell",
                "level_price": adj_level,
                "tp_price": lower,
                "sl_price": sell_sl,
                "status": "pending",
            })


    def run(self):
        active_trades = []

        for _, row in self.candles.iterrows():
            t = row["time"]
            h, l = float(row["high"]), float(row["low"])

            # 1️⃣ Fill pending orders
            for order in self.pending_orders:
                if order["status"] != "pending":
                    continue
                lp = order["level_price"]
                if l <= lp <= h:
                    self._add_adjacent_orders_if_missing(lp)
                    trade = {
                        "id": f"trade_{len(self.trades) + len(active_trades) + 1}",
                        "side": order["side"],
                        "open_time": t,
                        "open_price": lp,
                        "tp_price": order["tp_price"],
                        "sl_price": order["sl_price"],
                        "status": "open",
                    }
                    active_trades.append(trade)
                    order["status"] = "filled"

            # 2️⃣ Manage active trades
            still_active = []
            for tr in active_trades:
                side = tr["side"]
                tp = tr["tp_price"]
                sl = tr["sl_price"]
                open_price = tr["open_price"]

                closed = False
                close_price = None
                reason = None

                if side == "buy":
                    if tp and h >= tp:
                        close_price = tp
                        reason = "tp"
                        closed = True
                    elif sl and l <= sl:
                        close_price = sl
                        reason = "sl"
                        closed = True
                else:
                    if tp and l <= tp:
                        close_price = tp
                        reason = "tp"
                        closed = True
                    elif sl and h >= sl:
                        close_price = sl
                        reason = "sl"
                        closed = True

                if closed:
                    pips = (close_price - open_price) / PIP if side == "buy" else (open_price - close_price) / PIP
                    self.trades.append({
                        "trade_id": tr["id"],
                        "side": side,
                        "open_time": tr["open_time"],
                        "open_price": open_price,
                        "close_time": t,
                        "close_price": close_price,
                        "pips": round(pips, 1),
                        "outcome": "win" if pips > 0 else "loss",
                        "reason": reason,
                    })
                else:
                    still_active.append(tr)
            active_trades = still_active

        # 3️⃣ Close remaining trades at end of data
        final_time = self.candles.iloc[-1]["time"]
        final_close = float(self.candles.iloc[-1]["close"])
        for tr in active_trades:
            side = tr["side"]
            open_price = tr["open_price"]
            pips = (final_close - open_price) / PIP if side == "buy" else (open_price - final_close) / PIP
            self.trades.append({
                "trade_id": tr["id"],
                "side": side,
                "open_time": tr["open_time"],
                "open_price": open_price,
                "close_time": final_time,
                "close_price": final_close,
                "pips": round(pips, 1),
                "outcome": "win" if pips > 0 else "loss",
                "reason": "closed_end",
            })

    def results_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.trades)

    def summary(self) -> Dict:
        df = self.results_df()
        total = len(df)
        wins = len(df[df["pips"] > 0])
        losses = len(df[df["pips"] <= 0])
        total_pips = df["pips"].sum() if total else 0.0
        return {
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "total_pips": round(total_pips, 1),
            "win_rate": round(100 * wins / total, 2) if total else 0.0
        }


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    CSV_PATH = "EURUSD_M1.csv"  # previously downloaded file
    candles = load_candles_csv(CSV_PATH)

    bt = GridBacktester(
        candles,
        levels=100,
        distance_pips=10,
        up_max=None,
        bellow_max=None
    )
    bt.run()

    df = bt.results_df()
    summary = bt.summary()

    print("\n=== Grid Backtest Summary ===")
    print(summary)
    print("\nFirst 10 trades:")
    print(df.head(100).to_string(index=False))
    print(f"\n✅ Final total pips: {summary['total_pips']} pips")

    df.to_csv("grid_results.csv", index=False)
    print("📊 Results saved to grid_results.csv")
