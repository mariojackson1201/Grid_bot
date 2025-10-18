"""
hedging_grid_backtester_dynamic.py

Enhanced version:
 - Dynamic re-arming grid: whenever price hits a level, bot checks adjacent levels and re-adds pending orders if missing.
 - Summary now includes final total pips (pips won/lost).
"""

import requests
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional

PIP = 0.0001  # EUR/USD pip size


def fetch_oanda_candles(oanda_token: str,
                       instrument: str = "EUR_USD",
                       granularity: str = "H1",
                       start: Optional[str] = None,
                       end: Optional[str] = None,
                       count: Optional[int] = None) -> pd.DataFrame:
    url = f"https://api-fxpractice.oanda.com/v3/instruments/{instrument}/candles"
    headers = {"Authorization": f"Bearer {oanda_token}"}
    params = {"granularity": granularity, "price": "M"}
    if start:
        params["from"] = start
    if end:
        params["to"] = end
    if count:
        params["count"] = count

    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for c in data["candles"]:
        if not c["complete"]:
            continue
        rows.append({
            "time": pd.to_datetime(c["time"]),
            "open": float(c["mid"]["o"]),
            "high": float(c["mid"]["h"]),
            "low": float(c["mid"]["l"]),
            "close": float(c["mid"]["c"]),
        })
    df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    return df


class GridBacktester:
    def __init__(self,
                 candles: pd.DataFrame,
                 start_price: Optional[float] = None,
                 levels: int = 10,
                 distance_pips: float = 10.0,
                 up_max: Optional[float] = None,
                 bellow_max: Optional[float] = None):
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
        """
        When a level is hit, ensure the levels above/below have pending orders.
        """
        level_idx = self.level_prices.index(level)
        adjacent_levels = []
        if level_idx > 0:
            adjacent_levels.append(self.level_prices[level_idx - 1])
        if level_idx < len(self.level_prices) - 1:
            adjacent_levels.append(self.level_prices[level_idx + 1])

        for adj_level in adjacent_levels:
            # check if there's already a pending order at adj_level
            existing = [o for o in self.pending_orders if o["level_price"] == adj_level and o["status"] == "pending"]
            if not existing:
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

        for i, row in self.candles.iterrows():
            t = row["time"]
            h, l = float(row["high"]), float(row["low"])

            # 1. Check pending orders
            for order in self.pending_orders:
                if order["status"] != "pending":
                    continue
                lp = order["level_price"]
                if order["side"] == "buy" and l <= lp <= h:
                    self._add_adjacent_orders_if_missing(lp)  # new feature
                    trade = {
                        "id": f"trade_{len(self.trades) + len(active_trades) + 1}",
                        "side": "buy",
                        "open_time": t,
                        "open_price": lp,
                        "tp_price": order["tp_price"],
                        "sl_price": order["sl_price"],
                        "status": "open",
                    }
                    active_trades.append(trade)
                    order["status"] = "filled"

                elif order["side"] == "sell" and l <= lp <= h:
                    self._add_adjacent_orders_if_missing(lp)  # new feature
                    trade = {
                        "id": f"trade_{len(self.trades) + len(active_trades) + 1}",
                        "side": "sell",
                        "open_time": t,
                        "open_price": lp,
                        "tp_price": order["tp_price"],
                        "sl_price": order["sl_price"],
                        "status": "open",
                    }
                    active_trades.append(trade)
                    order["status"] = "filled"

            # 2. Manage active trades
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
                    tp_hit = tp and h >= tp
                    sl_hit = sl and l <= sl
                    if tp_hit:
                        close_price = tp
                        reason = "tp"
                        closed = True
                    elif sl_hit:
                        close_price = sl
                        reason = "sl"
                        closed = True
                else:
                    tp_hit = tp and l <= tp
                    sl_hit = sl and h >= sl
                    if tp_hit:
                        close_price = tp
                        reason = "tp"
                        closed = True
                    elif sl_hit:
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

        # Close all remaining trades at last candle
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
        df = pd.DataFrame(self.trades)
        if df.empty:
            return pd.DataFrame(columns=["trade_id","side","open_time","open_price","close_time","close_price","pips","outcome","reason"])
        return df

    def summary(self) -> Dict:
        df = self.results_df()
        total = len(df)
        wins = len(df[df["pips"] > 0])
        losses = len(df[df["pips"] <= 0])
        total_pips = df["pips"].sum() if not df.empty else 0.0
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
    OANDA_TOKEN = "28ea3ce61c08a436f559a2aa4f49cab2-c6732cd18d2ec0bbe4f8579adb722602"
    START = "2025-10-15T00:00:00Z"
    END = "2025-10-15T23:59:59Z"
    INSTRUMENT = "EUR_USD"
    GRANULARITY = "M1"

    # Grid configuration
    LEVELS = 8
    DISTANCE_PIPS = 10
    UP_MAX = None
    BELOW_MAX = None

    candles = fetch_oanda_candles(OANDA_TOKEN, INSTRUMENT, GRANULARITY, start=START, end=END)
    bt = GridBacktester(candles, levels=LEVELS, distance_pips=DISTANCE_PIPS, up_max=UP_MAX, bellow_max=BELOW_MAX)
    bt.run()
    df = bt.results_df()
    summary = bt.summary()

    print("\n=== Backtest Summary ===")
    print(summary)
    print("\nFirst 40 trades:")
    print(df.head(40).to_string(index=False))
    print(f"\n✅ Final total pips: {summary['total_pips']} pips")
