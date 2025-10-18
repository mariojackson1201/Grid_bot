"""
hedging_grid_backtester.py

Requirements:
  - python 3.8+
  - pip install pandas requests

How to use:
  - Set OANDA_API_TOKEN and optionally OANDA_ACCOUNT_ID (if using account endpoints).
  - Or export candles to CSV and pass csv_path.
  - Configure grid params in the example usage at bottom and run.
"""

import requests
import pandas as pd
import math
from datetime import datetime, timezone
from typing import List, Dict, Optional

PIP = 0.0001  # EUR/USD pip size


def fetch_oanda_candles(oanda_token: str,
                       instrument: str = "EUR_USD",
                       granularity: str = "M1",
                       start: Optional[str] = None,
                       end: Optional[str] = None,
                       count: Optional[int] = None) -> pd.DataFrame:
    """
    Fetch candles from OANDA (v20 REST).
    Provide ISO timestamps for start/end (e.g. "2024-01-01T00:00:00Z").
    If your environment blocks direct calls, you can export the CSV manually and use load_candles_csv.
    """
    url = "https://api-fxpractice.oanda.com/v3/instruments/{}/candles".format(instrument)
    headers = {"Authorization": f"Bearer {oanda_token}"}
    params = {"granularity": granularity, "price": "M"}  # mid prices
    if start: params["from"] = start
    if end: params["to"] = end
    if count: params["count"] = count

    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for c in data["candles"]:
        if not c["complete"]:
            # you may choose to include incomplete bars if desired
            continue
        t = c["time"]
        o = float(c["mid"]["o"])
        h = float(c["mid"]["h"])
        l = float(c["mid"]["l"])
        cl = float(c["mid"]["c"])
        rows.append({"time": t, "open": o, "high": h, "low": l, "close": cl})

    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


def load_candles_csv(path: str, time_col: str = "time") -> pd.DataFrame:
    """
    Load CSV containing time, open, high, low, close columns. Time parsed to datetime.
    """
    df = pd.read_csv(path, parse_dates=[time_col])
    df = df.rename(columns={time_col: "time", "o": "open", "h": "high", "l": "low", "c": "close"}).copy()
    df = df[["time", "open", "high", "low", "close"]]
    df = df.sort_values("time").reset_index(drop=True)
    return df


class GridBacktester:
    def __init__(self,
                 candles: pd.DataFrame,
                 start_price: Optional[float] = None,
                 levels: int = 10,
                 distance_pips: float = 10.0,
                 up_max: Optional[float] = None,
                 bellow_max: Optional[float] = None):
        """
        candles: DataFrame with columns time, open, high, low, close (chronological order)
        start_price: the initial reference price to place grid. If None, uses first candle's close.
        levels: number of grid levels (used per side total; we will create 'levels' above and 'levels' below ideally)
        distance_pips: distance between adjacent levels in pips (float)
        up_max: an absolute price that acts as upper stop for sells (if None, will be computed from grid)
        bellow_max: lower stop price for buys
        """
        self.candles = candles.copy().reset_index(drop=True)
        self.start_price = start_price if start_price is not None else float(self.candles.loc[0, "close"])
        self.levels = int(levels)
        self.distance = float(distance_pips) * PIP
        self.up_max = up_max
        self.bellow_max = bellow_max
        self.level_prices = self._build_levels()
        self.pending_orders = self._init_pending_orders()
        self.trades = []  # list of dicts with trade details

    def _build_levels(self) -> List[float]:
        """
        Build level prices symmetric around start_price according to levels and distance.
        The list is sorted ascending.
        """
        prices = []
        for i in range(-self.levels, self.levels + 1):
            prices.append(round(self.start_price + i * self.distance, 6))
        prices = sorted(list(set(prices)))
        # If up_max/bellow_max provided, ensure they are inside or outside? We'll enforce SLs separately.
        return prices

    def _init_pending_orders(self) -> List[Dict]:
        """
        For each level create a pending buy and pending sell order.
        Each order dict: {id, side, level_price, tp_price, sl_price, status}
        status: "pending" | "filled" | "closed" ; when filled we store trade id in 'trade_id' etc.
        """
        orders = []
        for idx, lp in enumerate(self.level_prices):
            # compute TP as adjacent level above for buy, below for sell (if exists)
            # find next higher level
            higher = None
            lower = None
            for p in self.level_prices:
                if p > lp:
                    higher = p
                    break
            for p in reversed(self.level_prices):
                if p < lp:
                    lower = p
                    break

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

    def run(self):
        """
        Run the simulation over the candle series.
        Execution rules:
         - buy pending at level L fills when candle.low <= L (filled at L)
         - sell pending at level L fills when candle.high >= L (filled at L)
         - after filled, trade is active until TP or SL is hit:
            * buy: TP = next level above (if exists), SL = bellow_max (global)
            * sell: TP = next level below (if exists), SL = up_max (global)
         - intrabar ordering:
            * buy: assume open -> high -> low -> close (so highs occur before lows)
            * sell: assume open -> low -> high -> close (so lows occur before highs)
        """
        active_trades = []

        for i, row in self.candles.iterrows():
            t = row["time"]
            o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])

            # 1) Check pending orders to fill at this candle
            for order in self.pending_orders:
                if order["status"] != "pending":
                    continue
                lp = order["level_price"]
                if order["side"] == "buy":
                    if l <= lp <= h:  # level inside candle; buy fills if low <= lp
                        # fill at lp
                        trade = {
                            "id": f"trade_{len(self.trades) + len(active_trades) + 1}",
                            "side": "buy",
                            "open_time": t,
                            "open_price": lp,
                            "tp_price": order["tp_price"],
                            "sl_price": order["sl_price"],
                            "status": "open",
                            "open_idx": i,
                        }
                        active_trades.append(trade)
                        order["status"] = "filled"
                else:  # sell
                    if l <= lp <= h:  # level inside candle; sell fills if high >= lp
                        trade = {
                            "id": f"trade_{len(self.trades) + len(active_trades) + 1}",
                            "side": "sell",
                            "open_time": t,
                            "open_price": lp,
                            "tp_price": order["tp_price"],
                            "sl_price": order["sl_price"],
                            "status": "open",
                            "open_idx": i,
                        }
                        active_trades.append(trade)
                        order["status"] = "filled"

            # 2) For active trades, check TP/SL hits in this candle
            still_active = []
            for tr in active_trades:
                side = tr["side"]
                tp = tr["tp_price"]
                sl = tr["sl_price"]
                open_price = tr["open_price"]

                closed = False
                close_price = None
                close_time = None
                reason = None  # 'tp' or 'sl'

                # If no TP (edge of grid), we treat TP as None -> only SL applies
                if side == "buy":
                    # if TP exists and candle high reaches TP -> TP hit
                    tp_hit = (tp is not None) and (h >= tp)
                    sl_hit = (l <= sl) if (sl is not None) else False

                    if tp_hit and sl_hit:
                        # both in same candle -> buy intrabar path open->high->low => tp first
                        close_price = tp
                        close_time = t
                        reason = "tp"
                        closed = True
                    elif tp_hit:
                        close_price = tp
                        close_time = t
                        reason = "tp"
                        closed = True
                    elif sl_hit:
                        close_price = sl
                        close_time = t
                        reason = "sl"
                        closed = True
                else:  # sell
                    tp_hit = (tp is not None) and (l <= tp)
                    sl_hit = (h >= sl) if (sl is not None) else False

                    if tp_hit and sl_hit:
                        # intrabar open->low->high => tp first for sells
                        close_price = tp
                        close_time = t
                        reason = "tp"
                        closed = True
                    elif tp_hit:
                        close_price = tp
                        close_time = t
                        reason = "tp"
                        closed = True
                    elif sl_hit:
                        close_price = sl
                        close_time = t
                        reason = "sl"
                        closed = True

                if closed:
                    # record trade
                    pips = (close_price - open_price) / PIP if side == "buy" else (open_price - close_price) / PIP
                    self.trades.append({
                        "trade_id": tr["id"],
                        "side": side,
                        "open_time": tr["open_time"],
                        "open_price": tr["open_price"],
                        "close_time": close_time,
                        "close_price": close_price,
                        "pips": round(pips, 1),
                        "outcome": "win" if pips > 0 else ("break_even" if abs(pips) < 1e-9 else "loss"),
                        "reason": reason,
                    })
                else:
                    # remains active
                    still_active.append(tr)
            active_trades = still_active

        # any still active trades at end: we close them at final close price (or leave open? we'll close)
        final_time = self.candles.iloc[-1]["time"]
        final_price = float(self.candles.iloc[-1]["close"])

        for tr in active_trades:
            side = tr["side"]
            open_price = tr["open_price"]
            close_price = final_price
            pips = (close_price - open_price) / PIP if side == "buy" else (open_price - close_price) / PIP
            self.trades.append({
                "trade_id": tr["id"],
                "side": side,
                "open_time": tr["open_time"],
                "open_price": tr["open_price"],
                "close_time": final_time,
                "close_price": close_price,
                "pips": round(pips, 1),
                "outcome": "win" if pips > 0 else ("break_even" if abs(pips) < 1e-9 else "loss"),
                "reason": "closed_end",
            })

    def results_df(self) -> pd.DataFrame:
        df = pd.DataFrame(self.trades)
        # ensure columns
        cols = ["trade_id", "side", "open_time", "open_price", "close_time", "close_price", "pips", "outcome", "reason"]
        for c in cols:
            if c not in df.columns:
                df[c] = None
        df = df[cols]
        return df

    def summary(self) -> Dict:
        df = self.results_df()
        if df.empty:
            return {"message": "No trades executed"}

        total = len(df)
        wins_df = df[df["pips"] > 0]
        losses_df = df[df["pips"] < 0]
        be_df = df[df["pips"] == 0]

        total_pips_won = wins_df["pips"].sum()
        total_pips_lost = abs(losses_df["pips"].sum())
        net_pips = total_pips_won - total_pips_lost

        summary = {
            "total_trades": total,
            "wins": len(wins_df),
            "losses": len(losses_df),
            "breakeven": len(be_df),
            "total_pips_won": round(float(total_pips_won), 1),
            "total_pips_lost": round(float(total_pips_lost), 1),
            "net_pips": round(float(net_pips), 1),
            "avg_win_pips": round(wins_df["pips"].mean(), 1) if not wins_df.empty else None,
            "avg_loss_pips": round(losses_df["pips"].mean(), 1) if not losses_df.empty else None,
            "win_rate_%": round(100 * len(wins_df) / total, 2),
        }
        return summary


# ---------------------------
# Example usage (fill your OANDA token or use CSV)
# ---------------------------
if __name__ == "__main__":
    # Example parameters - tweak these
    OANDA_TOKEN = "28ea3ce61c08a436f559a2aa4f49cab2-c6732cd18d2ec0bbe4f8579adb722602"
    CSV_PATH = None  # e.g. "EURUSD_H1_2024.csv"
    INSTRUMENT = "EUR_USD"
    GRANULARITY = "M1"
    END = "2025-10-15T23:59:59Z" 
    START = "2025-10-15T00:00:00Z"

    # Grid parameters
    START_PRICE = None  # None -> uses first candle close
    LEVELS = 8
    DISTANCE_PIPS = 10.0
    UP_MAX = None
    BELOW_MAX = None

    if OANDA_TOKEN and not CSV_PATH:
        candles = fetch_oanda_candles(OANDA_TOKEN, INSTRUMENT, GRANULARITY, start=START, end=END)
    elif CSV_PATH:
        candles = load_candles_csv(CSV_PATH)
    else:
        raise RuntimeError("Provide OANDA token or CSV path for candles")

    bt = GridBacktester(candles,
                        start_price=START_PRICE,
                        levels=LEVELS,
                        distance_pips=DISTANCE_PIPS,
                        up_max=UP_MAX,
                        bellow_max=BELOW_MAX)
    bt.run()
    results = bt.results_df()
    summary = bt.summary()
    print("\n===== GRID BOT BACKTEST SUMMARY =====")
    for k, v in summary.items():
        print(f"{k:20s}: {v}")

    print("SUMMARY:", summary)
    print("First 20 trades:")
    print(results.head(20).to_string(index=False))
    # save results
    results.to_csv("grid_backtest_trades.csv", index=False)
    print("Saved per-trade CSV to grid_backtest_trades.csv")
