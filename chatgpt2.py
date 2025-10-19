"""
hedging_grid_backtester_dynamic.py

Enhanced version:
 - Dynamic re-arming grid: whenever price hits a level, bot checks adjacent levels and re-adds pending orders if missing.
 - Summary now includes final total pips (pips won/lost).
"""
from __future__ import annotations
import time
from datetime import datetime, timedelta, timezone
from typing import Iterable, Tuple, Optional
import requests
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional





# --- helpers ---------------------------------------------------------------

_GRAN_SECONDS = {
    "S5": 5, "S10": 10, "S15": 15, "S30": 30,
    "M1": 60, "M2": 120, "M4": 240, "M5": 300, "M10": 600, "M15": 900, "M30": 1800,
    "H1": 3600, "H2": 7200, "H3": 10800, "H4": 14400, "H6": 21600, "H8": 28800, "H12": 43200,
    "D": 86400, "W": 604800, "M": 2592000,  # "M" is Oanda monthly granularity; 30d approx for chunking
}

def _ensure_dt(x: str | datetime) -> datetime:
    """Accept RFC3339 '...Z' or datetime; return timezone-aware UTC datetime."""
    if isinstance(x, datetime):
        return x if x.tzinfo else x.replace(tzinfo=timezone.utc)
    # robust-ish RFC3339 parse for ...Z
    return datetime.fromisoformat(x.replace("Z", "+00:00")).astimezone(timezone.utc)

def _iso(dt: datetime) -> str:
    """UTC RFC3339 with 'Z' suffix (what Oanda expects when we pass from/to)."""
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

def _candles_to_df(candles_json: list[dict]) -> pd.DataFrame:
    """Normalize Oanda candle JSON (mid prices) into a tidy DataFrame."""
    if not candles_json:
        return pd.DataFrame(columns=["time","open","high","low","close","complete","volume"])
    rows = []
    for c in candles_json:
        # Oanda returns {"mid": {"o":"...","h":"...","l":"...","c":"..."}, "time":"...", "complete":bool, "volume":int}
        mid = c.get("mid") or c.get("midpoint") or {}
        rows.append({
            "time": pd.to_datetime(c["time"], utc=True),
            "open": float(mid["o"]),
            "high": float(mid["h"]),
            "low":  float(mid["l"]),
            "close":float(mid["c"]),
            "complete": bool(c.get("complete", True)),
            "volume": int(c.get("volume", 0)),
        })
    df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    # keep only completed candles; drop dupes by time
    return (df[df["complete"]]
            .drop_duplicates(subset=["time"])
            .reset_index(drop=True))

def _iter_chunks(start_dt: datetime, end_dt: datetime, granularity: str, per_request_limit: int) -> Iterable[Tuple[datetime, datetime]]:
    """Yield [chunk_start, chunk_end] windows that stay under the candle cap."""
    step = _GRAN_SECONDS[granularity]
    # keep a little headroom under 5000 so boundary overlaps don't exceed the cap
    hard_cap = max(1, min(per_request_limit, 5000) - 50)
    span = timedelta(seconds=hard_cap * step)
    cur = start_dt
    while cur < end_dt:
        nxt = min(cur + span, end_dt)
        yield cur, nxt
        cur = nxt

def _get_with_retry(url: str, headers: dict, params: dict, timeout=30, max_retries=5):
    """GET with simple exponential backoff for 429/5xx."""
    delay = 1.0
    for attempt in range(max_retries):
        resp = requests.get(url, headers=headers, params=params, timeout=timeout)
        if resp.status_code in (429, 500, 502, 503, 504):
            # rate limit or transient server error
            time.sleep(delay)
            delay = min(delay * 2, 10)
            continue
        # other statuses handled by raise_for_status below
        return resp
    return resp  # give caller the last response to raise/log

# --- batched fetch ---------------------------------------------------------

def fetch_oanda_candles(
    token: str,
    instrument: str,
    granularity: str,
    *,
    start: Optional[str | datetime] = None,
    end: Optional[str | datetime] = None,
    count: Optional[int] = None,
    per_request_limit: int = 5000,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Fetch Oanda candles with automatic chunking for long windows.
    - If `count` is given (and start/end are None) → single request for last N candles.
    - Else requires `start` and `end`, and will split into API-safe chunks.

    Returns a tidy DataFrame: time/open/high/low/close/complete/volume
    """
    assert granularity in _GRAN_SECONDS, f"Unsupported granularity: {granularity}"
    url = f"https://api-fxpractice.oanda.com/v3/instruments/{instrument}/candles"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept-Datetime-Format": "RFC3339",
    }

    # Simple 'count' path (no chunking)
    if count is not None and start is None and end is None:
        params = {"granularity": granularity, "price": "M", "count": int(count)}
        if show_progress: print(f"[fetch] {instrument} {granularity} count={count}")
        resp = _get_with_retry(url, headers, params)
        try:
            resp.raise_for_status()
        except requests.HTTPError:
            raise RuntimeError(f"Oanda error {resp.status_code}: {resp.text}")
        return _candles_to_df(resp.json().get("candles", []))

    # Batched path (start/end required)
    if not (start and end):
        raise ValueError("Provide either count=N, or both start and end.")

    start_dt, end_dt = _ensure_dt(start), _ensure_dt(end)
    if start_dt >= end_dt:
        raise ValueError("start must be earlier than end.")

    chunks = list(_iter_chunks(start_dt, end_dt, granularity, per_request_limit))
    all_rows = []
    for i, (c0, c1) in enumerate(chunks, 1):
        params = {"granularity": granularity, "price": "M", "from": _iso(c0), "to": _iso(c1)}
        if show_progress:
            print(f"[fetch {i}/{len(chunks)}] {params['from']} → {params['to']}", flush=True)
        resp = _get_with_retry(url, headers, params)
        try:
            resp.raise_for_status()
        except requests.HTTPError:
            raise RuntimeError(
                f"Oanda error {resp.status_code} for window {params['from']}→{params['to']}: {resp.text}"
            )
        all_rows.extend(resp.json().get("candles", []))

    df = _candles_to_df(all_rows)
    if show_progress and not df.empty:
        print(f"[fetch done] {instrument} {granularity}: {len(df)} candles from {df.time.min()} to {df.time.max()}")
    return df


PIP = 0.0001  # EUR/USD pip size

"""
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
    """


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
            #for order in self.pending_orders:
            for order in list(self.pending_orders):  # iterate over a snapshot
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
    START = "2025-10-10T00:00:00Z"
    END = "2025-10-15T11:03:00Z"
    INSTRUMENT = "EUR_USD"
    GRANULARITY = "H1"

    # Grid configuration
    LEVELS = 50
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
    print(df.tail(60).to_string(index=False))
    print(f"\n✅ Final total pips: {summary['total_pips']} pips")
