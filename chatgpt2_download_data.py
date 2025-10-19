"""
download_oanda_data.py
Fetches historical OANDA candle data and saves it as CSV for local use.

Usage:
  python download_oanda_data.py
"""

import requests
import pandas as pd

def fetch_oanda_candles(
    oanda_token: str,
    instrument: str = "EUR_USD",
    granularity: str = "H1",
    start: str = "2025-10-14T00:00:00Z",
    end: str = "2025-10-16T23:59:00Z",
    filename: str = "EURUSD_H1.csv"
):
    url = f"https://api-fxpractice.oanda.com/v3/instruments/{instrument}/candles"
    headers = {"Authorization": f"Bearer {oanda_token}"}
    params = {"granularity": granularity, "price": "M", "from": start, "to": end}

    print(f"Downloading {instrument} candles from OANDA...")
    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    rows = []
    for c in data["candles"]:
        if not c["complete"]:
            continue
        rows.append({
            "time": c["time"],
            "open": float(c["mid"]["o"]),
            "high": float(c["mid"]["h"]),
            "low": float(c["mid"]["l"]),
            "close": float(c["mid"]["c"]),
        })

    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"])
    df.to_csv(filename, index=False)
    print(f"✅ Data saved to {filename} ({len(df)} candles)")
    return df


if __name__ == "__main__":
    OANDA_TOKEN = "28ea3ce61c08a436f559a2aa4f49cab2-c6732cd18d2ec0bbe4f8579adb722602"
    END = "2025-10-15T23:59:59Z" 
    START = "2025-10-15T00:00:00Z"

    fetch_oanda_candles(
        oanda_token=OANDA_TOKEN,
        instrument="EUR_USD",
        granularity="M1",
        start=START,
        end=END,
        filename="EURUSD_M1.csv",
    )
