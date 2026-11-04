#!/usr/bin/env python3
"""
H-C Fourier kNN v1 — BTCUSDT USDT perpetual (Bybit public API)
Standalone research backtest. No API key required.

Important:
- Daily OHLC candles from Bybit linear BTCUSDT.
- Funding history is applied at funding timestamps using daily close as a
  proxy for mark price (not tick-accurate).
- Conservative daily-bar stop model: gap fills at open; otherwise stop price.
- Signal at daily close, entry at next daily open.
- Position sizing: 2% equity risk budget, capped at 100% notional.
- Commission: 0.05% per side. Funding is additional.
- This is a research approximation of the supplied Pine logic, not a
  byte-for-byte TradingView broker-emulator replica.
"""
import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = "https://api.bybit.com"
SYMBOL = "BTCUSDT"
CATEGORY = "linear"
INTERVAL = "D"
W, HIST, K = 13, 588, 25
MIN_CONC, MIN_PRED = 0.35, 0.005
SMA_LEN, ATR_LEN = 50, 12
ATR_MULT, STOP_FLOOR = 2.0, 0.008
RISK_PCT, MAX_EXPOSURE = 0.02, 1.0
HOLD_BARS, MIN_NEIGHBORS = 13, 5
FEE = 0.0005  # 0.05% per side


def get_json(path, params):
    r = requests.get(BASE + path, params=params, timeout=30)
    r.raise_for_status()
    obj = r.json()
    if obj.get("retCode") != 0:
        raise RuntimeError(f"Bybit API error: {obj}")
    return obj["result"]


def fetch_klines(start_ms):
    rows, end = [], int(datetime.now(timezone.utc).timestamp() * 1000)
    cursor = end
    while True:
        result = get_json("/v5/market/kline", {
            "category": CATEGORY, "symbol": SYMBOL, "interval": INTERVAL,
            "start": start_ms, "end": cursor, "limit": 1000
        })
        batch = result.get("list", [])
        if not batch:
            break
        rows.extend(batch)
        oldest = min(int(x[0]) for x in batch)
        if oldest <= start_ms or len(batch) < 1000:
            break
        cursor = oldest - 1
        time.sleep(0.12)
    df = pd.DataFrame(rows, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "turnover"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume", "turnover"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    # Bybit returns the currently forming candle; discard it.
    now = pd.Timestamp.now(tz="UTC").normalize()
    df = df[df.timestamp < now].reset_index(drop=True)
    return df


def fetch_funding(start_ms, end_ms):
    rows, cursor = [], start_ms
    while cursor < end_ms:
        result = get_json("/v5/market/funding/history", {
            "category": CATEGORY, "symbol": SYMBOL,
            "startTime": cursor, "endTime": end_ms, "limit": 200
        })
        batch = result.get("list", [])
        if not batch:
            break
        rows.extend(batch)
        newest = max(int(x["fundingRateTimestamp"]) for x in batch)
        if len(batch) < 200 or newest <= cursor:
            break
        cursor = newest + 1
        time.sleep(0.12)
    if not rows:
        return pd.DataFrame(columns=["timestamp", "rate"])
    f = pd.DataFrame(rows)
    f["timestamp"] = pd.to_datetime(f["fundingRateTimestamp"].astype("int64"), unit="ms", utc=True)
    f["rate"] = pd.to_numeric(f["fundingRate"], errors="coerce")
    return f[["timestamp", "rate"]].drop_duplicates("timestamp").sort_values("timestamp")


def indicators(df):
    high, low, close = df.high, df.low, df.close
    prev = close.shift(1)
    tr = pd.concat([high-low, (high-prev).abs(), (low-prev).abs()], axis=1).max(axis=1)
    # Pine ta.atr uses Wilder's RMA.
    atr = tr.ewm(alpha=1/ATR_LEN, adjust=False, min_periods=ATR_LEN).mean()
    sma = close.rolling(SMA_LEN, min_periods=SMA_LEN).mean()
    return atr.to_numpy(), sma.to_numpy()


def fourier_feature(log_prices):
    """Residual to endpoint chord, first six nonzero DFT bins, normalized."""
    x = np.asarray(log_prices, dtype=float)
    n = len(x)
    chord = np.linspace(x[0], x[-1], n)
    residual = x - chord
    spec = np.fft.rfft(residual)
    coeff = spec[1:7]
    powers = np.abs(spec[1:]) ** 2
    total = float(powers.sum())
    if total <= 1e-20:
        return np.zeros(12), 0.0
    conc = float(powers.max() / total)
    norm = float(np.linalg.norm(coeff))
    if norm <= 1e-20:
        return np.zeros(12), conc
    coeff = coeff / norm
    vec = np.empty(12)
    vec[0::2], vec[1::2] = coeff.real, coeff.imag
    return vec, conc


def build_features(df):
    n = len(df)
    feats = np.full((n, 12), np.nan)
    conc = np.full(n, np.nan)
    logc = np.log(df.close.to_numpy())
    for t in range(W-1, n):
        feats[t], conc[t] = fourier_feature(logc[t-W+1:t+1])
    return feats, conc


def run_backtest(df, funding):
    df = df.copy().reset_index(drop=True)
    atr, sma = indicators(df)
    feats, conc = build_features(df)
    o, h, l, c = [df[x].to_numpy() for x in ["open", "high", "low", "close"]]
    ts = df.timestamp.to_numpy()
    equity, peak, max_dd = 10000.0, 10000.0, 0.0
    position = 0
    qty = entry = stop = 0.0
    entry_i = -1
    trades, equity_curve = [], []
    pending = None
    funding = funding.copy()
    funding["day"] = funding.timestamp.dt.floor("D")
    funding_by_day = funding.groupby("day").rate.apply(list).to_dict()

    for i in range(max(W, SMA_LEN), len(df)):
        # Entry scheduled from prior daily close, fills at today's open.
        if pending is not None and position == 0:
            direction, stop_dist = pending
            fill = o[i]
            risk_cash = equity * RISK_PCT
            qty_risk = risk_cash / max(stop_dist, fill * 1e-9)
            qty_cap = equity * MAX_EXPOSURE / fill
            qty = min(qty_risk, qty_cap)
            if qty > 0 and np.isfinite(qty):
                position, entry, entry_i = direction, fill, i
                stop = entry - direction * stop_dist
                fee = qty * fill * FEE
                equity -= fee
                trade = {"entry_time": str(ts[i]), "side": "LONG" if direction == 1 else "SHORT",
                         "entry": float(fill), "qty_btc": float(qty), "entry_fee": float(fee),
                         "funding": 0.0, "exit_time": None, "exit": None, "exit_reason": None}
                trades.append(trade)
            pending = None

        # Funding cashflows for each funding event on this UTC day.
        if position and str(pd.Timestamp(ts[i]).floor("D")) in funding_by_day:
            rates = funding_by_day[str(pd.Timestamp(ts[i]).floor("D"))]
            for rate in rates:
                # Positive funding: longs pay, shorts receive.
                cashflow = -position * qty * c[i] * float(rate)
                equity += cashflow
                trades[-1]["funding"] += cashflow

        # Stop check on today's range. A stop beyond the open fills at open.
        exited = False
        if position:
            hit = (position == 1 and l[i] <= stop) or (position == -1 and h[i] >= stop)
            if hit:
                fill = min(o[i], stop) if position == 1 else max(o[i], stop)
                fee = qty * fill * FEE
                pnl = position * qty * (fill - entry)
                equity += pnl - fee
                trades[-1].update(exit_time=str(ts[i]), exit=float(fill),
                                  exit_reason="stop", exit_fee=float(fee),
                                  gross_pnl=float(pnl), net_trade_pnl=float(
                                      pnl - trades[-1]["entry_fee"] - fee + trades[-1]["funding"]))
                position, qty, entry, stop, entry_i = 0, 0.0, 0.0, 0.0, -1
                exited = True

        # Time exit: after 13 daily bars, submitted at close and filled next open.
        if position and i - entry_i >= HOLD_BARS - 1 and i + 1 < len(df):
            fill = c[i]
            fee = qty * fill * FEE
            pnl = position * qty * (fill - entry)
            equity += pnl - fee
            trades[-1].update(exit_time=str(ts[i]), exit=float(fill),
                              exit_reason="time_close", exit_fee=float(fee),
                              gross_pnl=float(pnl), net_trade_pnl=float(
                                  pnl - trades[-1]["entry_fee"] - fee + trades[-1]["funding"]))
            position, qty, entry, stop, entry_i = 0, 0.0, 0.0, 0.0, -1
            exited = True

        # Mark-to-market for drawdown.
        mtm = equity + (position * qty * (c[i] - entry) if position else 0.0)
        peak = max(peak, mtm)
        max_dd = min(max_dd, mtm / peak - 1.0)
        equity_curve.append({"timestamp": str(ts[i]), "equity": float(mtm)})

        # Calculate today's signal only after today's close; fill next day's open.
        if position == 0 and pending is None and i + 1 < len(df) and np.isfinite(atr[i]) and np.isfinite(sma[i]):
            x = feats[i]
            if np.all(np.isfinite(x)):
                distances, outcomes = [], []
                # Candidate at j=i-offset; its future label is known by today.
                for offset in range(W, min(HIST, i-1) + 1):
                    j = i - offset
                    if j < W-1 or not np.isfinite(conc[j]) or conc[j] < MIN_CONC:
                        continue
                    target_idx = j + W
                    entry_idx = j + 1
                    if target_idx > i or entry_idx >= len(df):
                        continue
                    if not np.all(np.isfinite(feats[j])):
                        continue
                    d = float(np.linalg.norm(x - feats[j]))
                    y = c[target_idx] / o[entry_idx] - 1.0
                    distances.append(d)
                    outcomes.append(y)
                if len(distances) >= MIN_NEIGHBORS:
                    order = np.argsort(distances)[:K]
                    ds = np.asarray(distances)[order]
                    ys = np.asarray(outcomes)[order]
                    weights = 1.0 / np.maximum(ds, 1e-8)
                    pred = float(np.sum(weights * ys) / np.sum(weights))
                    direction = 1 if pred >= MIN_PRED and c[i] > sma[i] else (
                        -1 if pred <= -MIN_PRED and c[i] < sma[i] else 0)
                    if direction:
                        stop_dist = max(ATR_MULT * atr[i], STOP_FLOOR * c[i])
                        pending = (direction, float(stop_dist))

    # Liquidate any remaining position at final close for a closed-trade report.
    if position:
        i = len(df)-1
        fill = c[i]
        fee = qty * fill * FEE
        pnl = position * qty * (fill-entry)
        equity += pnl-fee
        trades[-1].update(exit_time=str(ts[i]), exit=float(fill), exit_reason="end_of_data",
                          exit_fee=float(fee), gross_pnl=float(pnl),
                          net_trade_pnl=float(pnl-trades[-1]["entry_fee"]-fee+trades[-1]["funding"]))
    return trades, pd.DataFrame(equity_curve), equity, max_dd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01-01", help="UTC start date (default 2018-01-01)")
    ap.add_argument("--out", default="bybit_hc_results")
    args = ap.parse_args()
    start = pd.Timestamp(args.start, tz="UTC")
    start_ms = int(start.timestamp() * 1000)
    print("Downloading Bybit BTCUSDT linear daily candles...")
    df = fetch_klines(start_ms)
    if len(df) < 700:
        raise SystemExit(f"Not enough candles: {len(df)}")
    print(f"Downloaded {len(df)} completed daily candles: {df.timestamp.iloc[0]} to {df.timestamp.iloc[-1]}")
    end_ms = int(df.timestamp.iloc[-1].timestamp() * 1000) + 86_400_000
    print("Downloading funding history...")
    funding = fetch_funding(start_ms, end_ms)
    print(f"Downloaded {len(funding)} funding events.")
    trades, curve, final_equity, max_dd = run_backtest(df, funding)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tdf = pd.DataFrame(trades)
    tdf.to_csv(out / "trades.csv", index=False)
    curve.to_csv(out / "equity_curve.csv", index=False)
    start_eq = 10000.0
    wins = int((tdf.net_trade_pnl > 0).sum()) if len(tdf) else 0
    gross_profit = float(tdf.loc[tdf.net_trade_pnl > 0, "net_trade_pnl"].sum()) if len(tdf) else 0
    gross_loss = float(-tdf.loc[tdf.net_trade_pnl < 0, "net_trade_pnl"].sum()) if len(tdf) else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else None
    summary = {
        "symbol": SYMBOL, "category": CATEGORY, "timeframe": "1D",
        "first_candle": str(df.timestamp.iloc[0]), "last_candle": str(df.timestamp.iloc[-1]),
        "candles": int(len(df)), "funding_events": int(len(funding)),
        "initial_equity": start_eq, "final_equity": float(final_equity),
        "total_return_pct": float((final_equity/start_eq-1)*100),
        "max_drawdown_pct": float(max_dd*100), "trades": int(len(tdf)),
        "wins": wins, "win_rate_pct": float(wins/len(tdf)*100) if len(tdf) else 0,
        "profit_factor": pf, "commission_rate_per_side": FEE,
        "funding_mark_price_proxy": "daily close; approximate",
        "stop_model": "daily OHLC; gap-aware; intraday path not reconstructed",
        "strategy_note": "research approximation, not exact TradingView broker emulator"
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if len(tdf):
        tdf["year"] = pd.to_datetime(tdf.exit_time, utc=True).dt.year
        annual = tdf.groupby("year").agg(trades=("side", "count"),
            net_pnl=("net_trade_pnl", "sum"),
            wins=("net_trade_pnl", lambda x: int((x > 0).sum()))).reset_index()
        annual.to_csv(out / "annual_results.csv", index=False)
        print("\nAnnual trade P/L:")
        print(annual.to_string(index=False))


if __name__ == "__main__":
    main()
