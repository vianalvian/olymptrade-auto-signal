# app_v3.py
# OlympTrade Auto Signal v3 — Real-time (TwelveData) + popup 1x per new signal
# Requirements: put your TwelveData API key into Streamlit Secrets as TWELVE_API_KEY = "your_key"

import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import html
from datetime import datetime

st.set_page_config(page_title="OlympTrade Auto Signal v3 (Real-time)", layout="wide")
st.title("📈 OlympTrade Auto Signal v3 — Real-time (1m)")
st.caption("Sinyal = RSI(14)+SMA(14)+MACD(12,26,9)+Bollinger(20,2). Popup 1x per sinyal baru. Data: TwelveData (1m).")

# ---------------- Settings (UI) ----------------
pairs_text = st.sidebar.text_area(
    "Pairs (one per line, TwelveData symbol style, e.g. EUR/USD, XAU/USD, BTC/USD)",
    value="EUR/USD\nGBP/USD\nAUD/USD\nUSD/CAD\nUSD/JPY\nXAU/USD\nBTC/USD"
)
PAIR_LIST = [p.strip() for p in pairs_text.splitlines() if p.strip()]
refresh_sec = st.sidebar.number_input("Refresh interval (seconds)", min_value=30, max_value=300, value=60)
bars = st.sidebar.number_input("History bars (for indicators)", min_value=60, max_value=500, value=200)

# indicator params (fixed but showable)
rsi_period = 14
sma_period = 14
macd_fast, macd_slow, macd_sig = 12, 26, 9
bb_period, bb_mult = 20, 2.0

# get API key from secrets
API_KEY = st.secrets.get("TWELVE_API_KEY", None)
if not API_KEY:
    st.error("TWELVE_API_KEY belum diset. Buka Advanced settings → Secrets di Streamlit Cloud dan tambahkan:\nTWELVE_API_KEY = \"your_key\"")
    st.stop()

# --------- Utility: fetch data from TwelveData ----------
def fetch_td(symbol, interval="1min", outputsize=200):
    # TwelveData expects symbol like "EUR/USD" or "BTC/USD"
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "format": "JSON",
        "apikey": API_KEY
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        j = r.json()
    except Exception as e:
        return None, f"Request failed: {e}"
    if "values" not in j:
        return None, j.get("message", str(j))
    df = pd.DataFrame(j["values"])
    # convert columns
    for c in ["open","high","low","close","volume"]:
        if c in df.columns:
            df[c] = df[c].astype(float)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df, None

# --------- Indicators ----------
def add_indicators(df):
    df = df.copy()
    df["SMA14"] = df["close"].rolling(window=sma_period, min_periods=1).mean()
    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=rsi_period, min_periods=1).mean()
    avg_loss = loss.rolling(window=rsi_period, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].fillna(method="bfill").fillna(50)
    # MACD
    ema_fast = df["close"].ewm(span=macd_fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=macd_slow, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=macd_sig, adjust=False).mean()
    # Bollinger
    df["BB_MID"] = df["close"].rolling(window=bb_period, min_periods=1).mean()
    df["BB_STD"] = df["close"].rolling(window=bb_period, min_periods=1).std()
    df["BB_UP"] = df["BB_MID"] + bb_mult * df["BB_STD"]
    df["BB_LOW"] = df["BB_MID"] - bb_mult * df["BB_STD"]
    return df

# --------- Signal logic (combined, conservative) ----------
def compute_signal(df):
    if df is None or len(df) < max(bars, bb_period, sma_period, rsi_period):
        return "NO_DATA"
    last = df.iloc[-1]
    prev = df.iloc[-2]
    price = last["close"]
    # base conditions
    rsi_rise = (prev["RSI"] <= 30) and (last["RSI"] > prev["RSI"])
    rsi_fall = (prev["RSI"] >= 70) and (last["RSI"] < prev["RSI"])
    above_sma = price > last["SMA14"]
    below_sma = price < last["SMA14"]
    macd_cross_up = (prev["MACD"] <= prev["MACD_SIGNAL"]) and (last["MACD"] > last["MACD_SIGNAL"])
    macd_cross_down = (prev["MACD"] >= prev["MACD_SIGNAL"]) and (last["MACD"] < last["MACD_SIGNAL"])
    near_bb_low = price <= (last["BB_LOW"] * 1.002 if not np.isnan(last["BB_LOW"]) else price)
    near_bb_up = price >= (last["BB_UP"] * 0.998 if not np.isnan(last["BB_UP"]) else price)
    # combine: base + (macd or bb)
    buy_base = above_sma and rsi_rise
    sell_base = below_sma and rsi_fall
    buy_conf = macd_cross_up or near_bb_low
    sell_conf = macd_cross_down or near_bb_up
    if buy_base and buy_conf:
        return "BUY"
    if sell_base and sell_conf:
        return "SELL"
    return "HOLD"

# --------- Session state for last signals (avoid repeated popups) ----------
if "last_signals" not in st.session_state:
    st.session_state["last_signals"] = {}

# --------- Main loop: fetch all pairs, compute & display ----------
st.subheader(f"Monitoring {len(PAIR_LIST)} pairs — refresh every {refresh_sec}s")
cols = st.columns([3,1,1,1])
cols[0].write("Pair")
cols[1].write("Price")
cols[2].write("Signal")
cols[3].write("Updated")

results = []
alerts_js = ""

for pair in PAIR_LIST:
    df, err = fetch_td(pair, interval="1min", outputsize=bars)
    if err:
        results.append((pair, None, "ERROR", err, None))
        continue
    df = add_indicators(df)
    sig = compute_signal(df)
    price = df.iloc[-1]["close"]
    updated = df.iloc[-1]["datetime"].strftime("%Y-%m-%d %H:%M:%S")
    results.append((pair, price, sig, None, updated))

    # popup only if changed and sig in BUY/SELL
    last = st.session_state["last_signals"].get(pair)
    if last != sig and sig in ("BUY", "SELL"):
        safe_pair = html.escape(pair)
        safe_sig = html.escape(sig)
        alerts_js += f"""
        <script>
          setTimeout(function() {{
            alert("Signal {safe_sig} detected for {safe_pair} (1m) - suggested entry 1-3 min");
          }}, 200);
        </script>
        """
        st.session_state["last_signals"][pair] = sig
    else:
        st.session_state["last_signals"].setdefault(pair, sig)

# render results table
for (pair, price, sig, note, updated) in results:
    cols[0].write(pair)
    cols[1].write("" if price is None else f"{price:.5f}")
    if sig == "BUY":
        cols[2].success(sig)
    elif sig == "SELL":
        cols[2].error(sig)
    elif sig == "HOLD":
        cols[2].info(sig)
    else:
        cols[2].write(sig)
    cols[3].write(updated if updated else "")

# render popups
if alerts_js:
    st.components.v1.html(alerts_js, height=0)

# simple log panel (last 50 signals)
st.markdown("---")
st.subheader("Last detected signals (session)")
log_rows = []
for pair, val in st.session_state["last_signals"].items():
    log_rows.append({"pair": pair, "last_signal": val})
log_df = pd.DataFrame(log_rows)
st.table(log_df)

# footer: countdown + auto refresh
st.markdown("---")
placeholder = st.empty()
for i in range(refresh_sec, 0, -1):
    placeholder.info(f"🔁 Next refresh in {i} sec. (Do not close the page)")
    time.sleep(1)
st.experimental_rerun()
