import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime

st.set_page_config(page_title="OlympTrade Auto Signal v2", layout="wide")

st.title("📊 OlympTrade Auto Signal v2 (1-Minute Timeframe)")
st.caption("Indikator gabungan RSI, SMA, MACD & Bollinger Bands — Auto update + popup notifikasi")

# ======================
# Pair / aset OTC
# ======================
pairs = [
    "EUR/USD OTC", "GBP/USD OTC", "AUD/USD OTC", "USD/CHF OTC",
    "USD/CAD OTC", "AUD/CAD OTC", "Crypto Composite Index", "Asia Composite Index"
]

# ======================
# Sidebar Settings
# ======================
st.sidebar.header("⚙️ Pengaturan Indikator")
rsi_period = st.sidebar.slider("RSI Period", 5, 20, 14)
sma_period = st.sidebar.slider("SMA Period", 5, 30, 14)
macd_fast = st.sidebar.slider("MACD Fast", 5, 15, 12)
macd_slow = st.sidebar.slider("MACD Slow", 10, 30, 26)
macd_signal = st.sidebar.slider("MACD Signal", 5, 15, 9)
bollinger_period = st.sidebar.slider("Bollinger Period", 10, 30, 20)
refresh_time = st.sidebar.slider("Waktu Refresh Otomatis (detik)", 30, 120, 60)

# ======================
# Fungsi indikator
# ======================
def RSI(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def SMA(series, period):
    return series.rolling(period).mean()

def MACD(series, fast, slow, signal):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def BollingerBands(series, period=20, std_factor=2):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + (std_factor * std)
    lower = sma - (std_factor * std)
    return upper, sma, lower

# ======================
# Data simulasi harga
# ======================
def get_price_data(pair):
    np.random.seed(hash(pair) % 1000000)
    base = np.cumsum(np.random.randn(120)) + 100
    df = pd.DataFrame({"close": base})
    return df

# ======================
# Logika sinyal
# ======================
def generate_signal(pair):
    df = get_price_data(pair)
    df["RSI"] = RSI(df["close"], rsi_period)
    df["SMA"] = SMA(df["close"], sma_period)
    df["MACD"], df["Signal"] = MACD(df["close"], macd_fast, macd_slow, macd_signal)
    df["Upper"], df["Mid"], df["Lower"] = BollingerBands(df["close"], bollinger_period)

    last = df.iloc[-1]
    price = last["close"]

    signal = "WAIT"
    if last["RSI"] < 30 and price < last["Lower"] and last["MACD"] > last["Signal"] and price > last["SMA"]:
        signal = "BUY"
    elif last["RSI"] > 70 and price > last["Upper"] and last["MACD"] < last["Signal"] and price < last["SMA"]:
        signal = "SELL"

    return signal, last["RSI"], price

# ======================
# Tampilan utama
# ======================
data = []
buy_alerts, sell_alerts = [], []

for pair in pairs:
    signal, rsi, price = generate_signal(pair)
    data.append({"Aset": pair, "Harga": round(price, 3), "RSI": round(rsi, 2), "Sinyal": signal})
    if signal == "BUY":
        buy_alerts.append(pair)
    elif signal == "SELL":
        sell_alerts.append(pair)

df_display = pd.DataFrame(data)
st.dataframe(df_display, use_container_width=True)

col1, col2 = st.columns(2)
col1.success(f"🟢 Total Sinyal BUY: {len(buy_alerts)}")
col2.error(f"🔴 Total Sinyal SELL: {len(sell_alerts)}")

# ======================
# Popup notifikasi
# ======================
if buy_alerts:
    st.toast(f"🟢 Sinyal BUY terdeteksi: {', '.join(buy_alerts)}", icon="✅")

if sell_alerts:
    st.toast(f"🔴 Sinyal SELL terdeteksi: {', '.join(sell_alerts)}", icon="⚠️")

# ======================
# Countdown dan auto refresh
# ======================
st.markdown("---")
placeholder = st.empty()
for i in range(refresh_time, 0, -1):
    placeholder.info(f"⏱ Update otomatis dalam {i} detik...")
    time.sleep(1)
st.rerun()
