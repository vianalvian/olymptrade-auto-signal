import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime

st.set_page_config(page_title="OlympTrade Smart Auto Signal", layout="wide")

st.title("📊 OlympTrade Smart Auto Signal (1-Minute Timeframe)")
st.markdown("Aplikasi ini membantu trader melihat sinyal **Buy/Sell otomatis** dari berbagai pair di timeframe 1 menit.")

# ======================
# Bagian konfigurasi aset
# ======================
pairs = ["EUR/USD OTC", "AUD/USD OTC", "AUD/CAD OTC", "GBP/USD OTC", "USD/CAD OTC", "USD/CHF OTC", 
          "Asia Composite Index", "Crypto Composite Index", "Europe Composite Index"]

st.sidebar.header("⚙️ Pengaturan Sinyal")
rsi_period = st.sidebar.slider("RSI Period", 5, 20, 14)
sma_period = st.sidebar.slider("SMA Period", 5, 30, 14)
macd_fast = st.sidebar.slider("MACD Fast", 5, 15, 12)
macd_slow = st.sidebar.slider("MACD Slow", 10, 30, 26)
macd_signal = st.sidebar.slider("MACD Signal", 5, 15, 9)
bollinger_period = st.sidebar.slider("Bollinger Period", 10, 30, 20)
auto_refresh = st.sidebar.checkbox("Aktifkan Mode Otomatis", value=True)

# ======================
# Fungsi indikator
# ======================
def RSI(series, period):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def SMA(series, period):
    return series.rolling(window=period).mean()

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
# Simulasi data harga
# ======================
def get_price_data(pair):
    np.random.seed(hash(pair) % 1000000)
    base = np.cumsum(np.random.randn(100)) + 100
    df = pd.DataFrame({"close": base})
    return df

# ======================
# Generate sinyal
# ======================
def generate_signal(pair):
    df = get_price_data(pair)
    df["RSI"] = RSI(df["close"], rsi_period)
    df["SMA"] = SMA(df["close"], sma_period)
    df["MACD"], df["Signal"] = MACD(df["close"], macd_fast, macd_slow, macd_signal)
    df["Upper"], df["Mid"], df["Lower"] = BollingerBands(df["close"], bollinger_period)

    last = df.iloc[-1]
    signal = ""
    if last["close"] > last["SMA"] and last["RSI"] < 30 and last["MACD"] > last["Signal"]:
        signal = "BUY"
    elif last["close"] < last["SMA"] and last["RSI"] > 70 and last["MACD"] < last["Signal"]:
        signal = "SELL"
    else:
        signal = "WAIT"

    return signal, last["RSI"], last["MACD"], last["Signal"]

# ======================
# Tampilan utama
# ======================
data = []
for pair in pairs:
    signal, rsi, macd, sig = generate_signal(pair)
    data.append({"Aset": pair, "Sinyal": signal, "RSI": round(rsi, 2), "MACD": round(macd, 3)})

df_display = pd.DataFrame(data)
st.dataframe(df_display, use_container_width=True)

buy_count = sum(df_display["Sinyal"] == "BUY")
sell_count = sum(df_display["Sinyal"] == "SELL")
st.success(f"Total Sinyal BUY: {buy_count} | Sinyal SELL: {sell_count}")

if auto_refresh:
    st.toast("Mode otomatis aktif — update setiap 60 detik 🔁")
    time.sleep(60)
    st.rerun()
