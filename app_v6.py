import streamlit as st
import pandas as pd
import numpy as np
import requests
import time

st.set_page_config(page_title="OlympTrade Real Signal v6", layout="wide")

st.title("📊 OlympTrade Smart Signal v6 (Real Data + Auto Mode)")
st.caption("Menggunakan data real-time 1-menit dari TwelveData API")

# ==========================
# SETUP SIDEBAR
# ==========================
st.sidebar.header("⚙️ Pengaturan")
refresh_time = st.sidebar.slider("Refresh (detik)", 30, 180, 60)
mode_auto = st.sidebar.toggle("Mode Otomatis", True)
st.sidebar.write("Pastikan kamu menaruh **TWELVE_API_KEY** di Secrets Streamlit.")

pairs = ["EUR/USD", "GBP/USD", "AUD/USD", "USD/JPY", "USD/CAD", "XAU/USD", "BTC/USD"]

# ==========================
# AMBIL DATA DARI API
# ==========================
def get_price_data(symbol):
    api_key = st.secrets["TWELVE_API_KEY"]
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1min&outputsize=100&apikey={api_key}"
    r = requests.get(url)
    data = r.json()
    df = pd.DataFrame(data["values"])
    df["close"] = df["close"].astype(float)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df

# ==========================
# INDIKATOR
# ==========================
def RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def SMA(series, period=14):
    return series.rolling(period).mean()

def MACD(series, fast=12, slow=26, signal=9):
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

# ==========================
# HASIL SINYAL
# ==========================
def generate_signal(df):
    df["RSI"] = RSI(df["close"])
    df["SMA"] = SMA(df["close"])
    df["MACD"], df["SIGNAL"] = MACD(df["close"])
    df["UPPER"], df["MID"], df["LOWER"] = BollingerBands(df["close"])
    last = df.iloc[-1]

    price = last["close"]
    signal = "HOLD"
    durasi = "-"
    if last["RSI"] < 30 and last["MACD"] > last["SIGNAL"] and price < last["LOWER"]:
        signal, durasi = "BUY", "1 menit"
    elif last["RSI"] > 70 and last["MACD"] < last["SIGNAL"] and price > last["UPPER"]:
        signal, durasi = "SELL", "2 menit"
    return signal, durasi, round(price, 5), round(last["RSI"], 2)

# ==========================
# LOOP UTAMA
# ==========================
placeholder = st.empty()

while True:
    data = []
    alerts = []
    for p in pairs:
        df = get_price_data(p.replace("/", ""))
        signal, durasi, price, rsi = generate_signal(df)
        data.append([p, signal, durasi, price, rsi])
        if signal in ["BUY", "SELL"]:
            alerts.append(f"🚨 {p}: {signal} ({durasi})")

    df_show = pd.DataFrame(data, columns=["Pair", "Signal", "Durasi", "Harga", "RSI"])
    with placeholder.container():
        st.dataframe(df_show, use_container_width=True)
        for a in alerts:
            st.toast(a, icon="⚡")

        if mode_auto:
            st.info(f"Auto refresh dalam {refresh_time} detik...")
        else:
            st.warning("Mode Manual Aktif. Klik tombol di bawah untuk update.")
            if st.button("🔄 Perbarui Sekarang"):
                st.rerun()

    if not mode_auto:
        break
    time.sleep(refresh_time)
    st.rerun()
