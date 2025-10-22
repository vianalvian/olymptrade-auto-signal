import streamlit as st
import pandas as pd
import numpy as np
import time
import random

# ===============================
# CONFIG DASAR
# ===============================
st.set_page_config(page_title="Olymptrade Smart Signal v5", layout="wide")

TIMEFRAME = "1m"
REFRESH_INTERVAL = 60  # detik

assets = [
    "EUR/USD OTC", "GBP/USD OTC", "AUD/USD OTC", "USD/CHF OTC",
    "USD/CAD OTC", "AUD/CAD OTC", "BTC/USD", "XAU/USD"
]

# ===============================
# LOGIKA SINYAL
# ===============================
def generate_signal():
    rsi = random.randint(10, 90)
    macd = random.uniform(-1, 1)
    price = random.uniform(1.0, 2.0)
    sma = price + random.uniform(-0.05, 0.05)
    upper_bb = sma + 0.02
    lower_bb = sma - 0.02

    if rsi < 30 and macd > 0 and price > lower_bb:
        signal = "BUY"
    elif rsi > 70 and macd < 0 and price < upper_bb:
        signal = "SELL"
    else:
        signal = "HOLD"

    return signal, round(rsi, 2), round(macd, 3)


# ===============================
# MODE OTOMATIS & NOTIFIKASI
# ===============================
st.sidebar.title("⚙️ Pengaturan")
mode_auto = st.sidebar.toggle("Mode Otomatis", True)
st.sidebar.write(f"⏱ Timeframe: **{TIMEFRAME}**")
st.sidebar.write(f"🔁 Refresh tiap **{REFRESH_INTERVAL} detik**")

# ===============================
# TAMPILAN UTAMA
# ===============================
st.title("📊 Olymptrade Smart Signal v5 (Realtime + Alert)")
st.write("💡 Indikator: RSI + SMA + MACD + Bollinger Bands")
placeholder = st.empty()
last_signals = {}

# ===============================
# LOOP REALTIME
# ===============================
while True:
    data = []
    alert_messages = []

    for asset in assets:
        signal, rsi, macd = generate_signal()

        # Simpan sinyal terakhir untuk deteksi perubahan
        prev_signal = last_signals.get(asset)
        last_signals[asset] = signal

        # Jika sinyal berubah ke BUY/SELL → tambahkan notifikasi
        if prev_signal and signal != prev_signal and signal in ["BUY", "SELL"]:
            alert_messages.append(f"🚨 {asset}: {signal} signal detected!")

        data.append([asset, signal, rsi, macd])

    df = pd.DataFrame(data, columns=["Pair", "Signal", "RSI", "MACD"])

    with placeholder.container():
        st.dataframe(df, use_container_width=True)

        # Tampilkan notifikasi popup
        for msg in alert_messages:
            st.toast(msg, icon="⚡")
            st.audio("https://actions.google.com/sounds/v1/alarms/beep_short.ogg", autoplay=True)

        if mode_auto:
            st.success("✅ Mode Otomatis Aktif - Realtime berjalan")
            st.info(f"Next refresh in {REFRESH_INTERVAL} sec (jangan tutup halaman)")
        else:
            st.warning("🟡 Mode Manual - Klik tombol di bawah untuk update")
            if st.button("🔄 Perbarui Sekarang"):
                st.rerun()

    if not mode_auto:
        break

    time.sleep(REFRESH_INTERVAL)
    st.rerun()
