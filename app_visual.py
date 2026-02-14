import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import yfinance as yf
from market_data import MarketDataFetcher
from technical_analysis import TechnicalAnalyzer
import google.generativeai as genai
import requests

# --- FUNCIÓN DE TELEGRAM CORREGIDA ---
def enviar_telegram(mensaje):
    tel_config = NOTIFICATIONS.get('telegram', {})
    if tel_config.get('enabled'):
        token = tel_config.get('bot_token')
        chat_id = tel_config.get('chat_id')
        
        # URL limpia para evitar el "Not Found"
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": mensaje}
        
        try:
            response = requests.post(url, data=payload)
            if response.status_code == 200:
                return True
            else:
                # Si falla, mostramos el error exacto de Telegram
                error_msg = response.json().get('description', 'Error desconocido')
                st.error(f"Telegram dice: {error_msg}")
                return False
        except Exception as e:
            st.error(f"Error de conexión: {e}")
            return False
    return False

# --- PUENTE DE SEGURIDAD ---
try:
    if "API_CONFIG" in st.secrets:
        API_CONFIG = st.secrets["API_CONFIG"]
        PORTFOLIO_CONFIG = st.secrets["PORTFOLIO_CONFIG"]
        TECHNICAL_INDICATORS = st.secrets["TECHNICAL_INDICATORS"]
        NOTIFICATIONS = st.secrets.get("NOTIFICATIONS", {}) 
    else:
        raise Exception("Nube sin secretos")
except:
    try:
        from config import API_CONFIG, PORTFOLIO_CONFIG, TECHNICAL_INDICATORS, NOTIFICATIONS
    except ImportError:
        st.error("❌ Error Crítico: No se encontró configuración.")
        st.stop()

# 1. Funciones de Persistencia
FILE_PATH = "watchlist.json"
def cargar_watchlist():
    if os.path.exists(FILE_PATH):
        with open(FILE_PATH, "r") as f: return json.load(f)
    return {"stocks": PORTFOLIO_CONFIG['stocks'], "crypto": PORTFOLIO_CONFIG['crypto']}

def guardar_watchlist(data_dict):
    with open(FILE_PATH, "w") as f: json.dump(data_dict, f)

# 2. Inicialización
st.set_page_config(page_title="Pato Quant Terminal Pro", layout="wide")
if 'mis_activos' not in st.session_state:
    st.session_state.mis_activos = cargar_watchlist()

fetcher = MarketDataFetcher(API_CONFIG)
analyzer = TechnicalAnalyzer(TECHNICAL_INDICATORS)

# 3. SIDEBAR
st.sidebar.header("🕹️ Gestión de Cartera")
nuevo = st.sidebar.text_input("Añadir Ticker:").upper()
if st.sidebar.button("➕ Agregar"):
    if nuevo and nuevo not in st.session_state.mis_activos['stocks']:
        st.session_state.mis_activos['stocks'].append(nuevo)
        guardar_watchlist(st.session_state.mis_activos)
        st.rerun()

st.sidebar.markdown("---")
lista_completa = st.session_state.mis_activos['stocks'] + st.session_state.mis_activos['crypto']
ticker = st.sidebar.selectbox("Selecciona Activo:", lista_completa)

if st.sidebar.button("🗑️ Eliminar Seleccionado"):
    for cat in ['stocks', 'crypto']:
        if ticker in st.session_state.mis_activos[cat]: st.session_state.mis_activos[cat].remove(ticker)
    guardar_watchlist(st.session_state.mis_activos)
    st.rerun()

# 4. CARGA DE DATOS
data = fetcher.get_portfolio_data([ticker], period='1y')[ticker]

if not data.empty:
    # --- CÁLCULOS TÉCNICOS ---
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    data['RSI_line'] = 100 - (100 / (1 + (gain / loss)))
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD_line'] = ema12 - ema26
    data['MACD_signal'] = data['MACD_line'].ewm(span=9, adjust=False).mean()
    data['MACD_hist'] = data['MACD_line'] - data['MACD_signal']

    tab1, tab2, tab3 = st.tabs(["📊 Análisis en Vivo", "🧪 Backtesting Pro", "📋 Scanner Maestro"])

    with tab1:
        st.title(f"{ticker} - Análisis")
        analysis = analyzer.analyze_asset(data, ticker)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precio", f"${analysis['price']['current']:.2f}")
        m2.metric("RSI", f"{data['RSI_line'].iloc[-1]:.2f}")
        m3.metric("MACD Hist", f"{data['MACD_hist'].iloc[-1]:.2f}")
        m4.metric("Señal", analysis['signals']['recommendation'])
        
        # Gráfica Simple
        st.plotly_chart(go.Figure(data=[go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'])]), use_container_width=True)

    with tab2:
        st.header(f"🧪 Backtesting Pro: {ticker}")
        cap_ini = st.number_input("Capital Inicial ($)", value=10000)
        target_profit, stop_loss = 0.05, 0.02 
        capital, posicion, p_compra, h_cap, trades = cap_ini, 0, 0, [], []

        for i in range(1, len(data)):
            p, rsi, macd, sig = data['Close'].iloc[i], data['RSI_line'].iloc[i], data['MACD_line'].iloc[i], data['MACD_signal'].iloc[i]
            if rsi < 35 and posicion == 0:
                posicion, p_compra, capital = capital / p, p, 0
                trades.append({"Fecha": data.index[i].date(), "Tipo": "🟢 COMPRA", "Precio": round(p, 2), "Motivo": "RSI Bajo"})
            elif posicion > 0:
                rend = (p - p_compra) / p_compra
                if rend >= target_profit or rend <= -stop_loss or (macd < sig and rsi > 50):
                    capital, posicion = posicion * p, 0
                    motivo = "💰 Profit" if rend >= target_profit else "🛡️ StopLoss" if rend <= -stop_loss else "📉 MACD"
                    trades.append({"Fecha": data.index[i].date(), "Tipo": "🔴 VENTA", "Precio": round(p, 2), "Motivo": motivo})
            h_cap.append(capital if posicion == 0 else posicion * p)

        # Mostrar Resultados
        st.plotly_chart(go.Figure(data=[go.Scatter(x=data.index[1:], y=h_cap, name="Capital", fill='tozeroy')]), use_container_width=True)
        st.write("### 📜 Historial de Trades")
        if trades: 
            st.dataframe(pd.DataFrame(trades).sort_values(by="Fecha", ascending=False), use_container_width=True)
            
            # --- EL BOTÓN AHORA ESTÁ DENTRO DE LA PESTAÑA 2 ---
            st.markdown("---")
            st.subheader("📲 Centro de Alertas")
            ultimo = trades[-1]
            alerta_msg = f"🤖 TERMINAL PATO:\nActivo: {ticker}\nSeñal: {ultimo['Tipo']}\nPrecio: ${ultimo['Precio']}\nMotivo: {ultimo['Motivo']}"
            
            if st.button("Enviar última señal a Telegram"):
                if enviar_telegram(alerta_msg):
                    st.success("✅ ¡Alerta enviada!")

    with tab3:
        st.header("📋 Scanner Maestro")
        if st.button("🔍 Iniciar Escaneo"):
            # ... (Tu código de escaneo aquí)
            st.info("Escaneo iniciado...")
