import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import yfinance as yf
from market_data import MarketDataFetcher
from technical_analysis import TechnicalAnalyzer

# --- PUENTE DE SEGURIDAD ---
try:
    if "API_CONFIG" in st.secrets:
        API_CONFIG = st.secrets["API_CONFIG"]
        PORTFOLIO_CONFIG = st.secrets["PORTFOLIO_CONFIG"]
        TECHNICAL_INDICATORS = st.secrets["TECHNICAL_INDICATORS"]
    else:
        raise Exception("Nube sin secretos")
except:
    try:
        from config import API_CONFIG, PORTFOLIO_CONFIG, TECHNICAL_INDICATORS
    except ImportError:
        st.error("❌ Error: No se encontró configuración.")
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

# 3. SIDEBAR: Gestión
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

if st.sidebar.button("🗑️ Eliminar"):
    for cat in ['stocks', 'crypto']:
        if ticker in st.session_state.mis_activos[cat]: st.session_state.mis_activos[cat].remove(ticker)
    guardar_watchlist(st.session_state.mis_activos)
    st.rerun()

# --- CARGA DE DATOS (1 año para backtesting) ---
data = fetcher.get_portfolio_data([ticker], period='1y')[ticker]

if not data.empty:
    # Cálculos Manuales (RSI, MACD, SMA)
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

    # --- PESTAÑAS ---
    tab1, tab2, tab3 = st.tabs(["📊 Análisis en Vivo", "🧪 Backtesting", "📋 Scanner"])

    with tab1:
        st.title(f"Análisis: {ticker}")
        # Aquí va tu código de gráficas y métricas (Omitido por brevedad, es el mismo que ya tienes)
        st.plotly_chart(go.Figure(data=[go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'])]), use_container_width=True)

    with tab2:
        st.header(f"Simulación de Estrategia: {ticker}")
        capital_inicial = st.number_input("Capital Inicial ($)", value=10000)
        
        # Lógica del Backtest
        capital = capital_inicial
        posicion = 0 # 0 = fuera, 1 = comprado
        historial_capital = []
        trades = []

        for i in range(1, len(data)):
            precio = data['Close'].iloc[i]
            rsi = data['RSI_line'].iloc[i]
            macd = data['MACD_line'].iloc[i]
            signal = data['MACD_signal'].iloc[i]

            # SEÑAL COMPRA: RSI < 30
            if rsi < 35 and posicion == 0:
                posicion = capital / precio
                capital = 0
                trades.append({"Fecha": data.index[i], "Tipo": "COMPRA", "Precio": precio})

            # SEÑAL VENTA: MACD cruza abajo
            elif macd < signal and posicion > 0:
                capital = posicion * precio
                posicion = 0
                trades.append({"Fecha": data.index[i], "Tipo": "VENTA", "Precio": precio})
            
            # Tracking del valor de la cuenta
            valor_actual = capital if posicion == 0 else posicion * precio
            historial_capital.append(valor_actual)

        # Resultados del Backtest
        final_val = capital if posicion == 0 else posicion * data['Close'].iloc[-1]
        rendimiento = ((final_val - capital_inicial) / capital_inicial) * 100
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Valor Final", f"${final_val:.2f}")
        c2.metric("Rendimiento", f"{rendimiento:.2f}%")
        c3.metric("Nº Operaciones", len(trades))

        # Gráfica de Crecimiento
        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(x=data.index[1:], y=historial_capital, name="Tu Estrategia", line=dict(color='cyan')))
        fig_equity.update_layout(title="Curva de Capital (1 Año)", template="plotly_dark")
        st.plotly_chart(fig_equity, use_container_width=True)
        
        if trades:
            st.write("### Historial de Trades")
            st.table(pd.DataFrame(trades).tail(10))

    with tab3:
        st.header("Resumen General")
        # Aquí pones tu botón de "Iniciar Escaneo" y la tabla maestra que ya funciona
