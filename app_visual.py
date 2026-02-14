import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import yfinance as yf
from market_data import MarketDataFetcher
from technical_analysis import TechnicalAnalyzer
from config import *

# 1. Funciones de Persistencia
FILE_PATH = "watchlist.json"

def cargar_watchlist():
    if os.path.exists(FILE_PATH):
        with open(FILE_PATH, "r") as f:
            return json.load(f)
    return {"stocks": PORTFOLIO_CONFIG['stocks'], "crypto": PORTFOLIO_CONFIG['crypto']}

def guardar_watchlist(data_dict):
    with open(FILE_PATH, "w") as f:
        json.dump(data_dict, f)

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
ticker = st.sidebar.selectbox("Analizar en detalle:", lista_completa)

@st.cache_data(ttl=86400)
def get_full_name(symbol):
    try:
        t = yf.Ticker(symbol)
        return t.info.get('longName', symbol)
    except: return symbol

nombre_completo = get_full_name(ticker)
st.title(f"📊 {nombre_completo} ({ticker})")

# 4. Análisis Detallado
data = fetcher.get_portfolio_data([ticker], period='6mo')[ticker]

if not data.empty:
    # Cálculos manuales para asegurar gráficas correctas
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    std = data['Close'].rolling(window=20).std()
    data['bb_upper'] = data['SMA20'] + (std * 2)
    data['bb_lower'] = data['SMA20'] - (std * 2)
    
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI_line'] = 100 - (100 / (1 + rs))

    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD_line'] = ema12 - ema26
    data['MACD_signal'] = data['MACD_line'].ewm(span=9, adjust=False).mean()
    data['MACD_hist'] = data['MACD_line'] - data['MACD_signal']

    # Métricas Superiores con formato de 2 decimales
    analysis = analyzer.analyze_asset(data, ticker)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precio", f"${analysis['price']['current']:.2f}", f"{analysis['price']['change_pct']:.2f}%")
    m2.metric("RSI", f"{data['RSI_line'].iloc[-1]:.2f}")
    m3.metric("MACD Hist", f"{data['MACD_hist'].iloc[-1]:.2f}")
    m4.metric("Señal", analysis['signals']['recommendation'])

    # Gráfica
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, 
                        row_heights=[0.5, 0.2, 0.3], subplot_titles=("Precio & Bandas", "RSI", "MACD"))
    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="Precio"), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['bb_upper'], line=dict(color='rgba(173,216,230,0.3)'), name="BB Sup"), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['bb_lower'], line=dict(color='rgba(173,216,230,0.3)'), fill='tonexty', name="BB Inf"), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA20'], line=dict(color='orange', width=1), name="SMA 20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA50'], line=dict(color='blue', width=1), name="SMA 50"), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['RSI_line'], line=dict(color='purple'), name="RSI"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
    fig.add_trace(go.Bar(x=data.index, y=data['MACD_hist'], marker_color=['green' if x > 0 else 'red' for x in data['MACD_hist']], name="MACD Hist"), row=3, col=1)
    fig.update_layout(height=700, template="plotly_dark", showlegend=False, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# --- SECCIÓN: TABLA MAESTRA CON CORRECCIÓN DE SMAS Y FORMATO ---
st.markdown("---")
st.header("📋 Scanner Maestro (Formato de Precisión)")

if st.button("🔍 Iniciar Escaneo"):
    resumen_lista = []
    progreso = st.progress(0)
    
    for i, t in enumerate(lista_completa):
        try:
            d_res = fetcher.get_portfolio_data([t], period='6mo')[t]
            if not d_res.empty:
                # CORRECCIÓN: Calculamos SMAs manuales antes de pasar al analyzer
                d_res['sma_20_calc'] = d_res['Close'].rolling(window=20).mean()
                d_res['sma_50_calc'] = d_res['Close'].rolling(window=50).mean()
                
                analisis_t = analyzer.analyze_asset(d_res, t)
                ind = analisis_t['indicators']
                
                resumen_lista.append({
                    "Ticker": t,
                    "Price": round(float(analisis_t['price']['current']), 2),
                    "Change %": round(float(analisis_t['price']['change_pct']), 2),
                    "SMA20": round(float(d_res['sma_20_calc'].iloc[-1]), 2),
                    "SMA50": round(float(d_res['sma_50_calc'].iloc[-1]), 2),
                    "RSI": round(float(ind.get('rsi', 0)), 2),
                    "stochRSI": round(float(ind.get('stoch_rsi', 0)), 2),
                    "RVOL": round(float(ind.get('rvol', 0)), 2),
                    "ADX": round(float(ind.get('adx', 0)), 2),
                    "ATR": round(float(ind.get('atr', 0)), 2),
                    "MACD_H": round(float(ind.get('macd_hist', 0)), 2),
                    "BB_Up": round(float(ind.get('bb_upper', 0)), 2),
                    "BB_Low": round(float(ind.get('bb_lower', 0)), 2),
                    "Rec": analisis_t['signals']['recommendation']
                })
        except: continue
        progreso.progress((i + 1) / len(lista_completa))
    
    df_resumen = pd.DataFrame(resumen_lista)
    
    # Ordenar por oportunidad
    prioridad = {"COMPRA FUERTE": 0, "COMPRA": 1, "MANTENER": 2, "VENTA": 3, "VENTA FUERTE": 4}
    df_resumen['sort_idx'] = df_resumen['Rec'].map(prioridad)
    df_resumen = df_resumen.sort_values('sort_idx').drop('sort_idx', axis=1)

    # Estilo visual de la tabla
    def style_rec(val):
        color = '#2ecc71' if 'COMPRA' in val else '#e74c3c' if 'VENTA' in val else '#f1c40f'
        return f'background-color: {color}; color: black; font-weight: bold'

    # Mostramos la tabla formateada
    st.dataframe(df_resumen.style.applymap(style_rec, subset=['Rec']).format(precision=2), use_container_width=True)