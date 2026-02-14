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

# --- FUNCIÓN DEL ORÁCULO IA ---
def consultar_ia(ticker, precio, rsi, macd, recomendacion):
    try:
        genai.configure(api_key=API_CONFIG['gemini_api_key'])
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        Actúa como un experto analista quant. 
        Analiza el activo {ticker} con estos datos:
        - Precio actual: ${precio:.2f}
        - RSI (14): {rsi:.2f}
        - MACD Histograma: {macd:.2f}
        - Recomendación técnica: {recomendacion}
        Dame un análisis de 3 oraciones sobre si es buen momento para entrar o no y por qué.
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return "⚠️ La IA está meditando (revisa tu API Key en Secrets)."

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

if st.sidebar.button("🗑️ Eliminar Seleccionado"):
    for cat in ['stocks', 'crypto']:
        if ticker in st.session_state.mis_activos[cat]: st.session_state.mis_activos[cat].remove(ticker)
    guardar_watchlist(st.session_state.mis_activos)
    st.rerun()

@st.cache_data(ttl=86400)
def get_full_name(symbol):
    try:
        t = yf.Ticker(symbol)
        return t.info.get('longName', symbol)
    except: return symbol

# 4. CARGA DE DATOS
data = fetcher.get_portfolio_data([ticker], period='1y')[ticker]

if not data.empty:
    # Cálculos Técnicos
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    std = data['Close'].rolling(window=20).std()
    data['bb_upper'] = data['SMA20'] + (std * 2)
    data['bb_lower'] = data['SMA20'] - (std * 2)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    data['RSI_line'] = 100 - (100 / (1 + (gain / loss)))
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD_line'] = ema12 - ema26
    data['MACD_signal'] = data['MACD_line'].ewm(span=9, adjust=False).mean()
    data['MACD_hist'] = data['MACD_line'] - data['MACD_signal']

    tab1, tab2, tab3 = st.tabs(["📊 Análisis en Vivo", "🧪 Backtesting Histórico", "📋 Scanner Maestro"])

    with tab1:
        st.title(f"{get_full_name(ticker)} ({ticker})")
        analysis = analyzer.analyze_asset(data, ticker)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precio", f"${analysis['price']['current']:.2f}", f"{analysis['price']['change_pct']:.2f}%")
        m2.metric("RSI", f"{data['RSI_line'].iloc[-1]:.2f}")
        m3.metric("MACD Hist", f"{data['MACD_hist'].iloc[-1]:.2f}")
        m4.metric("Señal", analysis['signals']['recommendation'])

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, 
                            row_heights=[0.5, 0.2, 0.3], subplot_titles=("Precio & Bandas", "RSI", "MACD"))
        fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="Precio"), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['bb_upper'], line=dict(color='rgba(173,216,230,0.3)'), name="BB Sup"), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['bb_lower'], line=dict(color='rgba(173,216,230,0.3)'), fill='tonexty', name="BB Inf"), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA20'], line=dict(color='orange', width=1), name="SMA 20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA50'], line=dict(color='blue', width=1), name="SMA 50"), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['RSI_line'], line=dict(color='purple'), name="RSI"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1); fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
        fig.add_trace(go.Bar(x=data.index, y=data['MACD_hist'], marker_color=['green' if x > 0 else 'red' for x in data['MACD_hist']], name="MACD Hist"), row=3, col=1)
        fig.update_layout(height=700, template="plotly_dark", showlegend=False, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("🤖 El Oráculo IA")
        if st.button("🔮 Consultar a Gemini"):
            with st.spinner("Analizando mercado..."):
                opinion = consultar_ia(ticker, analysis['price']['current'], data['RSI_line'].iloc[-1], data['MACD_hist'].iloc[-1], analysis['signals']['recommendation'])
                st.info(opinion)

    with tab2:
        st.header(f"🧪 Backtesting Completo: {ticker}")
        cap_ini = st.number_input("Capital Inicial ($)", value=10000)
        capital, posicion, h_cap, trades = cap_ini, 0, [], []
        for i in range(1, len(data)):
            p, rsi, macd, sig = data['Close'].iloc[i], data['RSI_line'].iloc[i], data['MACD_line'].iloc[i], data['MACD_signal'].iloc[i]
            if rsi < 35 and posicion == 0:
                posicion, capital = capital / p, 0
                trades.append({"Fecha": data.index[i].date(), "Tipo": "🟢 COMPRA", "Precio": round(p, 2), "Capital": round(posicion * p, 2)})
            elif macd < sig and posicion > 0:
                capital, posicion = posicion * p, 0
                trades.append({"Fecha": data.index[i].date(), "Tipo": "🔴 VENTA", "Precio": round(p, 2), "Capital": round(capital, 2)})
            h_cap.append(capital if posicion == 0 else posicion * p)
        val_final = capital if posicion == 0 else posicion * data['Close'].iloc[-1]
        rend = ((val_final - cap_ini) / cap_ini) * 100
        c1, c2, c3 = st.columns(3)
        c1.metric("Valor Final", f"${val_final:.2f}"); c2.metric("Rendimiento", f"{rend:.2f}%"); c3.metric("Trades", len(trades))
        st.plotly_chart(go.Figure(data=[go.Scatter(x=data.index[1:], y=h_cap, name="Capital", fill='tozeroy', line=dict(color='cyan'))]).update_layout(title="Curva de Capital (1 Año)", template="plotly_dark"), use_container_width=True)
        st.write("### 📜 Bitácora de Operaciones Completa")
        if trades: st.dataframe(pd.DataFrame(trades).sort_values(by="Fecha", ascending=False), use_container_width=True)

    with tab3:
        st.header("📋 Scanner Maestro de 13 Indicadores")
        if st.button("🔍 Iniciar Escaneo de Precisión"):
            res_lista = []
            prog = st.progress(0)
            for i, t in enumerate(lista_completa):
                try:
                    d_r = fetcher.get_portfolio_data([t], period='6mo')[t]
                    if not d_r.empty:
                        d_r['s20'] = d_r['Close'].rolling(20).mean(); d_r['s50'] = d_r['Close'].rolling(50).mean()
                        ana_t = analyzer.analyze_asset(d_r, t); ind = ana_t['indicators']
                        res_lista.append({
                            "Ticker": t, "Price": round(float(ana_t['price']['current']), 2), "Change %": round(float(ana_t['price']['change_pct']), 2),
                            "SMA20": round(float(d_r['s20'].iloc[-1]), 2), "SMA50": round(float(d_r['s50'].iloc[-1]), 2),
                            "RSI": round(float(ind.get('rsi', 0)), 2), "stochRSI": round(float(ind.get('stoch_rsi', 0)), 2),
                            "RVOL": round(float(ind.get('rvol', 0)), 2), "ADX": round(float(ind.get('adx', 0)), 2),
                            "ATR": round(float(ind.get('atr', 0)), 2), "MACD_H": round(float(ind.get('macd_hist', 0)), 2),
                            "BB_Up": round(float(ind.get('bb_upper', 0)), 2), "BB_Low": round(float(ind.get('bb_lower', 0)), 2),
                            "Rec": ana_t['signals']['recommendation']
                        })
                except: continue
                prog.progress((i + 1) / len(lista_completa))
            df = pd.DataFrame(res_lista)
            prio = {"COMPRA FUERTE": 0, "COMPRA": 1, "MANTENER": 2, "VENTA": 3, "VENTA FUERTE": 4}
            df['sort'] = df['Rec'].map(prio); df = df.sort_values('sort').drop('sort', axis=1)
            def st_rec(v):
                c = '#2ecc71' if 'COMPRA' in v else '#e74c3c' if 'VENTA' in v else '#f1c40f'
                return f'background-color: {c}; color: black; font-weight: bold'
            st.dataframe(df.style.applymap(st_rec, subset=['Rec']).format(precision=2), use_container_width=True)
