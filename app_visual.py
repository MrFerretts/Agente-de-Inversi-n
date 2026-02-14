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
from datetime import datetime
import pytz

# --- FUNCIONES DE APOYO ---
def enviar_telegram(mensaje):
    tel_config = NOTIFICATIONS.get('telegram', {})
    if tel_config.get('enabled'):
        token = tel_config.get('bot_token')
        chat_id = tel_config.get('chat_id')
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": mensaje}
        try:
            response = requests.post(url, data=payload)
            return response.status_code == 200
        except: return False
    return False

def consultar_ia(ticker, precio, rsi, macd, recomendacion):
    try:
        genai.configure(api_key=API_CONFIG['gemini_api_key'])
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"Analiza {ticker}: Precio ${precio:.2f}, RSI {rsi:.2f}, MACD {macd:.2f}, Rec: {recomendacion}. Dame 3 frases de experto quant."
        response = model.generate_content(prompt)
        return response.text
    except: return "⚠️ La IA está meditando..."

# --- PUENTE DE SEGURIDAD ---
try:
    if "API_CONFIG" in st.secrets:
        API_CONFIG, PORTFOLIO_CONFIG = st.secrets["API_CONFIG"], st.secrets["PORTFOLIO_CONFIG"]
        TECHNICAL_INDICATORS = st.secrets["TECHNICAL_INDICATORS"]
        NOTIFICATIONS = st.secrets.get("NOTIFICATIONS", {}) 
    else: raise Exception("Nube sin secretos")
except:
    try: from config import API_CONFIG, PORTFOLIO_CONFIG, TECHNICAL_INDICATORS, NOTIFICATIONS
    except: st.error("❌ Error de configuración."); st.stop()

# 1. Persistencia
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
    if nuevo:
        st.session_state.mis_activos['stocks'].append(nuevo)
        guardar_watchlist(st.session_state.mis_activos); st.rerun()

lista_completa = st.session_state.mis_activos['stocks'] + st.session_state.mis_activos['crypto']
ticker = st.sidebar.selectbox("Selecciona Activo:", lista_completa)

if st.sidebar.button("🗑️ Eliminar Seleccionado"):
    for c in ['stocks', 'crypto']:
        if ticker in st.session_state.mis_activos[c]: st.session_state.mis_activos[c].remove(ticker)
    guardar_watchlist(st.session_state.mis_activos); st.rerun()

# 4. CARGA DE DATOS
data = fetcher.get_portfolio_data([ticker], period='1y')[ticker]

if not data.empty:
    # Cálculos Técnicos
    data['SMA20'] = data['Close'].rolling(20).mean()
    data['SMA50'] = data['Close'].rolling(50).mean()
    std = data['Close'].rolling(20).std()
    data['bb_up'] = data['SMA20'] + (std * 2); data['bb_low'] = data['SMA20'] - (std * 2)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    data['RSI'] = 100 - (100 / (1 + (gain / loss)))
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD_L'] = ema12 - ema26
    data['MACD_S'] = data['MACD_L'].ewm(span=9, adjust=False).mean()
    data['MACD_H'] = data['MACD_L'] - data['MACD_S']

    tab1, tab2, tab3 = st.tabs(["📊 Análisis en Vivo", "🧪 Backtesting Pro", "📋 Scanner Maestro"])

    with tab1:
        # --- TAB 1 RESTAURADA ---
        ana = analyzer.analyze_asset(data, ticker)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precio", f"${ana['price']['current']:.2f}", f"{ana['price']['change_pct']:.2f}%")
        m2.metric("RSI (14)", f"{data['RSI'].iloc[-1]:.2f}")
        m3.metric("MACD Hist", f"{data['MACD_H'].iloc[-1]:.2f}")
        m4.metric("Señal", ana['signals']['recommendation'])

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, 
                            row_heights=[0.5, 0.2, 0.3], subplot_titles=("Precio & Bandas", "RSI", "MACD"))
        fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="Precio"), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['bb_up'], line=dict(color='rgba(173,216,230,0.3)'), name="BB Sup"), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['bb_low'], line=dict(color='rgba(173,216,230,0.3)'), fill='tonexty', name="BB Inf"), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA20'], line=dict(color='orange', width=1), name="SMA 20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA50'], line=dict(color='blue', width=1), name="SMA 50"), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], line=dict(color='purple'), name="RSI"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1); fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
        fig.add_trace(go.Bar(x=data.index, y=data['MACD_H'], marker_color=['green' if x > 0 else 'red' for x in data['MACD_H']], name="MACD"), row=3, col=1)
        fig.update_layout(height=800, template="plotly_dark", showlegend=False, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        
        if st.button("🔮 Consultar al Oráculo IA"):
            with st.spinner("Analizando..."):
                st.info(consultar_ia(ticker, ana['price']['current'], data['RSI'].iloc[-1], data['MACD_H'].iloc[-1], ana['signals']['recommendation']))

    with tab2:
        # --- TAB 2 RESTAURADA ---
        st.header(f"🧪 Resultados de Estrategia: {ticker}")
        cap_ini = st.number_input("Capital Inicial ($)", value=10000)
        t_profit, s_loss = 0.05, 0.02
        capital, posicion, p_compra, h_cap, trades = cap_ini, 0, 0, [], []

        for i in range(1, len(data)):
            p, rsi, macd, sig = data['Close'].iloc[i], data['RSI'].iloc[i], data['MACD_L'].iloc[i], data['MACD_S'].iloc[i]
            if rsi < 35 and posicion == 0:
                posicion, p_compra, capital = capital / p, p, 0
                trades.append({"Fecha": data.index[i].date(), "Tipo": "🟢 COMPRA", "Precio": round(p, 2), "Motivo": "RSI"})
            elif posicion > 0:
                rend = (p - p_compra) / p_compra
                if rend >= t_profit or rend <= -s_loss or (macd < sig and rsi > 50):
                    capital, posicion = posicion * p, 0
                    m = "💰 Profit" if rend >= t_profit else "🛡️ StopLoss" if rend <= -s_loss else "📉 MACD"
                    trades.append({"Fecha": data.index[i].date(), "Tipo": "🔴 VENTA", "Precio": round(p, 2), "Motivo": m})
            h_cap.append(capital if posicion == 0 else posicion * p)

        val_f = capital if posicion == 0 else posicion * data['Close'].iloc[-1]
        rend_t = ((val_f - cap_ini) / cap_ini) * 100
        c1, c2, c3 = st.columns(3)
        c1.metric("Valor Final", f"${val_f:.2f}"); c2.metric("Rendimiento", f"{rend_t:.2f}%"); c3.metric("Trades", len(trades))
        st.plotly_chart(go.Figure(data=[go.Scatter(x=data.index[1:], y=h_cap, name="Capital", fill='tozeroy', line=dict(color='cyan'))]).update_layout(template="plotly_dark"), use_container_width=True)
        
        st.write("### 📜 Bitácora de Operaciones")
        if trades:
            st.dataframe(pd.DataFrame(trades).sort_values(by="Fecha", ascending=False), use_container_width=True)
            ultimo = trades[-1]
            if ultimo['Fecha'] == data.index[-1].date():
                clave_alerta = f"auto_{ticker}_{ultimo['Fecha']}_{ultimo['Tipo']}"
                if clave_alerta not in st.session_state:
                    tz = pytz.timezone('America/Monterrey')
                    ahora = datetime.now(tz).strftime("%d/%m/%Y %H:%M:%S")
                    msg = (f"🚨 SEÑAL EN VIVO - {ahora} 🚨\n\n📈 Activo: {ticker}\n⚡ Acción: {ultimo['Tipo']}\n💵 Precio: ${ultimo['Precio']}\n📍 San Pedro GG, NL")
                    if enviar_telegram(msg): st.session_state[clave_alerta] = True; st.success(f"⚡ Alerta automática enviada ({ahora})")

    with tab3:
        # --- TAB 3: SCANNER COMPLETO + SENTIMIENTO VIX ---
        st.header("📋 Scanner Maestro de 13 Indicadores")
        if st.button("🔍 Iniciar Escaneo de Precisión"):
            res = []
            prog = st.progress(0)
            for i, t in enumerate(lista_completa):
                try:
                    d_r = fetcher.get_portfolio_data([t], period='6mo')[t]
                    if not d_r.empty:
                        d_r['s20'] = d_r['Close'].rolling(20).mean(); d_r['s50'] = d_r['Close'].rolling(50).mean()
                        a_t = analyzer.analyze_asset(d_r, t); ind = a_t['indicators']
                        res.append({
                            "Ticker": t, "Price": round(float(a_t['price']['current']), 2), "Change %": round(float(a_t['price']['change_pct']), 2),
                            "SMA20": round(float(d_r['s20'].iloc[-1]), 2), "SMA50": round(float(d_r['s50'].iloc[-1]), 2),
                            "RSI": round(float(ind.get('rsi', 0)), 2), "stochRSI": round(float(ind.get('stoch_rsi', 0)), 2),
                            "RVOL": round(float(ind.get('rvol', 0)), 2), "ADX": round(float(ind.get('adx', 0)), 2),
                            "ATR": round(float(ind.get('atr', 0)), 2), "MACD_H": round(float(ind.get('macd_hist', 0)), 2),
                            "BB_Up": round(float(ind.get('bb_upper', 0)), 2), "BB_Low": round(float(ind.get('bb_lower', 0)), 2),
                            "Rec": a_t['signals']['recommendation']
                        })
                except: continue
                prog.progress((i + 1) / len(lista_completa))
            st.session_state.ultimo_escaneo = res
            df = pd.DataFrame(res)
            prio = {"COMPRA FUERTE": 0, "COMPRA": 1, "MANTENER": 2, "VENTA": 3, "VENTA FUERTE": 4}
            df['sort'] = df['Rec'].map(prio); df = df.sort_values('sort').drop('sort', axis=1)
            def st_c(v):
                c = '#2ecc71' if 'COMPRA' in v else '#e74c3c' if 'VENTA' in v else '#f1c40f'
                return f'background-color: {c}; color: black; font-weight: bold'
            st.dataframe(df.style.applymap(st_c, subset=['Rec']).format(precision=2), use_container_width=True)

        if 'ultimo_escaneo' in st.session_state:
            st.markdown("---")
            if st.button("📲 Enviar Scanner Completo a Telegram"):
                # Obtenemos el VIX para el análisis de riesgo
                vix = yf.Ticker("^VIX").history(period="1d")['Close'].iloc[-1]
                sentiment = "🟢 RISK ON" if vix < 20 else "🔴 RISK OFF"
                
                tz = pytz.timezone('America/Monterrey')
                ahora = datetime.now(tz).strftime("%d/%m/%Y %H:%M")
                
                msg = f"📉 MARKET SENTIMENT ({ahora})\n"
                msg += f"VIX: {vix:.2f} | {sentiment}\n"
                msg += "--------------------------------\n"
                msg += "📋 SCANNER COMPLETO:\n"
                for item in st.session_state.ultimo_escaneo:
                    msg += f"{item['Ticker']}: ${item['Price']} | {item['Rec']}\n"
                
                if enviar_telegram(msg): st.success("✅ Informe completo enviado")
