import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import yfinance as yf
import requests
import google.generativeai as genai
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import pytz
from market_data import MarketDataFetcher
from technical_analysis import TechnicalAnalyzer

# --- FUNCIONES DE NOTIFICACIÓN Y APOYO ---
def enviar_telegram(mensaje):
    """Envía alertas cortas de backtesting a Telegram."""
    tel_config = NOTIFICATIONS.get('telegram', {})
    if tel_config.get('enabled'):
        token = tel_config.get('bot_token')
        chat_id = tel_config.get('chat_id')
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": mensaje}
        try:
            requests.post(url, data=payload)
            return True
        except: return False
    return False

def enviar_correo_maestro(vix_val, sentiment, df_html):
    """Envía el reporte completo del scanner por correo en formato Dark Mode."""
    gmail_config = NOTIFICATIONS.get('gmail', {})
    if not gmail_config.get('enabled'): return False
        
    sender = gmail_config['user']
    receiver = gmail_config['destinatario']
    pwd = gmail_config['password']

    msg = MIMEMultipart("alternative")
    tz = pytz.timezone('America/Monterrey')
    ahora = datetime.now(tz).strftime("%d/%m/%Y %H:%M")
    
    msg["Subject"] = f"🚀 REPORTE ESTRATÉGICO - {ahora}"
    msg["From"] = sender
    msg["To"] = receiver

    html = f"""
    <html>
      <body style="font-family: Arial, sans-serif; background-color: #121212; color: #ffffff; padding: 20px;">
        <h2 style="color: #2ecc71;">📊 Informe Quant: {sentiment}</h2>
        <p><b>Indicador VIX:</b> {vix_val:.2f} | <b>Ubicación:</b> San Pedro GG, NL</p>
        <hr style="border: 0.5px solid #333;">
        <div style="overflow-x: auto;">
            {df_html}
        </div>
        <p style="font-size: 11px; color: #666; margin-top: 20px;">Generado por Pato Quant Terminal Pro.</p>
      </body>
    </html>
    """
    msg.attach(MIMEText(html, "html"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, pwd)
            server.sendmail(sender, receiver, msg.as_string())
        return True
    except: return False

def consultar_ia(ticker, precio, rsi, macd, recomendacion):
    """Consulta al Oráculo Gemini para análisis narrativo."""
    try:
        genai.configure(api_key=API_CONFIG['gemini_api_key'])
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"Como analista quant, analiza {ticker} (Precio: {precio}, RSI: {rsi}, MACD: {macd}, Rec: {recomendacion}) en 3 frases."
        return model.generate_content(prompt).text
    except: return "⚠️ El Oráculo está fuera de línea."

# --- SEGURIDAD Y CONFIGURACIÓN ---
try:
    if "API_CONFIG" in st.secrets:
        API_CONFIG, PORTFOLIO_CONFIG = st.secrets["API_CONFIG"], st.secrets["PORTFOLIO_CONFIG"]
        TECHNICAL_INDICATORS = st.secrets["TECHNICAL_INDICATORS"]
        NOTIFICATIONS = st.secrets.get("NOTIFICATIONS", {})
    else: raise Exception("Sin secretos")
except:
    try: from config import API_CONFIG, PORTFOLIO_CONFIG, TECHNICAL_INDICATORS, NOTIFICATIONS
    except: st.error("❌ Fallo de configuración"); st.stop()

# 1. Watchlist
FILE_PATH = "watchlist.json"
def cargar_watchlist():
    if os.path.exists(FILE_PATH):
        with open(FILE_PATH, "r") as f: return json.load(f)
    return {"stocks": PORTFOLIO_CONFIG['stocks'], "crypto": PORTFOLIO_CONFIG['crypto']}

def guardar_watchlist(data_dict):
    with open(FILE_PATH, "w") as f: json.dump(data_dict, f)

st.set_page_config(page_title="Pato Quant Terminal Pro", layout="wide")
if 'mis_activos' not in st.session_state:
    st.session_state.mis_activos = cargar_watchlist()

fetcher = MarketDataFetcher(API_CONFIG)
analyzer = TechnicalAnalyzer(TECHNICAL_INDICATORS)

# 2. Sidebar
st.sidebar.header("🕹️ Panel de Control")
nuevo = st.sidebar.text_input("Añadir Ticker:").upper()
if st.sidebar.button("➕ Agregar"):
    if nuevo:
        st.session_state.mis_activos['stocks'].append(nuevo)
        guardar_watchlist(st.session_state.mis_activos); st.rerun()

lista_completa = st.session_state.mis_activos['stocks'] + st.session_state.mis_activos['crypto']
ticker = st.sidebar.selectbox("Activo a analizar:", lista_completa)

if st.sidebar.button("🗑️ Eliminar Seleccionado"):
    for c in ['stocks', 'crypto']:
        if ticker in st.session_state.mis_activos[c]: st.session_state.mis_activos[c].remove(ticker)
    guardar_watchlist(st.session_state.mis_activos); st.rerun()

# 3. Datos y Cálculos
data = fetcher.get_portfolio_data([ticker], period='1y')[ticker]

if not data.empty:
    # Cálculos para Tab 1 y Tab 2
    data['SMA20'] = data['Close'].rolling(20).mean()
    data['SMA50'] = data['Close'].rolling(50).mean()
    std = data['Close'].rolling(20).std()
    data['bb_up'] = data['SMA20'] + (std * 2); data['bb_low'] = data['SMA20'] - (std * 2)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    data['RSI'] = 100 - (100 / (1 + (gain / loss)))
    ema12, ema26 = data['Close'].ewm(span=12).mean(), data['Close'].ewm(span=26).mean()
    data['MACD_L'] = ema12 - ema26
    data['MACD_S'] = data['MACD_L'].ewm(span=9).mean()
    data['MACD_H'] = data['MACD_L'] - data['MACD_S']

    tab1, tab2, tab3 = st.tabs(["📊 Análisis en Vivo", "🧪 Backtesting Pro", "📋 Scanner Maestro"])

    with tab1:
        # ANÁLISIS VISUAL RESTAURADO
        ana = analyzer.analyze_asset(data, ticker)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precio", f"${ana['price']['current']:.2f}", f"{ana['price']['change_pct']:.2f}%")
        m2.metric("RSI (14)", f"{data['RSI'].iloc[-1]:.2f}")
        m3.metric("MACD Hist", f"{data['MACD_H'].iloc[-1]:.2f}")
        m4.metric("Señal", ana['signals']['recommendation'])

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, 
                            row_heights=[0.5, 0.2, 0.3], subplot_titles=("Precio", "RSI", "MACD"))
        fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close']), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['bb_up'], line=dict(color='rgba(173,216,230,0.2)')), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['bb_low'], line=dict(color='rgba(173,216,230,0.2)'), fill='tonexty'), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA20'], line=dict(color='orange')), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], line=dict(color='purple')), row=2, col=1)
        fig.add_trace(go.Bar(x=data.index, y=data['MACD_H'], marker_color=['green' if x > 0 else 'red' for x in data['MACD_H']]), row=3, col=1)
        fig.update_layout(height=800, template="plotly_dark", showlegend=False, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        
        if st.button("🔮 Consultar Oráculo"):
            st.info(consultar_ia(ticker, ana['price']['current'], data['RSI'].iloc[-1], data['MACD_H'].iloc[-1], ana['signals']['recommendation']))

    with tab2:
        # BACKTESTING CON MÉTRICAS Y TABLA
        cap_ini = st.number_input("Capital Inicial ($)", value=10000)
        t_profit, s_loss = 0.05, 0.02
        cap, pos, p_com, h_cap, trades = cap_ini, 0, 0, [], []

        for i in range(1, len(data)):
            p, rsi, macd, sig = data['Close'].iloc[i], data['RSI'].iloc[i], data['MACD_L'].iloc[i], data['MACD_S'].iloc[i]
            if rsi < 35 and pos == 0:
                pos, p_com, cap = cap / p, p, 0
                trades.append({"Fecha": data.index[i].date(), "Tipo": "🟢 COMPRA", "Precio": round(p, 2), "Motivo": "RSI"})
            elif pos > 0:
                rend = (p - p_com) / p_com
                if rend >= t_profit or rend <= -s_loss or (macd < sig and rsi > 50):
                    cap, pos = pos * p, 0
                    m = "💰 Profit" if rend >= t_profit else "🛡️ StopLoss" if rend <= -s_loss else "📉 MACD"
                    trades.append({"Fecha": data.index[i].date(), "Tipo": "🔴 VENTA", "Precio": round(p, 2), "Motivo": m})
            h_cap.append(cap if pos == 0 else pos * p)

        val_f = cap if pos == 0 else pos * data['Close'].iloc[-1]
        rend_t = ((val_f - cap_ini) / cap_ini) * 100
        c1, c2, c3 = st.columns(3)
        c1.metric("Valor Final", f"${val_f:.2f}"); c2.metric("Rendimiento", f"{rend_t:.2f}%"); c3.metric("Trades", len(trades))
        st.plotly_chart(go.Figure(data=[go.Scatter(x=data.index[1:], y=h_cap, fill='tozeroy', line=dict(color='cyan'))]).update_layout(template="plotly_dark"), use_container_width=True)
        
        if trades:
            st.dataframe(pd.DataFrame(trades).sort_values(by="Fecha", ascending=False), use_container_width=True)
            # Alerta Telegram Automática
            u = trades[-1]
            if u['Fecha'] == data.index[-1].date():
                tz = pytz.timezone('America/Monterrey')
                ahora = datetime.now(tz).strftime("%H:%M")
                if enviar_telegram(f"🤖 SEÑAL ({ahora}): {ticker} {u['Tipo']} @ ${u['Precio']}"):
                    st.success(f"Alerta enviada a las {ahora}")

    with tab3:
        # SCANNER MAESTRO Y ENVÍO POR CORREO
        st.header("📋 Scanner Maestro")
        if st.button("🔍 Iniciar Escaneo"):
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
                            "RSI": round(float(ind.get('rsi', 0)), 2), "ADX": round(float(ind.get('adx', 0)), 2),
                            "ATR": round(float(ind.get('atr', 0)), 2), "MACD_H": round(float(ind.get('macd_hist', 0)), 2),
                            "BB_Up": round(float(ind.get('bb_upper', 0)), 2), "BB_Low": round(float(ind.get('bb_lower', 0)), 2),
                            "Rec": a_t['signals']['recommendation']
                        })
                except: continue
                prog.progress((i + 1) / len(lista_completa))
            st.session_state.df_scan = pd.DataFrame(res)
            
            def st_c(v):
                c = '#2ecc71' if 'COMPRA' in v else '#e74c3c' if 'VENTA' in v else '#f1c40f'
                return f'background-color: {c}; color: black; font-weight: bold'
            st.dataframe(st.session_state.df_scan.style.applymap(st_c, subset=['Rec']).format(precision=2), use_container_width=True)

        if 'df_scan' in st.session_state:
            st.markdown("---")
            if st.button("📧 Enviar Reporte Completo a mi Correo"):
                vix = yf.Ticker("^VIX").history(period="1d")['Close'].iloc[-1]
                sentiment = "🟢 RISK ON" if vix < 20 else "🔴 RISK OFF"
                df_html = st.session_state.df_scan.to_html(index=False)
                # Estilo Dark para el correo
                df_html = df_html.replace('table', 'table border="1" style="border-collapse: collapse; width: 100%; color: white; background-color: #1e1e1e;"')
                if enviar_correo_maestro(vix, sentiment, df_html):
                    st.success("✅ Reporte completo enviado a Gmail")
