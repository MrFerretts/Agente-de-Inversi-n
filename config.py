"""
Configuración del Agente de Inversiones
Personaliza estos valores según tus necesidades
"""

# ============= CONFIGURACIÓN DE ACTIVOS =============
PORTFOLIO_CONFIG = {
    'stocks': ['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
    'etfs': [],
    'crypto': ['BTC-USD'],
    'forex': ['GC=F']
}

# ============= INDICADORES TÉCNICOS =============
TECHNICAL_INDICATORS = {
    'sma_short': 20,      # Media móvil simple corta
    'sma_long': 50,       # Media móvil simple larga
    'ema_period': 12,     # Media móvil exponencial
    'rsi_period': 14,     # Período RSI
    'rsi_oversold': 30,   # RSI sobreventa
    'rsi_overbought': 70, # RSI sobrecompra
    'macd_fast': 12,      # MACD línea rápida
    'macd_slow': 26,      # MACD línea lenta
    'macd_signal': 9,     # MACD señal
    'bb_period': 20,      # Bandas de Bollinger período
    'bb_std': 2           # Bandas de Bollinger desviación estándar
}

# ============= ALERTAS DE PRECIO =============
PRICE_ALERTS = {
    'variation_threshold': 5.0,  # % de cambio para alerta
    'volume_spike': 2.0,         # Multiplicador de volumen promedio
    'check_interval': 420        # Segundos entre verificaciones (5 min)
}

# ============= GESTIÓN DE RIESGOS =============
RISK_MANAGEMENT = {
    'max_position_size': 0.10,    # Máximo 10% del portfolio por activo
    'max_sector_exposure': 0.30,  # Máximo 30% por sector
    'stop_loss': 0.05,            # Stop loss 5%
    'take_profit': 0.15,          # Take profit 15%
    'max_daily_trades': 10        # Máximo de operaciones por día
}

# ============= DIVERSIFICACIÓN =============
DIVERSIFICATION_TARGETS = {
    'stocks': 0.00,     # 0% en acciones
    'etfs': 0.00,       # 0% en ETFs
    'crypto': 0.00,     # 0% en cripto
    'forex': 0.00,      # 0% en divisas
    'cash': 0.00        # 0% en efectivo
}

# ============= NOTIFICACIONES =============
NOTIFICATIONS = {
    'email': {
        'enabled': True,
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'sender': 'patoayalai0202@gmail.com',
        'password': 'ghaw wdau sldd zdvq',  # Usar variables de entorno
        'recipient': 'patomty2002@hotmail.com'
    },
    'telegram': {
        'enabled': False,
        'bot_token': '',  # Obtener de @BotFather
        'chat_id': ''     # Tu chat ID
    },
    'console': {
        'enabled': True   # Mostrar en consola por defecto
    }
}

# ============= API KEYS (usar variables de entorno) =============
API_CONFIG = {
    'gemini_api_key': 'AIzaSyAvmSKEG9npd_-WAdHzSVXa8iizB5ITiIE',
    'alpha_vantage': '',  # https://www.alphavantage.co/
    'binance': {
        'api_key': '',
        'api_secret': ''
    },
    'coinmarketcap': ''  # https://coinmarketcap.com/api/
}

# ============= DATA STORAGE =============
DATA_CONFIG = {
    'historical_days': 365,  # Días de historial a mantener
    'update_frequency': 3600,  # Actualizar cada hora
    'database': 'data/portfolio.db'
}

# ============= ANÁLISIS =============
ANALYSIS_CONFIG = {
    'timeframes': ['1d', '1wk', '1mo'],
    'lookback_period': 90,  # Días para análisis
    'min_data_points': 30   # Mínimo de datos para análisis válido
}
