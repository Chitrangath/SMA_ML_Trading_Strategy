import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Page setup
st.set_page_config(page_title="SMA ML Trading Bot", page_icon="📈", layout="wide")
st.title("🤖 SMA ML Crossover Trading Dashboard")
st.markdown("A machine learning-enhanced SMA crossover strategy for crypto, indices, and gold.")

# Sidebar controls
st.sidebar.header("Settings")
symbol = st.sidebar.selectbox("Symbol", ["BTC-USD", "ETH-USD", "GC=F", "NQ=F"])
period = st.sidebar.selectbox("Data Period", ["1mo", "3mo", "6mo", "1y"], index=2)
interval = st.sidebar.selectbox("Timeframe", ["15m", "1h", "4h", "1d"], index=1)

if st.sidebar.button("🚀 Run Strategy"):
    st.info("Running ML strategy on latest data...")
    # Download Data
    data = yf.download(symbol, period=period, interval=interval, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    data['sma8'] = data['Close'].rolling(8).mean()
    data['sma30'] = data['Close'].rolling(30).mean()
    data['long_signal'] = ((data['sma8'] > data['sma30']) & (data['sma8'].shift(1) <= data['sma30'].shift(1))).astype(int)
    # Features for ML
    data['future_price'] = data['Close'].shift(-5)
    data['future_return'] = (data['future_price'] / data['Close'] - 1) * 100
    data['profitable'] = (data['future_return'] > 0.5).astype(int)
    data['sma_distance'] = (data['sma8'] - data['sma30']) / data['Close'] * 100
    data['volatility'] = data['Close'].pct_change().rolling(20).std() * 100
    data['volume_ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()
    data.dropna(inplace=True)
    features = ['sma8', 'sma30', 'sma_distance', 'volatility', 'volume_ratio']
    X = data[features]
    y = data['profitable']
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    model = RandomForestClassifier(n_estimators=80, max_depth=12, random_state=42)
    model.fit(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    data_test = data[split:].copy()
    data_test['ml_pred'] = model.predict(X_test)
    data_test['trade_signal'] = ((data_test['long_signal'] == 1) & (data_test['ml_pred'] == 1))
    # Output metrics
    col1, col2 = st.columns(2)
    col1.metric("ML Accuracy", f"{test_acc:.1%}")
    latest_price = data['Close'].iloc[-1]
    col2.metric("Current Price", f"${latest_price:,.2f}")
    st.write(f"Total crossovers: {data['long_signal'].sum()} | ML-confirmed signals (test): {data_test['trade_signal'].sum()}")
    # Chart
    st.subheader("Price & SMA Chart")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Close", line=dict(color="white")))
    fig.add_trace(go.Scatter(x=data.index, y=data['sma8'], name="SMA 8", line=dict(color="green")))
    fig.add_trace(go.Scatter(x=data.index, y=data['sma30'], name="SMA 30", line=dict(color="red")))
    signals = data_test[data_test['trade_signal'] == 1]
    fig.add_trace(go.Scatter(x=signals.index, y=signals['Close'], mode='markers',
                             name="ML Trade Signal", marker=dict(color='yellow', size=8)))
    fig.update_layout(template='plotly_dark', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig, use_container_width=True)
    # List trade stats
    trades = []
    for i in range(len(data_test)-5):
        if data_test['trade_signal'].iloc[i]:
            entry = data_test['Close'].iloc[i]
            exit_p = data_test
