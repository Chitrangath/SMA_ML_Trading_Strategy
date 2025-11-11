"""
SMA ML Trading Strategy - FIXED VERSION
"""
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("SMA ML TRADING STRATEGY")
print("=" * 60)

# Download BTC data
print("\n[1/5] Downloading BTC data...")
btc = yf.download('BTC-USD', period='6mo', interval='1h', progress=False)

# Fix column names (yfinance multi-index issue)
if isinstance(btc.columns, pd.MultiIndex):
    btc.columns = btc.columns.droplevel(1)

print(f"✓ Downloaded {len(btc)} candles")

# Calculate SMAs (from your Pine Script)
print("\n[2/5] Calculating SMAs...")
btc['sma8'] = btc['Close'].rolling(8).mean()
btc['sma30'] = btc['Close'].rolling(30).mean()
btc['sma10'] = btc['Close'].rolling(10).mean()
btc['sma3'] = btc['Close'].rolling(3).mean()

btc['long_signal'] = ((btc['sma8'] > btc['sma30']) & 
                       (btc['sma8'].shift(1) <= btc['sma30'].shift(1))).astype(int)

print(f"✓ Found {btc['long_signal'].sum()} crossover signals")

# Create ML features
print("\n[3/5] Engineering features...")
btc['future_price'] = btc['Close'].shift(-5)
btc['future_return'] = (btc['future_price'] / btc['Close'] - 1) * 100
btc['profitable'] = (btc['future_return'] > 0.5).astype(int)

btc['sma_distance'] = (btc['sma8'] - btc['sma30']) / btc['Close'] * 100
btc['volatility'] = btc['Close'].pct_change().rolling(20).std() * 100
btc['volume_ratio'] = btc['Volume'] / btc['Volume'].rolling(20).mean()

btc.dropna(inplace=True)
print(f"✓ Created features, {len(btc)} clean candles")

# Train ML model
print("\n[4/5] Training Random Forest...")
features = ['sma8', 'sma30', 'sma10', 'sma3', 'sma_distance', 'volatility', 'volume_ratio']
X = btc[features]
y = btc['profitable']

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

print(f"✓ Train accuracy: {model.score(X_train, y_train):.2%}")
print(f"✓ Test accuracy: {model.score(X_test, y_test):.2%}")

# Feature importance
importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop Features:")
for idx, row in importance.head(3).iterrows():
    print(f"  • {row['feature']}: {row['importance']:.3f}")

# Backtest
print("\n[5/5] Backtesting...")
btc_test = btc[split:].copy()
btc_test['ml_pred'] = model.predict(X_test)
btc_test['trade_signal'] = ((btc_test['long_signal'] == 1) & (btc_test['ml_pred'] == 1))

trades = []
for i in range(len(btc_test) - 5):
    if btc_test['trade_signal'].iloc[i]:
        entry = btc_test['Close'].iloc[i]
        exit_price = btc_test['Close'].iloc[i + 5]
        profit = (exit_price - entry) / entry * 100
        profit_usd = 5 * (profit / 100)  # $5 per trade
        trades.append({'profit_pct': profit, 'profit_usd': profit_usd})

if trades:
    trades_df = pd.DataFrame(trades)
    wins = len(trades_df[trades_df['profit_pct'] > 0])
    
    print(f"\n✓ Total trades: {len(trades)}")
    print(f"✓ Winning trades: {wins}")
    print(f"✓ Win rate: {(wins/len(trades)*100):.1f}%")
    print(f"✓ Avg profit: {trades_df['profit_pct'].mean():.2f}%")
    print(f"✓ Total profit: ${trades_df['profit_usd'].sum():.2f}")
    print(f"✓ Best trade: +{trades_df['profit_pct'].max():.2f}%")
    print(f"✓ Worst trade: {trades_df['profit_pct'].min():.2f}%")
else:
    print("✗ No trades generated")

# Live prediction
print("\n" + "=" * 60)
print("LIVE PREDICTION")
print("=" * 60)
latest = model.predict(X_test.iloc[-1:].values)[0]
prob = model.predict_proba(X_test.iloc[-1:].values)[0]
current_price = btc['Close'].iloc[-1]

print(f"Current BTC Price: ${current_price:,.2f}")
print(f"ML Prediction: {'✓ TAKE TRADE' if latest == 1 else '✗ SKIP TRADE'}")
print(f"Confidence: {max(prob):.1%}")

if btc['long_signal'].iloc[-1] == 1:
    print(f"\n🔔 SMA Crossover detected!")
    if latest == 1:
        print("✅ ML CONFIRMS: Strong signal")
    else:
        print("⚠️  ML WARNS: Low probability setup")
else:
    print("\n💤 No crossover signal at the moment")

print("=" * 60)
print("✓ Strategy Complete!")
print("=" * 60)
