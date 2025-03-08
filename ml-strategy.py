import ccxt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import time
import traceback  # Add this import
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuration
API_CREDENTIALS = {
    'api_key': '074c23e1-6750-4ce2-a7a5-3b86c4a0a03f',
    'secret_key': '07BB467EFE2A1DAE7B3AEA30738B61BF',
    'passphrase': '#Dinywa15'
}

TRADING_CONFIG = {
    'symbol': 'DOGE/USDT:USDT',
    'leverage': 3,
    'timeframe': '15m',
    'historical_limit': 1000,
    'retrain_interval': 60
}

RISK_CONFIG = {
    'max_position_size': 1.0,
    'stop_loss_pct': 0.02,
    'take_profit_pct': 0.04,
    'min_confidence': 0.6
}

# Indicator calculation functions
def calculate_hull_ma(df, period=14):
    half_period = int(period/2)
    sqrt_period = int(np.sqrt(period))
    
    weighted_data = pd.DataFrame(index=df.index)
    weighted_data['half_period'] = df['Close'].rolling(window=half_period).apply(
        lambda x: np.average(x, weights=range(1, len(x)+1))
    )
    weighted_data['full_period'] = df['Close'].rolling(window=period).apply(
        lambda x: np.average(x, weights=range(1, len(x)+1))
    )
    weighted_data['raw_hma'] = 2 * weighted_data['half_period'] - weighted_data['full_period']
    hull_ma = weighted_data['raw_hma'].rolling(window=sqrt_period).mean()
    return hull_ma

def calculate_ichimoku(df):
    high_values = df['High']
    low_values = df['Low']
    
    tenkan_high = high_values.rolling(window=9).max()
    tenkan_low = low_values.rolling(window=9).min()
    tenkan_sen = (tenkan_high + tenkan_low) / 2
    
    kijun_high = high_values.rolling(window=26).max()
    kijun_low = low_values.rolling(window=26).min()
    kijun_sen = (kijun_high + kijun_low) / 2
    
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    
    senkou_high = high_values.rolling(window=52).max()
    senkou_low = low_values.rolling(window=52).min()
    senkou_span_b = ((senkou_high + senkou_low) / 2).shift(26)
    
    return pd.DataFrame({
        'tenkan_sen': tenkan_sen,
        'kijun_sen': kijun_sen,
        'senkou_span_a': senkou_span_a,
        'senkou_span_b': senkou_span_b
    })

def calculate_stoch_rsi(df, period=14, smooth_k=3, smooth_d=3):
    close_delta = df['Close'].diff()
    gains = close_delta.where(close_delta > 0, 0)
    losses = -close_delta.where(close_delta < 0, 0)
    
    avg_gain = gains.rolling(window=period).mean()
    avg_loss = losses.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    stoch_rsi = pd.DataFrame()
    stoch_rsi['k'] = ((rsi - rsi.rolling(period).min()) / 
                      (rsi.rolling(period).max() - rsi.rolling(period).min())).rolling(smooth_k).mean() * 100
    stoch_rsi['d'] = stoch_rsi['k'].rolling(smooth_d).mean()
    return stoch_rsi

def calculate_supertrend(df, period=10, multiplier=3):
    hl2 = (df['High'] + df['Low']) / 2
    atr = df['High'].sub(df['Low']).rolling(period).mean()
    
    # Calculate basic upper and lower bands
    basic_ub = hl2 + (multiplier * atr)
    basic_lb = hl2 - (multiplier * atr)
    
    # Initialize final upper and lower bands
    final_ub = basic_ub.copy()
    final_lb = basic_lb.copy()
    
    # Calculate final upper and lower bands
    for i in range(period, len(df)):
        final_ub[i] = basic_ub[i] if (
            basic_ub[i] < final_ub[i-1] or df['Close'][i-1] > final_ub[i-1]
        ) else final_ub[i-1]
        
        final_lb[i] = basic_lb[i] if (
            basic_lb[i] > final_lb[i-1] or df['Close'][i-1] < final_lb[i-1]
        ) else final_lb[i-1]
    
    # Calculate SuperTrend
    supertrend = pd.Series(1, index=df.index)
    for i in range(period, len(df)):
        supertrend[i] = 1 if df['Close'][i] > final_ub[i] else -1 if df['Close'][i] < final_lb[i] else supertrend[i-1]
    
    return supertrend

def calculate_volatility(df, window=20):
    return df['Close'].pct_change().rolling(window=window).std()

def calculate_trend_strength(df, period=14):
    # Calculate directional movement
    plus_dm = df['High'].diff()
    minus_dm = -df['Low'].diff()
    plus_dm = plus_dm.where(plus_dm > 0, 0)
    minus_dm = minus_dm.where(minus_dm > 0, 0)
    
    # Calculate true range
    tr = pd.DataFrame([
        df['High'] - df['Low'],
        abs(df['High'] - df['Close'].shift(1)),
        abs(df['Low'] - df['Close'].shift(1))
    ]).max()
    
    # Smoothed calculations
    tr_smoothed = tr.ewm(span=period).mean()
    plus_di = 100 * plus_dm.ewm(span=period).mean() / tr_smoothed
    minus_di = 100 * minus_dm.ewm(span=period).mean() / tr_smoothed
    
    # Calculate ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(span=period).mean() / 100  # Normalized to 0-1
    return adx

def calculate_ultimate_oscillator(df, period1=7, period2=14, period3=28):
    # True Range
    tr = pd.DataFrame()
    tr['h_l'] = df['High'] - df['Low']
    tr['h_pc'] = abs(df['High'] - df['Close'].shift(1))
    tr['l_pc'] = abs(df['Low'] - df['Close'].shift(1))
    tr['tr'] = tr[['h_l', 'h_pc', 'l_pc']].max(axis=1)
    
    # Buying Pressure
    bp = df['Close'] - pd.DataFrame([df['Low'], df['Close'].shift(1)]).min()
    
    # Average True Range calculations
    avg7 = bp.rolling(period1).sum() / tr['tr'].rolling(period1).sum()
    avg14 = bp.rolling(period2).sum() / tr['tr'].rolling(period2).sum()
    avg28 = bp.rolling(period3).sum() / tr['tr'].rolling(period3).sum()
    
    # Ultimate Oscillator calculation
    uo = 100 * ((4 * avg7 + 2 * avg14 + avg28) / 7)
    return uo

def calculate_aroon(df, period=25):
    aroon = pd.DataFrame()
    aroon['aroon_up'] = 100 * df['High'].rolling(period + 1).apply(
        lambda x: float(np.argmax(x)) / period
    )
    aroon['aroon_down'] = 100 * df['Low'].rolling(period + 1).apply(
        lambda x: float(np.argmin(x)) / period
    )
    return aroon

def calculate_elder_ray(df, period=13):
    ema = df['Close'].ewm(span=period, adjust=False).mean()
    elder = pd.DataFrame()
    elder['bull_power'] = df['High'] - ema
    elder['bear_power'] = df['Low'] - ema
    return elder

class MLTradingBot:
    def __init__(self):
        # Initialize exchange connection with proper sandbox configuration
        self.exchange = ccxt.okx({
            'apiKey': API_CREDENTIALS['api_key'],
            'secret': API_CREDENTIALS['secret_key'],
            'password': API_CREDENTIALS['passphrase'],
            'enableRateLimit': True,
            'sandboxMode': True,
            'options': {
                'defaultType': 'swap',  # For futures trading
                'createMarketBuyOrderRequiresPrice': False
            }
        })
        
        # Ensure sandbox mode is properly set
        self.exchange.set_sandbox_mode(True)
        
        try:
            # Test authentication
            self.exchange.check_required_credentials()
            # Load markets without fetching currencies
            self.exchange.loadMarkets(True)
            
            # Initialize trading parameters
            self.symbol = TRADING_CONFIG['symbol']
            self.leverage = TRADING_CONFIG['leverage']
            self.timeframe = TRADING_CONFIG['timeframe']
            
            # Initialize ML model
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.scaler = StandardScaler()
            
            print(f"Bot initialized with {self.symbol} on {self.timeframe} timeframe")
            
        except Exception as e:
            print(f"Initialization error: {str(e)}")
            raise e

    def prepare_features(self, df):
        try:
            # Calculate all indicators
            df = self.calculate_all_indicators(df)
            
            # Select and clean features
            feature_columns = [
                'Hull_MA', 'ichimoku_tenkan', 'ichimoku_kijun',
                'stoch_rsi_k', 'stoch_rsi_d', 'ultimate_oscillator',
                'kst', 'elder_bull', 'elder_bear', 'aroon_up',
                'aroon_down', 'supertrend'
            ]
            
            X = pd.DataFrame()
            for col in feature_columns:
                if col in df.columns:
                    series = df[col].replace([np.inf, -np.inf], np.nan)
                    X[col] = series.fillna(method='ffill').fillna(0)
            
            # Add trend and volatility features
            X['trend_strength'] = calculate_trend_strength(df)
            X['volatility'] = calculate_volatility(df)
            
            # Scale features
            for col in X.columns:
                max_val = abs(X[col]).max()
                if max_val > 0:
                    X[col] = X[col] / max_val
            
            return X.values
        except Exception as e:
            print(f"Feature preparation error: {str(e)}")
            raise e

    def calculate_all_indicators(self, df):
        try:
            # Calculate basic indicators
            df['Hull_MA'] = calculate_hull_ma(df)
            ichimoku = calculate_ichimoku(df)
            df['ichimoku_tenkan'] = ichimoku['tenkan_sen']
            df['ichimoku_kijun'] = ichimoku['kijun_sen']
            
            # Calculate momentum indicators
            stoch_rsi = calculate_stoch_rsi(df)
            df['stoch_rsi_k'] = stoch_rsi['k']
            df['stoch_rsi_d'] = stoch_rsi['d']
            df['ultimate_oscillator'] = calculate_ultimate_oscillator(df)
            
            # Calculate additional indicators
            df['supertrend'] = calculate_supertrend(df)
            elder = calculate_elder_ray(df)
            df['elder_bull'] = elder['bull_power']
            df['elder_bear'] = elder['bear_power']
            aroon = calculate_aroon(df)
            df['aroon_up'] = aroon['aroon_up']
            df['aroon_down'] = aroon['aroon_down']
            
            return df
            
        except Exception as e:
            print(f"Indicator calculation error: {str(e)}")
            raise e

    def execute_trade(self, signal, confidence):
        try:
            # Get current position
            positions = self.exchange.fetch_positions([self.symbol])
            current_size = 0
            current_side = None
            if positions:
                for pos in positions:
                    if pos['symbol'] == self.symbol:
                        current_size = float(pos['contracts'])
                        current_side = pos['side']
                        break
            
            # Calculate position size based on confidence
            desired_size = RISK_CONFIG['max_position_size'] * confidence * signal
            
            # Set leverage
            self.exchange.set_leverage(self.leverage, self.symbol)
            
            # Close existing position if direction changes
            if current_size != 0:
                if (signal > 0 and current_side == 'short') or (signal < 0 and current_side == 'long'):
                    self.exchange.create_order(
                        self.symbol,
                        'market',
                        'buy' if current_side == 'short' else 'sell',
                        abs(current_size),
                        params={
                            'posSide': 'short' if current_side == 'short' else 'long',
                            'reduceOnly': True
                        }
                    )
            
            # Open new position
            if abs(desired_size) > 0:
                side = 'buy' if desired_size > 0 else 'sell'
                pos_side = 'long' if desired_size > 0 else 'short'
                
                # Create the order with proper position side
                self.exchange.create_order(
                    self.symbol,
                    'market',
                    side,
                    abs(desired_size),
                    params={
                        'posSide': pos_side,
                        'tdMode': 'cross',  # Use cross margin mode
                        'leverage': self.leverage
                    }
                )
                
                print(f"Trade executed: {side} {abs(desired_size)} {self.symbol} ({pos_side})")
            
        except Exception as e:
            print(f"Trade execution error: {str(e)}")
            print("Stack trace:", traceback.format_exc())

    def run(self):
        print("Starting OKX Trading Bot...")
        
        while True:
            try:
                # Fetch latest data
                ohlcv = self.exchange.fetch_ohlcv(
                    self.symbol,
                    self.timeframe,
                    limit=TRADING_CONFIG['historical_limit']
                )
                
                # Create DataFrame with proper column names
                df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
                df = df.set_index('Timestamp')
                
                # Validate data
                if df.empty:
                    print("No data received, waiting...")
                    time.sleep(60)
                    continue
                
                # Convert values to float
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    df[col] = df[col].astype(float)
                
                # Handle missing values
                if df.isnull().any().any():
                    print("Warning: Missing values in data")
                    df = df.fillna(method='ffill').fillna(method='bfill')
                
                # Prepare features and train model if needed
                features = self.prepare_features(df)
                
                if not hasattr(self.model, 'classes_'):
                    print("Model not trained yet, training with current data...")
                    labels = (df['Close'].shift(-1) > df['Close']).astype(int)[:-1]
                    self.model.fit(features[:-1], labels)
                    continue
                
                # Make predictions
                prediction = self.model.predict(features[-1:])
                confidence = max(self.model.predict_proba(features[-1:])[0])
                
                # Execute trade if confidence is high enough
                if confidence > RISK_CONFIG['min_confidence']:
                    signal = 1 if prediction[0] == 1 else -1
                    self.execute_trade(signal, confidence)
                
                # Wait before next iteration
                time.sleep(60)
                
            except Exception as e:
                print(f"Error in main loop: {str(e)}")
                print("Stack trace:", traceback.format_exc())
                time.sleep(60)

if __name__ == "__main__":
    try:
        bot = MLTradingBot()
        bot.run()
    except Exception as e:
        print(f"Critical error: {str(e)}")
