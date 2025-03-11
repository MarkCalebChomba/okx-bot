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
    'symbol': '{coin}/USDT:USDT',
    'leverage': 3,
    'timeframe': '15m',
    'historical_limit': 1500,  # Changed from 1000 to 1500 for better ML training
    'backtest_limit': 7500,   # Changed from 5000 to 7500 for more robust backtesting
    'retrain_interval': 60,
    'restart_interval': 6 * 60 * 60  # 6 hours in seconds
}

RISK_CONFIG = {
    'max_position_size': 1.0,
    'min_confidence': 0.65    # Only keep confidence threshold
}

# Indicator calculation functions
def calculate_hull_ma(df, period=9):  # Changed from 14 to 9
    # Hull MA performs better with shorter periods for crypto due to higher volatility
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

def calculate_stoch_rsi(df, period=14, smooth_k=2, smooth_d=3):  # Changed smooth_k from 3 to 2
    # Faster response to price movements while maintaining signal quality
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

def calculate_supertrend(df, period=7, multiplier=3.5):  # Changed from period=10 to 7, multiplier from 3 to 3.5
    # Crypto markets need faster response and slightly wider bands for volatility
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

def calculate_aroon(df, period=14):  # Changed from 25 to 14
    # 14-period Aroon is standard and more responsive for crypto markets
    aroon = pd.DataFrame()
    aroon['aroon_up'] = 100 * df['High'].rolling(period + 1).apply(
        lambda x: float(np.argmax(x)) / period
    )
    aroon['aroon_down'] = 100 * df['Low'].rolling(period + 1).apply(
        lambda x: float(np.argmin(x)) / period
    )
    return aroon

def calculate_elder_ray(df, period=21):  # Changed from 13 to 21
    # 21-period provides better trend identification and fewer false signals
    ema = df['Close'].ewm(span=period, adjust=False).mean()
    elder = pd.DataFrame()
    elder['bull_power'] = df['High'] - ema
    elder['bear_power'] = df['Low'] - ema
    return elder

def calculate_volume_profile(df, n_bins=24):
    try:
        # Volume Profile calculation
        max_price = df['High'].max()
        min_price = df['Low'].min()
        price_bins = np.linspace(min_price, max_price, n_bins)
        
        volume_profile = pd.DataFrame(index=price_bins)
        volume_profile['volume'] = 0
        
        for i in range(len(df)):
            price = df['Close'].iloc[i]
            volume = df['Volume'].iloc[i]
            bin_idx = np.digitize(price, price_bins) - 1
            if 0 <= bin_idx < len(price_bins):
                volume_profile['volume'].iloc[bin_idx] += volume
        
        return volume_profile
    except Exception as e:
        print(f"Volume profile calculation error: {str(e)}")
        return None

def calculate_institutional_flow(df):
    try:
        # Large trade threshold (adjust based on your asset)
        volume_threshold = df['Volume'].mean() * 2
        
        # Identify large trades
        large_trades = df[df['Volume'] > volume_threshold].copy()
        
        # Calculate buy/sell pressure
        large_trades['direction'] = np.where(large_trades['Close'] > large_trades['Open'], 1, -1)
        large_trades['flow'] = large_trades['Volume'] * large_trades['direction']
        
        # Cumulative flow
        institutional_flow = large_trades['flow'].cumsum()
        
        return institutional_flow
    except Exception as e:
        print(f"Institutional flow calculation error: {str(e)}")
        return None

def calculate_money_flow_index(df, period=14):
    try:
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        
        positive_flow = pd.Series(0, index=df.index)
        negative_flow = pd.Series(0, index=df.index)
        
        # Calculate positive and negative money flow
        price_diff = typical_price.diff()
        positive_flow[price_diff > 0] = money_flow[price_diff > 0]
        negative_flow[price_diff < 0] = money_flow[price_diff < 0]
        
        # Calculate MFI
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        return mfi
    except Exception as e:
        print(f"MFI calculation error: {str(e)}")
        return None

def calculate_market_regime(df, period=14):  # Changed from 20 to 14
    # 14-period aligns with other indicators and provides more timely regime changes
    """Identify market regime (trending/ranging/volatile)"""
    atr = df['High'].sub(df['Low']).rolling(period).mean()
    volatility = df['Close'].pct_change().rolling(period).std()
    trend = abs(df['Close'].diff(period).rolling(period).mean())
    
    is_volatile = volatility > volatility.rolling(period*2).mean()
    is_trending = trend > trend.rolling(period*2).mean()
    
    regime = pd.Series(index=df.index)
    regime[is_volatile] = 'volatile'
    regime[is_trending] = 'trending'
    regime[~(is_volatile | is_trending)] = 'ranging'
    return regime

class MLTradingBot:
    def __init__(self):
        # Initialize exchange connection with proper sandbox configuration
        self.exchange = ccxt.okx({
            'apiKey': API_CREDENTIALS['api_key'],
            'secret': API_CREDENTIALS['secret_key'],
            'password': API_CREDENTIALS['passphrase'],
            'enableRateLimit': True,
            'sandboxMode': True,  # changed from True to False
            'options': {
                'defaultType': 'swap',  # For futures trading
                'createMarketBuyOrderRequiresPrice': False
            }
        })
        
        # Ensure live mode is properly set
        self.exchange.set_sandbox_mode(True)  # changed from True to False
        
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
            
            # Initialize ensemble model
            self.models = {
                'trending': RandomForestClassifier(n_estimators=100, random_state=42),
                'ranging': RandomForestClassifier(n_estimators=100, random_state=43),
                'volatile': RandomForestClassifier(n_estimators=100, random_state=44)
            }
            
            print(f"Bot initialized with {self.symbol} on {self.timeframe} timeframe")
            
        except Exception as e:
            print(f"Initialization error: {str(e)}")
            raise e

        self.start_time = time.time()
        self.is_restarting = False
        self.historical_trades = []

    def prepare_features(self, df):
        try:
            df = self.calculate_all_indicators(df)
            
            # Enhanced feature engineering
            X = pd.DataFrame()
            
            # Price action features
            X['price_momentum'] = df['Close'].pct_change(5)
            X['price_acceleration'] = X['price_momentum'].diff()
            X['volatility'] = calculate_volatility(df)
            
            # Volume features
            X['volume_momentum'] = df['Volume'].pct_change(5)
            X['volume_price_trend'] = df['Volume'] * df['Close'].pct_change()
            
            # Technical features
            X['hull_ma_trend'] = df['Hull_MA'].pct_change()
            X['supertrend_signal'] = df['supertrend']
            X['stoch_rsi_crossover'] = (df['stoch_rsi_k'] > df['stoch_rsi_d']).astype(int)
            
            # Market regime features
            X['market_regime'] = calculate_market_regime(df)
            X['trend_strength'] = calculate_trend_strength(df)
            
            # Institutional features
            if 'institutional_flow' in df.columns:
                X['inst_flow'] = df['institutional_flow']
                X['inst_flow_ma'] = df['institutional_flow'].rolling(10).mean()
                X['inst_flow_trend'] = X['inst_flow_ma'].pct_change()
            
            # Clean and scale features
            for col in X.columns:
                if X[col].dtype != 'object':  # Skip categorical columns
                    series = X[col].replace([np.inf, -np.inf], np.nan)
                    X[col] = series.fillna(method='ffill').fillna(0)
                    # Normalize using tanh instead of min-max scaling
                    X[col] = np.tanh(X[col])
            
            return X
            
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
            
            # Add institutional money flow indicators
            df['institutional_flow'] = calculate_institutional_flow(df)
            df['mfi'] = calculate_money_flow_index(df)
            
            # Calculate volume profile periodically
            if not hasattr(self, 'last_volume_profile_time') or \
               time.time() - self.last_volume_profile_time > 3600:  # Update hourly
                self.volume_profile = calculate_volume_profile(df)
                self.last_volume_profile_time = time.time()
            
            return df
            
        except Exception as e:
            print(f"Indicator calculation error: {str(e)}")
            raise e

    def predict_with_ensemble(self, features):
        """Make predictions using regime-specific models"""
        current_regime = features['market_regime'].iloc[-1]
        feature_values = features.drop('market_regime', axis=1).iloc[-1:]
        
        # Get predictions from all models
        predictions = {}
        confidences = {}
        
        for regime, model in self.models.items():
            if hasattr(model, 'classes_'):
                pred = model.predict(feature_values)
                conf = max(model.predict_proba(feature_values)[0])
                predictions[regime] = pred[0]
                confidences[regime] = conf
        
        # Weight predictions by regime relevance
        if current_regime in predictions:
            primary_prediction = predictions[current_regime]
            primary_confidence = confidences[current_regime]
        else:
            # Fallback to ensemble average
            primary_prediction = 1 if np.mean(list(predictions.values())) > 0.5 else 0
            primary_confidence = np.mean(list(confidences.values()))
        
        return primary_prediction, primary_confidence

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
            
            # Calculate position size based on 1% of available balance
            balance_info = self.exchange.fetch_balance()
            available_balance = balance_info['USDT']['free']  # adjust key if needed
            ticker = self.exchange.fetch_ticker(self.symbol)
            current_price = ticker['last']
            contract_size = 100
            trade_value = available_balance * 0.01
            desired_size = (trade_value / (current_price * contract_size)) * signal
            
            # Ensure a minimum acceptable trade size if 1% is too low
            min_trade_size = 0.01
            if abs(desired_size) < min_trade_size:
                desired_size = min_trade_size * signal
            
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
                
                self.exchange.create_order(
                    self.symbol,
                    'market',
                    side,
                    abs(desired_size),
                    params={
                        'posSide': pos_side,
                        'tdMode': 'cross',
                        'leverage': self.leverage
                    }
                )
                
                print(f"Trade executed: {side} {abs(desired_size)} {self.symbol} ({pos_side})")
            
        except Exception as e:
            print(f"Trade execution error: {str(e)}")
            print("Stack trace:", traceback.format_exc())

    def check_active_positions(self):
        """Check if there are any open positions"""
        try:
            positions = self.exchange.fetch_positions([self.symbol])
            for pos in positions:
                if pos['symbol'] == self.symbol and float(pos['contracts']) > 0:
                    return True
            return False
        except Exception as e:
            print(f"Error checking positions: {str(e)}")
            return False

    def close_all_positions(self):
        """Close all open positions"""
        try:
            positions = self.exchange.fetch_positions([self.symbol])
            for pos in positions:
                if pos['symbol'] == self.symbol and float(pos['contracts']) > 0:
                    side = 'sell' if pos['side'] == 'long' else 'buy'
                    self.exchange.create_order(
                        self.symbol,
                        'market',
                        side,
                        abs(float(pos['contracts'])),
                        params={
                            'posSide': pos['side'],
                            'reduceOnly': True
                        }
                    )
                    print(f"Closed position: {pos['side']} {pos['contracts']} {self.symbol}")
            return True
        except Exception as e:
            print(f"Error closing positions: {str(e)}")
            return False

    def should_restart(self):
        """Check if bot should restart"""
        elapsed_time = time.time() - self.start_time
        return elapsed_time >= TRADING_CONFIG['restart_interval']

    def backtest(self, df, features, labels):
        """Modified backtest method to return more metrics"""
        try:
            print("\nRunning backtest on historical data...")
            
            # Split data for backtesting
            train_size = int(len(df) * 0.7)
            
            # Handle categorical features
            categorical_cols = ['market_regime']
            numeric_features = features.drop(categorical_cols, axis=1)
            train_features = numeric_features[:train_size]
            train_labels = labels[:train_size]
            test_features = numeric_features[train_size:]
            test_labels = labels[train_size:]
            
            # Train on historical data
            backtest_model = RandomForestClassifier(n_estimators=100, random_state=42)
            backtest_model.fit(train_features, train_labels)
            
            # Test predictions
            predictions = backtest_model.predict(test_features)
            confidence_scores = backtest_model.predict_proba(test_features)
            
            # Calculate results
            correct_predictions = sum(predictions == test_labels)
            accuracy = correct_predictions / len(test_labels)
            
            # Simulate trading
            balance = 1000 # Starting with 1000 USDT
            position = None
            test_prices = df['Close'][train_size:].values
            
            # Track trades for analysis
            trades = []
            for i in range(len(predictions)):
                confidence = max(confidence_scores[i])
                if confidence > RISK_CONFIG['min_confidence']:
                    signal = 1 if predictions[i] == 1 else -1
                    
                    # Close existing position
                    if position:
                        pnl = position['size'] * (test_prices[i] - position['entry']) * position['side']
                        balance += pnl
                        trades.append({
                            'exit_price': test_prices[i],
                            'pnl': pnl,
                            'balance': balance
                        })
                        position = None
                    
                    # Open new position
                    if signal != 0:
                        position = {
                            'side': signal,
                            'entry': test_prices[i],
                            'size': (balance * 0.1) / test_prices[i]  # 10% of balance
                        }
                        trades.append({
                            'entry_price': test_prices[i],
                            'side': 'long' if signal == 1 else 'short',
                            'size': position['size']
                        })
            
            # Close final position
            if position:
                pnl = position['size'] * (test_prices[-1] - position['entry']) * position['side']
                balance += pnl
                trades.append({
                    'exit_price': test_prices[-1],
                    'pnl': pnl,
                    'balance': balance
                })
            
            # Print detailed results
            print(f"\nBacktest Results:")
            print(f"Accuracy: {accuracy:.2%}")
            print(f"Final Balance: {balance:.2f} USDT")
            print(f"Return: {((balance-1000)/1000)*100:.2f}%")
            print(f"Total Trades: {len(trades)}")
            
            # Calculate additional metrics
            winning_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
            if len(trades) > 0:
                win_rate = winning_trades / len(trades)
                print(f"Win Rate: {win_rate:.2%}")
            
            # Return tuple of (success, accuracy, final_balance, returns)
            return (
                accuracy > 0.55 and ((balance-1000)/1000)*100 > 5,
                accuracy * 100,
                balance,
                ((balance-1000)/1000)*100
            )
            
        except Exception as e:
            print(f"Backtest error: {str(e)}")
            print("Stack trace:", traceback.format_exc())
            return False, 0, 1000, 0

    def run(self):
        print("Starting Enhanced ML Trading Bot...")
        try:
            # Fetch extended historical data for backtesting
            backtest_data = self.exchange.fetch_ohlcv(
                self.symbol,
                self.timeframe,
                limit=TRADING_CONFIG['backtest_limit']
            )
            
            # Prepare backtest data
            backtest_df = pd.DataFrame(backtest_data, 
                                     columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            backtest_df['Timestamp'] = pd.to_datetime(backtest_df['Timestamp'], unit='ms')
            backtest_df = backtest_df.set_index('Timestamp')
            
            # Convert values to float
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                backtest_df[col] = backtest_df[col].astype(float)
            
            # Prepare features and labels for backtesting
            backtest_features = self.prepare_features(backtest_df)
            backtest_labels = (backtest_df['Close'].shift(-1) > backtest_df['Close']).astype(int)[:-1]
            
            # Run backtest
            if not self.backtest(backtest_df, backtest_features[:-1], backtest_labels):
                print("Backtest results unsatisfactory. Consider adjusting strategy.")
                if input("Continue anyway? (y/n): ").lower() != 'y':
                    return False
            
            print("\nStarting live trading...")
            
            # Continue with regular trading loop
            while True:
                try:
                    # Check if restart is needed
                    if self.should_restart():
                        print("Restart interval reached...")
                        if not self.check_active_positions():
                            print("No active positions, restarting bot...")
                            return True  # Signal for restart
                        else:
                            print("Active positions found, waiting for trades to close...")
                            self.is_restarting = True
                    
                    # Regular trading loop
                    balance_info = self.exchange.fetch_balance()
                    positions = self.exchange.fetch_positions([self.symbol])
                    available_balance = balance_info['USDT']['free']  # adjust key if needed
                    total_profit = sum((pos.get('unrealizedPnl') or 0) for pos in positions)
                    print(f"Balance: {available_balance} | Profit: {total_profit}")
                    
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
                    
                    # Prepare features and make prediction
                    features = self.prepare_features(df)
                    
                    # Train regime-specific models if needed
                    current_regime = features['market_regime'].iloc[-1]
                    if not hasattr(self.models[current_regime], 'classes_'):
                        print(f"Training {current_regime} model...")
                        labels = (df['Close'].shift(-1) > df['Close']).astype(int)[:-1]
                        regime_features = features[features['market_regime'] == current_regime].drop('market_regime', axis=1)
                        if len(regime_features) > 0:
                            self.models[current_regime].fit(regime_features[:-1], labels[:len(regime_features)-1])
                    
                    # Make ensemble prediction
                    prediction, confidence = self.predict_with_ensemble(features)
                    
                    # Execute trade with enhanced confidence check
                    if confidence > RISK_CONFIG['min_confidence']:
                        signal = 1 if prediction == 1 else -1
                        # Adjust signal strength based on regime
                        if current_regime == 'volatile':
                            signal *= 0.7  # Reduce position size in volatile markets
                        self.execute_trade(signal, confidence)
                    
                    # If restarting and no positions, exit
                    if self.is_restarting and not self.check_active_positions():
                        print("All positions closed, restarting bot...")
                        return True
                    
                    # Wait before next iteration
                    time.sleep(60)
                
                except Exception as e:
                    print(f"Error in main loop: {str(e)}")
                    print("Stack trace:", traceback.format_exc())
                    time.sleep(60)
        
        except Exception as e:
            print(f"Error in main loop: {str(e)}")
            print("Stack trace:", traceback.format_exc())
            time.sleep(60)

def run_backtest_all_coins():
    while True:  # Add continuous operation loop
        try:
            coins = ['DOGE/USDT:USDT', 'PEPE/USDT:USDT', 'ADA/USDT:USDT', 'XRP/USDT:USDT']
            results = {}
            best_coin = None
            best_accuracy = 0
            
            for coin in coins:
                print(f"\nRunning backtest for {coin}...")
                bot = MLTradingBot()
                bot.symbol = coin
                
                try:
                    # ...existing code...
                    backtest_data = bot.exchange.fetch_ohlcv(
                        bot.symbol,
                        bot.timeframe,
                        limit=TRADING_CONFIG['backtest_limit']
                    )
                    
                    backtest_df = pd.DataFrame(backtest_data, 
                                             columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
                    backtest_df['Timestamp'] = pd.to_datetime(backtest_df['Timestamp'], unit='ms')
                    backtest_df = backtest_df.set_index('Timestamp')
                    
                    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                        backtest_df[col] = backtest_df[col].astype(float)
                    
                    backtest_features = bot.prepare_features(backtest_df)
                    backtest_labels = (backtest_df['Close'].shift(-1) > backtest_df['Close']).astype(int)[:-1]
                    
                    is_successful, accuracy, final_balance, returns = bot.backtest(backtest_df, backtest_features[:-1], backtest_labels)
                    
                    results[coin] = {
                        'Accuracy': accuracy,
                        'Returns': returns,
                        'Final Balance': final_balance
                    }
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_coin = coin
                
                except Exception as e:
                    print(f"Error testing {coin}: {str(e)}")
                    continue
            
            print("\n=== Complete Backtest Results ===")
            for coin, metrics in results.items():
                print(f"\n{coin} Results:")
                print(f"Accuracy: {metrics['Accuracy']:.2f}%")
                print(f"Returns: {metrics['Returns']:.2f}%")
                print(f"Final Balance: {metrics['Final Balance']:.2f} USDT")
            
            if best_coin and best_accuracy >= 55:
                print(f"\nTrading best performing coin: {best_coin}")
                print(f"Accuracy: {best_accuracy:.2f}%")
                print(f"Expected Return: {results[best_coin]['Returns']:.2f}%")
                
                bot = MLTradingBot()
                bot.symbol = best_coin
                should_restart = bot.run()  # Capture restart signal
                
                if should_restart:
                    print("Scheduled restart initiated...")
                    time.sleep(5)  # Allow time for logs to be written
                    continue  # Restart the entire process
            
            else:
                print("\nNo coin met the minimum accuracy threshold of 55%. Waiting 15 minutes before retry...")
                time.sleep(900)  # Wait 15 minutes before trying again
        
        except Exception as e:
            print(f"Critical error in main loop: {str(e)}")
            print("Stack trace:", traceback.format_exc())
            print("Restarting in 5 minutes...")
            time.sleep(300)

if __name__ == "__main__":
    print("Starting Enhanced ML Trading Bot...\n")
    run_backtest_all_coins()  # This will now run indefinitely
