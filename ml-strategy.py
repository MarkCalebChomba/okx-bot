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
import json
from pathlib import Path

# Configuration
API_CREDENTIALS = {
    'api_key': '544d6587-0a7d-4b73-bb06-0e3656c08a18',
    'secret_key': '9C2CA165254391E4B4638DE6577288BD',
    'passphrase': '#Dinywa15'
}

TRADING_CONFIG = {
    'symbol': 'DOGE/USDT:USDT',  # Changed to focus exclusively on DOGE
    'leverage': 5,
    'timeframe': '15m',
    'historical_limit': 1500,
    'backtest_limit': 7500,
    'retrain_interval': 60,
    'restart_interval': 6 * 60 * 60  # 6 hours in seconds
}

RISK_CONFIG = {
    'position_size_pct': 0.01,  # Position percentage (1% of balance)
    'min_confidence': 0.57,    # Confidence threshold for trades
    'trailing_stop_initial': 0.2,  # Initial trailing stop distance (5%)
    'trailing_stop_min': 0.2,  # Minimum trailing stop distance (2%)
    'trailing_stop_max': 0.10   # Maximum trailing stop distance (10%)
}

# DOGE-specific contract configuration
DOGE_CONFIG = {
    'contract_size': 1000,      # Each contract is 1000 DOGE
    'min_contracts': 0.01,     # Minimum order size is 0.01 contracts (corrected from 0.1)
    'precision': 2             # Round to 2 decimal places (changed from 1 to allow for 0.03 contracts)
}

# Indicator calculation functions
def calculate_hull_ma(df, period=9):
    """Calculate Hull Moving Average with the given period"""
    half_period = int(period/2)
    sqrt_period = int(np.sqrt(period))
    
    # Create DataFrame to store intermediate calculations
    weighted_data = pd.DataFrame(index=df.index)
    
    # Calculate half period weighted moving average
    weighted_data['half_period'] = df['Close'].rolling(window=half_period).apply(
        lambda x: np.average(x, weights=range(1, len(x)+1))
    )
    
    # Calculate full period weighted moving average
    weighted_data['full_period'] = df['Close'].rolling(window=period).apply(
        lambda x: np.average(x, weights=range(1, len(x)+1))
    )
    
    # Calculate raw Hull MA
    weighted_data['raw_hma'] = 2 * weighted_data['half_period'] - weighted_data['full_period']
    
    # Calculate final Hull MA
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
            
            # Initialize model with optimal parameters
            self.model = RandomForestClassifier(
                n_estimators=300,          # Increased for better accuracy
                max_depth=8,               # Reduced to prevent overfitting
                min_samples_split=10,      # Increased for more robust splits
                min_samples_leaf=4,        # Increased for better stability
                max_features='sqrt',       # Keep sqrt for feature selection
                class_weight={0: 1.0, 1: 1.0},  # Adjusted to remove bias
                random_state=42,
                n_jobs=-1
            )
            
            # Add feature importance tracking
            self.feature_importance = {}
            
            # Track trailing stops for positions
            self.active_positions = {}
            
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
            
            X = pd.DataFrame()
            
            # Market Regime Features
            regime = calculate_market_regime(df)
            X['regime_volatile'] = (regime == 'volatile').astype(int)
            X['regime_trending'] = (regime == 'trending').astype(int)
            X['regime_ranging'] = (regime == 'ranging').astype(int)
            
            # Price Action Features
            X['hull_ma_trend'] = (df['Close'] > df['Hull_MA']).astype(int)
            X['price_momentum'] = df['Close'].pct_change(5)
            X['price_acceleration'] = X['price_momentum'].diff()
            X['volatility'] = calculate_volatility(df)
            
            # Ichimoku Features
            X['above_tenkan'] = (df['Close'] > df['ichimoku_tenkan']).astype(int)
            X['above_kijun'] = (df['Close'] > df['ichimoku_kijun']).astype(int)
            
            # Momentum Features
            X['stoch_rsi_crossover'] = (df['stoch_rsi_k'] > df['stoch_rsi_d']).astype(int)
            X['uo_oversold'] = (df['ultimate_oscillator'] < 30).astype(int)
            X['uo_overbought'] = (df['ultimate_oscillator'] > 70).astype(int)
            
            # Trend Features
            X['supertrend_direction'] = df['supertrend']
            X['elder_bull_power'] = df['elder_bull']
            X['elder_bear_power'] = df['elder_bear']
            X['aroon_trend'] = (df['aroon_up'] > df['aroon_down']).astype(int)
            
            # Volume and Money Flow Features
            X['institutional_flow'] = df['institutional_flow']
            X['mfi_oversold'] = (df['mfi'] < 20).astype(int)
            X['mfi_overbought'] = (df['mfi'] > 80).astype(int)
            
            # Interaction Features
            X['trend_strength'] = calculate_trend_strength(df)
            X['trend_strength_in_trend'] = X['regime_trending'] * X['trend_strength']
            X['volatility_in_volatile'] = X['regime_volatile'] * X['volatility']
            X['momentum_in_ranging'] = X['regime_ranging'] * X['price_momentum']
            
            # Clean and normalize features
            for col in X.columns:
                series = X[col].replace([np.inf, -np.inf], np.nan)
                X[col] = series.fillna(method='ffill').fillna(0)
                X[col] = np.tanh(X[col])  # Normalize to [-1, 1]
            
            return X
            
        except Exception as e:
            print(f"Feature preparation error: {str(e)}")
            raise e

    def analyze_feature_importance(self, features):
        """Analyze and track feature importance"""
        try:
            importances = self.model.feature_importances_
            feature_imp = pd.Series(importances, index=features.columns)
            self.feature_importance = feature_imp.sort_values(ascending=False)
            
            print("\nTop 10 Most Important Features:")
            print(self.feature_importance.head(10))
            
            # Remove low importance features (optional)
            important_features = self.feature_importance[self.feature_importance > 0.01].index
            return features[important_features]
            
        except Exception as e:
            print(f"Feature importance analysis error: {str(e)}")
            return features

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
        """Simplified prediction with single model"""
        try:
            feature_values = features.iloc[-1:] 
            
            # Make prediction and get confidence
            pred = self.model.predict(feature_values)
            conf = max(self.model.predict_proba(feature_values)[0])
            
            return pred[0], conf
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return 0, 0

    def verify_trade_execution(self, order_id):
        """Verify if trade was executed properly"""
        try:
            max_attempts = 5
            attempt = 0
            while attempt < max_attempts:
                order = self.exchange.fetch_order(order_id, self.symbol)
                if order['status'] == 'closed':
                    return True
                elif order['status'] == 'canceled' or order['status'] == 'expired':
                    return False
                time.sleep(2)
                attempt += 1
            return False
        except Exception as e:
            print(f"Trade verification error: {str(e)}")
            return False

    def execute_trade(self, signal, confidence):
        try:
            print(f"\nAttempting to execute trade - Signal: {signal}, Confidence: {confidence:.2f}")
            
            # Get current position
            positions = self.exchange.fetch_positions([self.symbol])
            current_position = None
            
            for pos in positions:
                if pos['symbol'] == self.symbol and float(pos['contracts']) > 0:
                    current_position = {
                        'side': pos['side'],
                        'size': float(pos['contracts']),
                        'entry_price': float(pos['entryPrice']),
                        'leverage': float(pos['leverage']),
                        'unrealizedPnl': float(pos['unrealizedPnl'])
                    }
                    break
            
            # Get market info for better quantity calculation
            market = self.exchange.market(self.symbol)
            ticker = self.exchange.fetch_ticker(self.symbol)
            current_price = float(ticker['last'])
            
            # Log detailed market information for debugging
            print(f"\n----- Market Information for {self.symbol} -----")
            print(f"Current Price: {current_price} USDT")
            
            # Use DOGE-specific contract details
            contract_size = DOGE_CONFIG['contract_size']  # 100 DOGE per contract
            min_amount = DOGE_CONFIG['min_contracts']     # 0.01 contracts minimum
            amount_precision = DOGE_CONFIG['precision']   # 2 decimal places
            
            print(f"Contract Size: {contract_size} DOGE per contract")
            print(f"Minimum Amount: {min_amount} contracts")
            print(f"Amount Precision: {amount_precision} decimal places")
            
            # Calculate position size with improved error handling
            balance_info = self.exchange.fetch_balance()
            available_balance = float(balance_info['USDT']['free'])
            
            # Also check the actual margin available for trading
            margin_available = float(balance_info.get('USDT', {}).get('free', 0))
            if 'info' in balance_info and 'data' in balance_info['info']:
                for item in balance_info['info']['data']:
                    if item.get('ccy') == 'USDT' and 'availEq' in item:
                        margin_available = float(item['availEq'])
                        break
            
            print(f"\n----- Account Information -----")
            print(f"Available Balance: {available_balance} USDT")
            print(f"Available Margin: {margin_available} USDT")
            
            # Calculate position size based on risk percentage (1% of balance)
            actual_available = min(available_balance, margin_available)
            
            # Scale position size based on confidence level
            # Normalize confidence between min_confidence and 1.0
            confidence_scale = (confidence - RISK_CONFIG['min_confidence']) / (1.0 - RISK_CONFIG['min_confidence'])
            # Ensure it's between 0.3 (minimum) and 1.0 (maximum)
            confidence_scale = max(0.3, min(1.0, confidence_scale))
            
            # Apply scaled risk
            scaled_risk = RISK_CONFIG['position_size_pct'] * confidence_scale
            risk_amount_usdt = actual_available * scaled_risk
            
            print(f"Confidence: {confidence:.2f}, Scale: {confidence_scale:.2f}")
            print(f"Risk Amount ({scaled_risk*100:.2f}%): {risk_amount_usdt} USDT")
            
            # Apply leverage to calculate notional value
            leveraged_value = risk_amount_usdt * self.leverage
            print(f"Leveraged Value: {leveraged_value} USDT")
            
            # Calculate contract quantity for DOGE specifically
            try:
                contract_value = current_price * contract_size  # Value of one contract in USDT
                contract_quantity = leveraged_value / contract_value
                
                print(f"Contract calculation: {leveraged_value} USDT / ({current_price} USDT * {contract_size} DOGE) = {contract_quantity} contracts")
                
                # Round to 2 decimal places for DOGE
                contract_quantity = round(contract_quantity, amount_precision)
                print(f"Rounded to {amount_precision} decimal place: {contract_quantity} contracts")
                
                # Check against minimum amount (0.01 contracts for DOGE)
                if contract_quantity < min_amount:
                    print(f"Increasing quantity to minimum: {min_amount} contracts")
                    contract_quantity = min_amount
                
                # Adjust for insufficient margin by reducing position size if needed
                original_quantity = contract_quantity
                for reduction_factor in [1.0, 0.8, 0.5, 0.3, 0.2]:
                    contract_quantity = round(original_quantity * reduction_factor, amount_precision)
                    if contract_quantity < min_amount:
                        contract_quantity = min_amount
                    
                    actual_exposure = contract_quantity * contract_value
                    margin_required = actual_exposure / self.leverage
                    
                    print(f"Trying with {contract_quantity} contracts")
                    print(f"Required margin: {margin_required:.2f} USDT")
                    print(f"Available margin: {margin_available:.2f} USDT")
                    
                    if margin_required <= margin_available * 0.95:  # 5% safety buffer
                        print(f"Using {contract_quantity} contracts ({reduction_factor*100:.0f}% of original calculation)")
                        break
                        
                    if reduction_factor == 0.2:
                        if min_amount * contract_value / self.leverage <= margin_available * 0.95:
                            contract_quantity = min_amount
                            print(f"Using minimum order size: {min_amount} contracts")
                        else:
                            print(f"Insufficient margin even for minimum contract size. Cannot trade.")
                            return False
                
                actual_risk_pct = (contract_quantity * contract_value / self.leverage) / actual_available
                print(f"Actual exposure: {contract_quantity * contract_value:.2f} USDT ({actual_risk_pct*100:.2f}% of balance with leverage)")
                
                desired_size = contract_quantity * (1 if signal > 0 else -1)
                
            except (TypeError, ValueError, ZeroDivisionError) as e:
                print(f"Error calculating position size: {e}")
                return False
            
            # Set leverage with retry
            max_leverage_attempts = 3
            leverage_set = False
            
            for attempt in range(max_leverage_attempts):
                try:
                    self.exchange.set_leverage(self.leverage, self.symbol)
                    leverage_set = True
                    print(f"Leverage set to {self.leverage}x")
                    break
                except Exception as e:
                    print(f"Leverage setting attempt {attempt + 1} failed: {str(e)}")
                    time.sleep(1)
            
            if not leverage_set:
                print("Failed to set leverage after multiple attempts")
                return False
            
            # Check if we have an existing position
            if current_position:
                current_side = current_position['side']
                signal_side = 'long' if signal > 0 else 'short'
                
                if (signal > 0 and current_side == 'short') or (signal < 0 and current_side == 'long'):
                    print(f"Signal direction change detected: {current_side} -> {signal_side}")
                    
                    try:
                        close_order = self.exchange.create_order(
                            self.symbol,
                            'market',
                            'buy' if current_side == 'short' else 'sell',
                            current_position['size'],
                            params={'reduceOnly': True, 'posSide': current_side}
                        )
                        
                        print(f"Closed existing {current_side} position: {close_order['id']}")
                        print(f"Position size: {current_position['size']} contracts")
                        print(f"Entry price: {current_position['entry_price']}")
                        
                        current_position = None
                    except Exception as e:
                        print(f"Error closing position: {e}")
                        print(traceback.format_exc())
                        return False
                
                elif (signal > 0 and current_side == 'long') or (signal < 0 and current_side == 'short'):
                    if confidence > (RISK_CONFIG['min_confidence'] + 0.05):
                        print(f"Adding to existing {current_side} position")
                        print(f"Current position: {current_position['size']} contracts at {current_position['entry_price']}")
                        
                        total_position_value = (current_position['size'] * current_position['entry_price']) + (abs(desired_size) * current_price)
                        total_size = current_position['size'] + abs(desired_size)
                        new_avg_entry = total_position_value / total_size
                        
                        side = 'buy' if signal > 0 else 'sell'
                        try:
                            order = self.exchange.create_order(
                                self.symbol,
                                'market',
                                side,
                                abs(desired_size),
                                None,
                                {'tdMode': 'cross', 'posSide': current_side}
                            )
                            
                            print(f"Added to position: {order['id']}")
                            print(f"Additional size: {abs(desired_size)} contracts")
                            print(f"New total size: {total_size} contracts")
                            print(f"New average entry: {new_avg_entry}")
                            
                            if self.symbol in self.active_positions:
                                self.active_positions[self.symbol].update({
                                    'size': total_size,
                                    'avg_entry': new_avg_entry,
                                    'initial_stop': new_avg_entry * (1 - RISK_CONFIG['trailing_stop_initial'] if signal > 0 else 1 + RISK_CONFIG['trailing_stop_initial'])
                                })
                            
                            return True
                        except ccxt.InsufficientFunds as e:
                            print(f"Insufficient funds when adding to position: {e}")
                            try:
                                smaller_size = abs(desired_size) * 0.5
                                if smaller_size >= min_amount:
                                    print(f"Retrying with smaller position: {smaller_size} contracts")
                                    order = self.exchange.create_order(
                                        self.symbol,
                                        'market',
                                        side,
                                        smaller_size,
                                        None,
                                        {'tdMode': 'cross', 'posSide': current_side}
                                    )
                                    print(f"Successfully added smaller position: {order['id']}")
                                    return True
                                else:
                                    print(f"Cannot reduce further below minimum: {min_amount}")
                                    return False
                            except Exception as inner_e:
                                print(f"Error with smaller position: {inner_e}")
                                return False
                        except Exception as e:
                            print(f"Error adding to position: {e}")
                            return False
                    else:
                        print(f"Not adding to position - confidence {confidence:.2f} below threshold {RISK_CONFIG['min_confidence'] + 0.05:.2f}")
                        return False
            
            if not current_position and abs(desired_size) >= min_amount:
                side = 'buy' if signal > 0 else 'sell'
                pos_side = 'long' if signal > 0 else 'short'
                
                try:
                    order_params = {
                        'tdMode': 'cross',
                        'posSide': pos_side,
                    }

                    print(f"Placing {side} order for {abs(desired_size)} contracts of {self.symbol}")
                
                    order = self.exchange.create_order(
                        self.symbol,
                        'market',
                        side,
                        abs(desired_size),
                        None,
                        order_params
                    )
                    
                    print(f"New position opened: {order['id']}")
                    print(f"Trade details: {side} {abs(desired_size)} contracts at ~{current_price}")
                    
                    self.active_positions[self.symbol] = {
                        'side': pos_side,
                        'size': abs(desired_size),
                        'entry_price': current_price,
                        'current_price': current_price,
                        'initial_stop': current_price * (1 - RISK_CONFIG['trailing_stop_initial'] if signal > 0 else 1 + RISK_CONFIG['trailing_stop_initial']),
                        'current_stop': current_price * (1 - RISK_CONFIG['trailing_stop_initial'] if signal > 0 else 1 + RISK_CONFIG['trailing_stop_initial']),
                        'highest_price': current_price if signal > 0 else float('-inf'),
                        'lowest_price': current_price if signal < 0 else float('inf')
                    }
                    
                    return True
                except ccxt.InsufficientFunds as e:
                    print(f"Insufficient funds error: {e}")
                    try:
                        smaller_size = abs(desired_size) * 0.5
                        if smaller_size >= min_amount:
                            print(f"Retrying with smaller position: {smaller_size} contracts")
                            order = self.exchange.create_order(
                                self.symbol,
                                'market',
                                side,
                                smaller_size,
                                None,
                                {'tdMode': 'cross', 'posSide': pos_side}
                            )
                            print(f"Successfully opened smaller position: {order['id']}")
                            
                            self.active_positions[self.symbol] = {
                                'side': pos_side,
                                'size': smaller_size,
                                'entry_price': current_price,
                                'current_price': current_price,
                                'initial_stop': current_price * (1 - RISK_CONFIG['trailing_stop_initial'] if signal > 0 else 1 + RISK_CONFIG['trailing_stop_initial']),
                                'current_stop': current_price * (1 - RISK_CONFIG['trailing_stop_initial'] if signal > 0 else 1 + RISK_CONFIG['trailing_stop_initial']),
                                'highest_price': current_price if signal > 0 else float('-inf'),
                                'lowest_price': current_price if signal < 0 else float('inf')
                            }
                            return True
                    except Exception as inner_e:
                        print(f"Error with smaller position: {inner_e}")
                        return False
                except Exception as e:
                    print(f"Error opening new position: {e}")
                    print(traceback.format_exc())
                    return False
            else:
                if current_position:
                    print("Position unchanged - signal aligns with current position")
                else:
                    print(f"Position size {abs(desired_size)} is below minimum {min_amount}")
                return False
                
        except Exception as e:
            print(f"Trade execution error: {str(e)}")
            print(traceback.format_exc())
            return False

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
        """Run backtest on historical data"""
        try:
            print("\nRunning backtest on historical data...")
            
            # Split data for backtesting
            train_size = int(len(df) * 0.7)
            
            # Extract features for training and testing
            train_features = features[:train_size]
            train_labels = labels[:train_size]
            test_features = features[train_size:]
            test_labels = labels[train_size:]
            
            # Train model
            self.model.fit(train_features, train_labels)
            
            # Get feature importance
            feature_columns = train_features.columns
            self.feature_importance = pd.Series(
                self.model.feature_importances_, 
                index=feature_columns
            ).sort_values(ascending=False)
            
            # Print top 10 features
            print("\nTop 10 Most Important Features:")
            print(self.feature_importance.head(10))
            
            # Make predictions on test data
            predictions = self.model.predict(test_features)
            confidence_scores = self.model.predict_proba(test_features)
            
            # Calculate accuracy
            correct_predictions = sum(predictions == test_labels)
            accuracy = correct_predictions / len(test_labels)
            
            # Simulate trading
            balance = 1000  # Starting with 1000 USDT
            initial_balance = balance
            position = None
            test_prices = df['Close'][train_size:].values
            
            # Track trades for analysis
            trades = []
            
            for i in range(len(predictions)):
                confidence = max(confidence_scores[i])
                signal = 1 if predictions[i] == 1 else -1
                    
                # Only trade if confidence is sufficient
                if confidence > RISK_CONFIG['min_confidence']:
                    # Close existing position if direction changes
                    if position and ((signal > 0 and position['side'] < 0) or (signal < 0 and position['side'] > 0)):
                        # Calculate profit/loss
                        pnl = position['size'] * (test_prices[i] - position['entry']) * position['side']
                        balance += pnl
                        
                        trades.append({
                            'exit_price': test_prices[i],
                            'pnl': pnl,
                            'balance': balance,
                            'hold_bars': i - position['entry_bar']
                        })
                        
                        position = None
                    
                    # Open new position if none exists
                    if position is None:
                        position_size = (balance * RISK_CONFIG['position_size_pct']) / test_prices[i]
                        
                        position = {
                            'side': signal,
                            'entry': test_prices[i],
                            'size': position_size,
                            'entry_bar': i,
                            'trailing_stop': test_prices[i] * (1 - RISK_CONFIG['trailing_stop_initial'] if signal > 0 else 1 + RISK_CONFIG['trailing_stop_initial'])
                        }
                        
                        trades.append({
                            'entry_bar': i,
                            'entry_price': test_prices[i],
                            'side': 'long' if signal > 0 else 'short',
                            'size': position_size
                        })
            
                # Update trailing stop if position exists
            if position:
                    if position['side'] > 0:  # Long position
                        # Update highest price seen
                        highest_price = max(test_prices[i], position.get('highest_price', test_prices[i]))
                        position['highest_price'] = highest_price
                        
                        # Calculate new stop based on volatility
                        price_range = df['High'][train_size+i] - df['Low'][train_size+i]
                        atr_factor = min(max(price_range / test_prices[i], RISK_CONFIG['trailing_stop_min']), RISK_CONFIG['trailing_stop_max'])
                        
                        new_stop = highest_price * (1 - atr_factor)
                        
                        # Only move stop up, never down
                        if new_stop > position['trailing_stop']:
                            position['trailing_stop'] = new_stop
                        
                        # Check if stop is hit
                        if test_prices[i] < position['trailing_stop']:
                            pnl = position['size'] * (position['trailing_stop'] - position['entry'])
                            balance += pnl
                            
                            trades.append({
                                'exit_price': position['trailing_stop'],
                                'pnl': pnl,
                                'balance': balance,
                                'hold_bars': i - position['entry_bar'],
                                'exit_type': 'trailing_stop'
                            })
                            
                            position = None
                    
                    else:  # Short position
                        # Update lowest price seen
                        lowest_price = min(test_prices[i], position.get('lowest_price', test_prices[i]))
                        position['lowest_price'] = lowest_price
                        
                        # Calculate new stop based on volatility
                        price_range = df['High'][train_size+i] - df['Low'][train_size+i]
                        atr_factor = min(max(price_range / test_prices[i], RISK_CONFIG['trailing_stop_min']), RISK_CONFIG['trailing_stop_max'])
                        
                        new_stop = lowest_price * (1 + atr_factor)
                        
                        # Only move stop down, never up
                        if new_stop < position['trailing_stop'] or position['trailing_stop'] == 0:
                            position['trailing_stop'] = new_stop
                        
                        # Check if stop is hit
                        if test_prices[i] > position['trailing_stop']:
                            pnl = position['size'] * (position['entry'] - position['trailing_stop'])
                            balance += pnl
                            
                            trades.append({
                                'exit_price': position['trailing_stop'],
                                'pnl': pnl,
                                'balance': balance,
                                'hold_bars': i - position['entry_bar'],
                                'exit_type': 'trailing_stop'
                            })
                            
                            position = None
            
            # Close any remaining position at the end
            if position:
                if position['side'] > 0:
                    pnl = position['size'] * (test_prices[-1] - position['entry'])
                else:
                    pnl = position['size'] * (position['entry'] - test_prices[-1])
                    
                balance += pnl
                
                trades.append({
                    'exit_price': test_prices[-1],
                    'pnl': pnl,
                    'balance': balance,
                    'hold_bars': len(test_prices) - position['entry_bar'],
                    'exit_type': 'end_of_test'
                })
            
            # Calculate performance metrics
            winning_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
            total_trades = sum(1 for t in trades if 'pnl' in t)
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            profit_factor = sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0) / abs(sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0)) if sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0) != 0 else float('inf')
            
            returns = ((balance - initial_balance) / initial_balance) * 100
            
            # Print backtest results
            print(f"\nBacktest Results:")
            print(f"Accuracy: {accuracy:.2%}")
            print(f"Win Rate: {win_rate:.2%}")
            print(f"Total Trades: {total_trades}")
            print(f"Profit Factor: {profit_factor:.2f}")
            print(f"Final Balance: {balance:.2f} USDT")
            print(f"Return: {returns:.2f}%")
            
            # Return multiple values - ONLY check accuracy, not returns
            is_successful = accuracy > 0.5
            return is_successful, accuracy * 100, balance, returns
            
        except Exception as e:
            print(f"Backtest error: {str(e)}")
            print("Stack trace:", traceback.format_exc())
            return False, 0, 0, 0

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
            
            # Run backtest - properly unpack the multi-value return
            is_successful, accuracy, final_balance, returns = self.backtest(backtest_df, backtest_features[:-1], backtest_labels)
            
            if not is_successful:
                print("Backtest results unsatisfactory. Continuing anyway with trading.")
            
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
                    prediction, confidence = self.predict_with_ensemble(features)
                    
                    # Log prediction details whether we trade or not
                    print(f"Prediction: {'UP' if prediction == 1 else 'DOWN'}, Confidence: {confidence:.2f}, Min Threshold: {RISK_CONFIG['min_confidence']:.2f}")
                    
                    # Execute trade with enhanced confidence check
                    if confidence > RISK_CONFIG['min_confidence']:
                        signal = 1 if prediction == 1 else -1
                        print(f"!!! Signal triggered: {'LONG' if signal > 0 else 'SHORT'} with confidence {confidence:.2f}")
                        
                        # Check if we already have a position in the same direction
                        has_matching_position = False
                        for pos in positions:
                            if pos['symbol'] == self.symbol and float(pos['contracts']) > 0:
                                current_side = pos['side']
                                if (signal > 0 and current_side == 'long') or (signal < 0 and current_side == 'short'):
                                    has_matching_position = True
                                    print(f"Already have a matching {current_side} position - will add to it if confidence is sufficient")
                                    break
                        
                        # Adjust signal strength based on regime
                        current_regime = features['regime_volatile'].iloc[-1]
                        if current_regime == 1:
                            signal *= 0.7  # Reduce position size in volatile markets
                            
                        # Only execute trade if not already trading in same direction or confidence is high enough to add
                        if not has_matching_position or confidence > (RISK_CONFIG['min_confidence'] + 0.01):
                            self.execute_trade(signal, confidence)
                        else:
                            print(f"Skipping trade - already have a position in the same direction and confidence {confidence:.2f} not high enough to add")
                    else:
                        print(f"No trade - confidence {confidence:.2f} below threshold {RISK_CONFIG['min_confidence']:.2f}")
                    
                    # If restarting and no positions, exit
                    if self.is_restarting and not self.check_active_positions():
                        print("All positions closed, restarting bot...")
                        return True
                    
                    # Wait before next iteration
                    print(f"Waiting 60 seconds until next check...")
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
    try:
        print(f"\nRunning backtest for DOGE/USDT:USDT...")
        bot = MLTradingBot()
        bot.symbol = 'DOGE/USDT:USDT'
        
        try:
            # Get market info before testing
            market = bot.exchange.market(bot.symbol)
            ticker = bot.exchange.fetch_ticker(bot.symbol)
            current_price = float(ticker['last'])
            
            print("\nDOGE Market Information:")
            print(f"Current Price: {current_price} USDT")
            print(f"Contract Size: {DOGE_CONFIG['contract_size']} DOGE")
            print(f"Minimum Contract: {DOGE_CONFIG['min_contracts']} contracts")
            print(f"Precision: {DOGE_CONFIG['precision']} decimal places")
            
            # Fetch historical data
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
            
            # Run backtest and unpack the multiple return values
            is_successful, accuracy, final_balance, returns = bot.backtest(backtest_df, backtest_features[:-1], backtest_labels)
            
            print("\n=== DOGE Backtest Results ===")
            print(f"Accuracy: {accuracy:.2f}%")
            print(f"Returns: {returns:.2f}%")
            print(f"Final Balance: {final_balance:.2f} USDT")
            
            # Start live trading if backtest was successful
            if is_successful:
                print("\nBacktest successful! Starting live trading...")
                bot.run()
            else:
                print("\nBacktest not successful. Accuracy below threshold.")
                print("Waiting 15 minutes before retrying...")
                time.sleep(900)  # Wait 15 minutes
                run_backtest_all_coins()  # Try again
                
        except Exception as e:
            print(f"Error in DOGE backtest: {str(e)}")
            print(traceback.format_exc())
            time.sleep(60)
            run_backtest_all_coins()  # Try again after error
    
    except Exception as e:
        print(f"Critical error: {str(e)}")
        print(traceback.format_exc())
        print("Restarting in 5 seconds...")
        time.sleep(5)
        run_backtest_all_coins()

if __name__ == "__main__":
    print("Starting DOGE-focused ML Trading Bot...\n")
    run_backtest_all_coins()
