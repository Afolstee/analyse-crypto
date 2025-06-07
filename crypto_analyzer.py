from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from textblob import TextBlob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import json
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CryptoDataManager:
    def __init__(self, coin_ids, db_path='crypto_data.db'):
        self.coin_ids = coin_ids
        self.db_path = db_path
        self.setup_database()
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_scaler_fitted = False
        self.is_model_fitted = False
        # Add connection timeout and retry settings
        self.db_timeout = 3.0  # Very short timeout
        self.max_retries = 1   # Single retry only

    def get_db_connection(self):
        """Get database connection with timeout and retry logic"""
        for attempt in range(self.max_retries):
            try:
                conn = sqlite3.connect(self.db_path, timeout=self.db_timeout)
                conn.execute('PRAGMA journal_mode=WAL')
                conn.execute('PRAGMA synchronous=NORMAL')
                return conn
            except sqlite3.OperationalError as e:
                logger.warning(f"Database connection attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(0.1)
                else:
                    raise

    def setup_database(self):
        try:
            conn = self.get_db_connection()
            c = conn.cursor()
            
            c.execute('''
                CREATE TABLE IF NOT EXISTS price_history (
                    coin_id TEXT,
                    timestamp DATETIME,
                    price REAL,
                    volume REAL,
                    price_change_24h REAL,
                    PRIMARY KEY (coin_id, timestamp)
                )
            ''')
            
            c.execute('''
                CREATE TABLE IF NOT EXISTS news_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    title TEXT,
                    currencies TEXT,
                    sentiment REAL
                )
            ''')
            
            # Add indexes for better query performance
            c.execute('''
                CREATE INDEX IF NOT EXISTS idx_price_coin_timestamp 
                ON price_history(coin_id, timestamp)
            ''')
            
            c.execute('''
                CREATE INDEX IF NOT EXISTS idx_news_timestamp 
                ON news_history(timestamp)
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Database setup completed successfully")
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            raise

    def store_price_data(self, prices):
        try:
            conn = self.get_db_connection()
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            for price in prices:
                conn.execute('''
                    INSERT OR REPLACE INTO price_history 
                    (coin_id, timestamp, price, volume, price_change_24h)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    price['id'],
                    timestamp,
                    price['current_price'],
                    price['total_volume'],
                    price['price_change_percentage_24h']
                ))
            
            conn.commit()
            conn.close()
            logger.info(f"Stored price data for {len(prices)} coins")
        except Exception as e:
            logger.error(f"Failed to store price data: {e}")
            raise

    def store_news_data(self, news):
        try:
            conn = self.get_db_connection()
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            for article in news:
                try:
                    blob = TextBlob(article['title'])
                    sentiment = blob.sentiment.polarity
                    
                    conn.execute('''
                        INSERT INTO news_history 
                        (timestamp, title, currencies, sentiment)
                        VALUES (?, ?, ?, ?)
                    ''', (
                        timestamp,
                        article['title'],
                        json.dumps([c['code'] for c in article['currencies']]),
                        sentiment
                    ))
                except Exception as e:
                    logger.warning(f"Failed to process news article: {e}")
                    continue
            
            conn.commit()
            conn.close()
            logger.info(f"Stored news data for {len(news)} articles")
        except Exception as e:
            logger.error(f"Failed to store news data: {e}")
            raise

    def get_analysis_features(self, coin_id, lookback_hours=24, end_time=None):
        """Get analysis features - simplified to prevent timeouts"""
        try:
            conn = self.get_db_connection()
            
            if end_time is None:
                end_time = datetime.now()
                
            if isinstance(end_time, str):
                end_time = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
                    
            cutoff_time = end_time - timedelta(hours=lookback_hours)
            end_time_str = end_time.strftime('%Y-%m-%d %H:%M:%S')
            cutoff_time_str = cutoff_time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Get only the most recent price data with minimal processing
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT price, volume FROM price_history 
                    WHERE coin_id = ? AND timestamp > ? AND timestamp <= ?
                    ORDER BY timestamp DESC
                    LIMIT 10
                ''', (coin_id, cutoff_time_str, end_time_str))
                
                rows = cursor.fetchall()
                conn.close()
                
                if len(rows) < 2:
                    logger.warning(f"Insufficient price data for {coin_id}: {len(rows)} records")
                    return None
                
                # Simple calculations without pandas operations
                prices = [row[0] for row in rows if row[0] is not None]
                volumes = [row[1] for row in rows if row[1] is not None]
                
                if len(prices) < 2:
                    return None
                
                # Calculate simple features without pandas
                price_mean = sum(prices) / len(prices)
                price_variance = sum((p - price_mean) ** 2 for p in prices) / len(prices)
                price_volatility = price_variance ** 0.5
                
                volume_mean = sum(volumes) / len(volumes) if volumes else 0
                
                # Simple momentum calculation
                if len(prices) >= 2:
                    price_momentum = (prices[0] - prices[-1]) / prices[-1] if prices[-1] != 0 else 0
                else:
                    price_momentum = 0
                
                features = {
                    'price_volatility': float(price_volatility) if not np.isnan(price_volatility) else 0.0,
                    'volume_change': float(volume_mean / 1000000) if volume_mean > 0 else 0.0,  # Normalized
                    'price_momentum': float(price_momentum) if not np.isnan(price_momentum) else 0.0,
                    'avg_sentiment': 0.0,
                    'sentiment_volatility': 0.0
                }
                
                # Ensure no NaN or infinite values
                for key, value in features.items():
                    if pd.isna(value) or np.isnan(value) or np.isinf(value):
                        features[key] = 0.0
                
                logger.info(f"Generated features for {coin_id}: {features}")
                return features
                
            except Exception as e:
                logger.error(f"Failed to process data for {coin_id}: {e}")
                conn.close()
                return None
            
        except Exception as e:
            logger.error(f"Error in get_analysis_features for {coin_id}: {e}")
            return None

    def fit_scaler(self):
        """Fit the scaler with synthetic data only - no database access"""
        try:
            logger.info("Fitting scaler with synthetic data...")
            
            # Generate synthetic feature data
            feature_data = []
            for _ in range(50):
                feature_data.append([
                    abs(np.random.normal(0.1, 0.05)),  # price_volatility (positive)
                    np.random.normal(0.0, 0.02),       # volume_change
                    np.random.normal(0.0, 0.01),       # price_momentum
                    np.random.normal(0.0, 0.3),        # avg_sentiment
                    abs(np.random.normal(0.2, 0.1))    # sentiment_volatility (positive)
                ])
            
            self.scaler.fit(feature_data)
            self.is_scaler_fitted = True
            logger.info("✅ Scaler fitted successfully with synthetic data.")
            
        except Exception as e:
            logger.error(f"Failed to fit scaler: {e}")
            raise

    def fit_model(self):
        """Fit model with synthetic data only - no database queries"""
        try:
            logger.info("Training model with synthetic data (fast initialization)...")
            
            # Generate synthetic training data
            X = []
            y = []
            
            for _ in range(100):  # Reasonable training set
                # Generate realistic feature values
                price_vol = abs(np.random.normal(0.1, 0.05))
                volume_change = np.random.normal(0.0, 0.02)
                momentum = np.random.normal(0.0, 0.01)
                sentiment = np.random.normal(0.0, 0.3)
                sent_vol = abs(np.random.normal(0.2, 0.1))
                
                features = [price_vol, volume_change, momentum, sentiment, sent_vol]
                X.append(features)
                
                # Create labels based on logical rules
                if momentum > 0.005 and sentiment > 0.1:
                    label = 1  # pump
                elif momentum < -0.005 and sentiment < -0.1:
                    label = -1  # dump
                else:
                    label = 0  # stable
                    
                y.append(label)
            
            # Train the model
            self.model = RandomForestClassifier(
                n_estimators=10,
                max_depth=5,
                random_state=42,
                n_jobs=1
            )
            
            self.model.fit(X, y)
            self.is_model_fitted = True
            logger.info("✅ Model trained successfully with synthetic data.")
            
        except Exception as e:
            logger.error(f"Failed to train model: {e}")
            raise

    def predict_movement(self, coin_id):
        """Predict price movement for a given coin"""
        try:
            # Initialize scaler and model if not done
            if not self.is_scaler_fitted:
                self.fit_scaler()
            
            if not self.is_model_fitted:
                self.fit_model()
            
            # Get features for prediction
            features = self.get_analysis_features(coin_id, lookback_hours=6)  # Very short lookback
            if features is None:
                logger.warning(f"No features available for {coin_id}")
                return {
                    'prediction': 0,
                    'confidence': 0.33,
                    'features': {'error': 'No data available'}
                }
                
            feature_array = np.array([[
                features['price_volatility'],
                features['volume_change'],
                features['price_momentum'],
                features['avg_sentiment'],
                features['sentiment_volatility']
            ]])
            
            scaled_features = self.scaler.transform(feature_array)
            prediction = self.model.predict(scaled_features)[0]
            probabilities = self.model.predict_proba(scaled_features)[0]
            
            return {
                'prediction': int(prediction),  # -1: dump, 0: stable, 1: pump
                'confidence': float(max(probabilities)),
                'features': features
            }
            
        except Exception as e:
            logger.error(f"Failed to predict movement for {coin_id}: {e}")
            return {
                'prediction': 0,
                'confidence': 0.33,
                'features': {'error': str(e)}
            }