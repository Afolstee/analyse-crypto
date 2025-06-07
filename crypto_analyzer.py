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
import gc
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CryptoDataManager:
    def __init__(self, coin_ids, db_path='crypto_data.db'):
        self.coin_ids = coin_ids
        self.db_path = db_path
        self.setup_database()
        self.scaler = StandardScaler()
        # Use much smaller model for memory efficiency
        self.model = RandomForestClassifier(
            n_estimators=5,  # Reduced from 100
            max_depth=3,     # Limited depth
            random_state=42,
            n_jobs=1,        # Single thread to avoid memory issues
            max_samples=0.5  # Use only 50% of training data
        )
        self.is_scaler_fitted = False
        self.is_model_fitted = False
        # Reduce timeout even more for deployment
        self.db_timeout = 1.0
        self.max_retries = 1

    def get_db_connection(self):
        """Get database connection with timeout and retry logic"""
        for attempt in range(self.max_retries):
            try:
                conn = sqlite3.connect(self.db_path, timeout=self.db_timeout)
                conn.execute('PRAGMA journal_mode=WAL')
                conn.execute('PRAGMA synchronous=NORMAL')
                # Add memory optimization
                conn.execute('PRAGMA cache_size=1000')  # Smaller cache
                conn.execute('PRAGMA temp_store=MEMORY')
                return conn
            except sqlite3.OperationalError as e:
                logger.warning(f"Database connection attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(0.05)  # Shorter wait
                else:
                    # Return None instead of raising to prevent crashes
                    logger.error(f"Database connection failed after {self.max_retries} attempts")
                    return None

    def setup_database(self):
        try:
            conn = self.get_db_connection()
            if conn is None:
                logger.warning("Database connection failed during setup - using in-memory fallback")
                return
                
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
            # Don't raise - allow fallback behavior

    def store_price_data(self, prices):
        try:
            conn = self.get_db_connection()
            if conn is None:
                logger.warning("Cannot store price data - database unavailable")
                return
                
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            for price in prices:
                try:
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
                except Exception as e:
                    logger.warning(f"Failed to store price for {price.get('id', 'unknown')}: {e}")
                    continue
            
            conn.commit()
            conn.close()
            logger.info(f"Stored price data for {len(prices)} coins")
        except Exception as e:
            logger.error(f"Failed to store price data: {e}")

    def store_news_data(self, news):
        try:
            conn = self.get_db_connection()
            if conn is None:
                logger.warning("Cannot store news data - database unavailable")
                return
                
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            for article in news:
                try:
                    # Simplified sentiment analysis to reduce memory usage
                    title = article.get('title', '')[:200]  # Limit title length
                    if title:
                        blob = TextBlob(title)
                        sentiment = blob.sentiment.polarity
                    else:
                        sentiment = 0.0
                    
                    conn.execute('''
                        INSERT INTO news_history 
                        (timestamp, title, currencies, sentiment)
                        VALUES (?, ?, ?, ?)
                    ''', (
                        timestamp,
                        title,
                        json.dumps([c['code'] for c in article.get('currencies', [])]),
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

    def get_analysis_features(self, coin_id, lookback_hours=6, end_time=None):
        """Get analysis features with fallback for database issues"""
        try:
            conn = self.get_db_connection()
            if conn is None:
                logger.warning(f"Database unavailable for {coin_id} - using fallback features")
                return self._get_fallback_features()
            
            if end_time is None:
                end_time = datetime.now()
                
            if isinstance(end_time, str):
                end_time = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
                    
            cutoff_time = end_time - timedelta(hours=lookback_hours)
            end_time_str = end_time.strftime('%Y-%m-%d %H:%M:%S')
            cutoff_time_str = cutoff_time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Get minimal price data
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT price, volume FROM price_history 
                    WHERE coin_id = ? AND timestamp > ? AND timestamp <= ?
                    ORDER BY timestamp DESC
                    LIMIT 5
                ''', (coin_id, cutoff_time_str, end_time_str))
                
                rows = cursor.fetchall()
                conn.close()
                
                if len(rows) < 2:
                    logger.warning(f"Insufficient price data for {coin_id}: {len(rows)} records")
                    return self._get_fallback_features()
                
                # Memory-efficient calculations
                prices = [float(row[0]) for row in rows if row[0] is not None]
                volumes = [float(row[1]) for row in rows if row[1] is not None]
                
                if len(prices) < 2:
                    return self._get_fallback_features()
                
                # Simple calculations
                price_mean = sum(prices) / len(prices)
                price_variance = sum((p - price_mean) ** 2 for p in prices) / len(prices)
                price_volatility = max(0.0, price_variance ** 0.5)
                
                volume_mean = sum(volumes) / len(volumes) if volumes else 0
                
                # Simple momentum
                price_momentum = 0.0
                if len(prices) >= 2 and prices[-1] != 0:
                    price_momentum = (prices[0] - prices[-1]) / prices[-1]
                
                features = {
                    'price_volatility': min(1.0, max(0.0, float(price_volatility))),
                    'volume_change': min(10.0, max(0.0, float(volume_mean / 1000000))),
                    'price_momentum': min(1.0, max(-1.0, float(price_momentum))),
                    'avg_sentiment': 0.0,
                    'sentiment_volatility': 0.1
                }
                
                # Clean up memory
                del prices, volumes
                gc.collect()
                
                return features
                
            except Exception as e:
                logger.error(f"Failed to process data for {coin_id}: {e}")
                if conn:
                    conn.close()
                return self._get_fallback_features()
            
        except Exception as e:
            logger.error(f"Error in get_analysis_features for {coin_id}: {e}")
            return self._get_fallback_features()

    def _get_fallback_features(self):
        """Return default features when database is unavailable"""
        return {
            'price_volatility': 0.05,
            'volume_change': 0.01,
            'price_momentum': 0.0,
            'avg_sentiment': 0.0,
            'sentiment_volatility': 0.1
        }

    def fit_scaler(self):
        """Fit the scaler with minimal synthetic data"""
        try:
            logger.info("Fitting scaler with minimal synthetic data...")
            
            # Minimal synthetic data to save memory
            feature_data = []
            for i in range(20):  # Reduced from 50
                feature_data.append([
                    abs(np.random.normal(0.05, 0.02)),  # price_volatility
                    abs(np.random.normal(0.01, 0.005)), # volume_change
                    np.random.normal(0.0, 0.005),       # price_momentum
                    np.random.normal(0.0, 0.1),         # avg_sentiment
                    abs(np.random.normal(0.1, 0.05))    # sentiment_volatility
                ])
            
            self.scaler.fit(feature_data)
            self.is_scaler_fitted = True
            
            # Clean up
            del feature_data
            gc.collect()
            
            logger.info("✅ Scaler fitted successfully")
            
        except Exception as e:
            logger.error(f"Failed to fit scaler: {e}")
            raise

    def fit_model(self):
        """Fit model with minimal synthetic data to prevent memory issues"""
        try:
            logger.info("Training lightweight model...")
            
            # Minimal training data
            X = []
            y = []
            
            # Generate only 30 samples to minimize memory usage
            for i in range(30):
                price_vol = abs(np.random.normal(0.05, 0.02))
                volume_change = abs(np.random.normal(0.01, 0.005))
                momentum = np.random.normal(0.0, 0.005)
                sentiment = np.random.normal(0.0, 0.1)
                sent_vol = abs(np.random.normal(0.1, 0.05))
                
                features = [price_vol, volume_change, momentum, sentiment, sent_vol]
                X.append(features)
                
                # Simple labeling logic
                if momentum > 0.002 and sentiment > 0.05:
                    label = 1  # pump
                elif momentum < -0.002 and sentiment < -0.05:
                    label = -1  # dump
                else:
                    label = 0  # stable
                    
                y.append(label)
            
            # Ultra-lightweight model
            self.model = RandomForestClassifier(
                n_estimators=3,    # Minimal trees
                max_depth=2,       # Very shallow
                random_state=42,
                n_jobs=1,          # Single thread
                max_samples=0.8,   # Use less data per tree
                max_features=3     # Limit features per split
            )
            
            self.model.fit(X, y)
            self.is_model_fitted = True
            
            # Clean up training data
            del X, y
            gc.collect()
            
            logger.info("✅ Lightweight model trained successfully")
            
        except Exception as e:
            logger.error(f"Failed to train model: {e}")
            # Create an even simpler fallback
            try:
                logger.info("Creating ultra-simple fallback model...")
                simple_X = [[0.05, 0.01, 0.0, 0.0, 0.1], [0.1, 0.02, 0.01, 0.1, 0.2], [0.02, 0.005, -0.01, -0.1, 0.05]]
                simple_y = [0, 1, -1]
                
                self.model = RandomForestClassifier(n_estimators=1, max_depth=1, random_state=42)
                self.model.fit(simple_X, simple_y)
                self.is_model_fitted = True
                logger.info("✅ Fallback model created")
            except Exception as fallback_error:
                logger.error(f"Even fallback model failed: {fallback_error}")
                raise

    def predict_movement(self, coin_id):
        """Predict price movement with error handling"""
        try:
            # Initialize components if needed
            if not self.is_scaler_fitted:
                self.fit_scaler()
            
            if not self.is_model_fitted:
                self.fit_model()
            
            # Get features
            features = self.get_analysis_features(coin_id, lookback_hours=3)  # Shorter lookback
            if features is None:
                features = self._get_fallback_features()
                
            feature_array = np.array([[
                features['price_volatility'],
                features['volume_change'],
                features['price_momentum'],
                features['avg_sentiment'],
                features['sentiment_volatility']
            ]])
            
            # Make prediction
            try:
                scaled_features = self.scaler.transform(feature_array)
                prediction = self.model.predict(scaled_features)[0]
                probabilities = self.model.predict_proba(scaled_features)[0]
                confidence = float(max(probabilities))
            except Exception as pred_error:
                logger.warning(f"Prediction error for {coin_id}: {pred_error}")
                prediction = 0
                confidence = 0.33
            
            # Clean up
            del feature_array
            gc.collect()
            
            return {
                'prediction': int(prediction),
                'confidence': confidence,
                'features': features
            }
            
        except Exception as e:
            logger.error(f"Failed to predict movement for {coin_id}: {e}")
            return {
                'prediction': 0,
                'confidence': 0.33,
                'features': self._get_fallback_features()
            }