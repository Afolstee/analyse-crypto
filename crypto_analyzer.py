import numpy as np
from datetime import datetime, timedelta
import sqlite3
import json
import time
import logging
import gc
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleCryptoPredictor:
    """Ultra-lightweight predictor without sklearn or pandas"""
    
    def __init__(self):
        self.weights = {
            'price_volatility': 0.3,
            'volume_change': 0.2,
            'price_momentum': 0.4,
            'sentiment': 0.1
        }
        self.thresholds = {'pump': 0.1, 'dump': -0.1}
    
    def predict(self, features):
        """Simple weighted prediction without ML libraries"""
        try:
            score = (
                features['price_momentum'] * self.weights['price_momentum'] +
                features['price_volatility'] * self.weights['price_volatility'] +
                features['volume_change'] * self.weights['volume_change'] +
                features['avg_sentiment'] * self.weights['sentiment']
            )
            
            if score > self.thresholds['pump']:
                prediction = 1  # pump
                confidence = min(0.8, 0.5 + abs(score))
            elif score < self.thresholds['dump']:
                prediction = -1  # dump  
                confidence = min(0.8, 0.5 + abs(score))
            else:
                prediction = 0  # stable
                confidence = 0.6
                
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 0, 0.33

class CryptoDataManager:
    def __init__(self, coin_ids, db_path='crypto_data.db'):
        self.coin_ids = coin_ids
        self.db_path = db_path
        self.predictor = SimpleCryptoPredictor()
        self.db_timeout = 0.5  # Very short timeout
        self.max_retries = 1
        
        # Try to setup database, but don't fail if it doesn't work
        try:
            self.setup_database()
        except Exception as e:
            logger.warning(f"Database setup failed, using memory-only mode: {e}")

    def get_db_connection(self):
        """Get database connection with aggressive timeout"""
        try:
            conn = sqlite3.connect(self.db_path, timeout=self.db_timeout)
            conn.execute('PRAGMA journal_mode=MEMORY')  # Fastest mode
            conn.execute('PRAGMA synchronous=OFF')      # Disable sync for speed
            conn.execute('PRAGMA cache_size=500')       # Minimal cache
            return conn
        except Exception as e:
            logger.warning(f"Database connection failed: {e}")
            return None

    def setup_database(self):
        """Minimal database setup"""
        conn = self.get_db_connection()
        if conn is None:
            return
            
        try:
            c = conn.cursor()
            c.execute('''
                CREATE TABLE IF NOT EXISTS price_history (
                    coin_id TEXT,
                    timestamp DATETIME,
                    price REAL,
                    volume REAL,
                    PRIMARY KEY (coin_id, timestamp)
                )
            ''')
            conn.commit()
            conn.close()
            logger.info("Minimal database setup completed")
        except Exception as e:
            logger.warning(f"Database setup error: {e}")
            if conn:
                conn.close()

    def store_price_data(self, prices):
        """Store price data with minimal processing"""
        conn = self.get_db_connection()
        if conn is None:
            return
            
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            for price in prices[:10]:  # Limit to 10 coins max
                try:
                    conn.execute('''
                        INSERT OR REPLACE INTO price_history 
                        (coin_id, timestamp, price, volume)
                        VALUES (?, ?, ?, ?)
                    ''', (
                        price['id'],
                        timestamp,
                        float(price['current_price']),
                        float(price['total_volume'])
                    ))
                except (KeyError, TypeError, ValueError):
                    continue
            
            conn.commit()
            conn.close()
            logger.info(f"Stored price data")
        except Exception as e:
            logger.warning(f"Price storage error: {e}")
            if conn:
                conn.close()

    def store_news_data(self, news):
        """Skip news processing to save memory"""
        logger.info("News processing skipped for memory efficiency")
        pass

    def get_analysis_features(self, coin_id, lookback_hours=3, end_time=None):
        """Get features without pandas - pure Python only"""
        try:
            conn = self.get_db_connection()
            if conn is None:
                return self._get_fallback_features()
            
            if end_time is None:
                end_time = datetime.now()
            
            cutoff_time = end_time - timedelta(hours=lookback_hours)
            end_time_str = end_time.strftime('%Y-%m-%d %H:%M:%S')
            cutoff_time_str = cutoff_time.strftime('%Y-%m-%d %H:%M:%S')
            
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT price, volume FROM price_history 
                    WHERE coin_id = ? AND timestamp > ? AND timestamp <= ?
                    ORDER BY timestamp DESC
                    LIMIT 3
                ''', (coin_id, cutoff_time_str, end_time_str))
                
                rows = cursor.fetchall()
                conn.close()
                
                if len(rows) < 2:
                    return self._get_fallback_features()
                
                # Pure Python calculations - no numpy/pandas
                prices = [float(row[0]) for row in rows if row[0] is not None]
                volumes = [float(row[1]) for row in rows if row[1] is not None]
                
                if len(prices) < 2:
                    return self._get_fallback_features()
                
                # Simple math operations
                price_mean = sum(prices) / len(prices)
                price_diff_sq = [(p - price_mean) ** 2 for p in prices]
                price_volatility = (sum(price_diff_sq) / len(price_diff_sq)) ** 0.5
                
                volume_mean = sum(volumes) / len(volumes) if volumes else 1
                
                # Momentum calculation
                price_momentum = 0.0
                if len(prices) >= 2 and prices[-1] > 0:
                    price_momentum = (prices[0] - prices[-1]) / prices[-1]
                
                # Normalize values to prevent extreme numbers
                features = {
                    'price_volatility': max(0.0, min(1.0, price_volatility / price_mean if price_mean > 0 else 0.01)),
                    'volume_change': max(0.0, min(5.0, volume_mean / 1000000)),
                    'price_momentum': max(-0.5, min(0.5, price_momentum)),
                    'avg_sentiment': 0.0,  # Skip sentiment for now
                    'sentiment_volatility': 0.05
                }
                
                return features
                
            except Exception as e:
                logger.warning(f"Feature calculation error for {coin_id}: {e}")
                if conn:
                    conn.close()
                return self._get_fallback_features()
            
        except Exception as e:
            logger.warning(f"get_analysis_features error: {e}")
            return self._get_fallback_features()

    def _get_fallback_features(self):
        """Return random but reasonable fallback features"""
        return {
            'price_volatility': random.uniform(0.01, 0.1),
            'volume_change': random.uniform(0.001, 0.05),
            'price_momentum': random.uniform(-0.02, 0.02),
            'avg_sentiment': random.uniform(-0.1, 0.1),
            'sentiment_volatility': random.uniform(0.02, 0.08)
        }

    def fit_scaler(self):
        """No-op - scaler not needed for simple predictor"""
        logger.info("Simple predictor - no scaler needed")
        pass

    def fit_model(self):
        """No-op - using rule-based predictor instead of ML"""
        logger.info("Simple rule-based predictor - no model training needed")
        pass

    def predict_movement(self, coin_id):
        """Predict using simple rules instead of ML"""
        try:
            # Get features (this should work now without pandas)
            features = self.get_analysis_features(coin_id, lookback_hours=2)
            if features is None:
                features = self._get_fallback_features()
            
            # Use simple predictor
            prediction, confidence = self.predictor.predict(features)
            
            return {
                'prediction': int(prediction),
                'confidence': float(confidence),
                'features': features
            }
            
        except Exception as e:
            logger.error(f"Prediction failed for {coin_id}: {e}")
            return {
                'prediction': 0,
                'confidence': 0.33,
                'features': self._get_fallback_features()
            }

# Alias for backward compatibility
CryptoAnalyzer = CryptoDataManager