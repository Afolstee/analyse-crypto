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
        # Add connection timeout and retry settings
        self.db_timeout = 10.0  # 10 seconds timeout
        self.max_retries = 3

    def get_db_connection(self):
        """Get database connection with timeout and retry logic"""
        for attempt in range(self.max_retries):
            try:
                conn = sqlite3.connect(self.db_path, timeout=self.db_timeout)
                conn.execute('PRAGMA journal_mode=WAL')  # Better concurrency
                conn.execute('PRAGMA synchronous=NORMAL')  # Better performance
                return conn
            except sqlite3.OperationalError as e:
                logger.warning(f"Database connection attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(0.5)  # Wait before retry
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
        """Get analysis features up to a specific end time with proper error handling
        
        Args:
            coin_id (str): ID of the cryptocurrency
            lookback_hours (int): Number of hours to look back for analysis
            end_time (datetime|str|None): End time for analysis, defaults to current time
        """
        try:
            conn = self.get_db_connection()
            
            if end_time is None:
                end_time = datetime.now()
                
            if isinstance(end_time, str):
                end_time = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
                    
            cutoff_time = end_time - timedelta(hours=lookback_hours)
            
            end_time_str = end_time.strftime('%Y-%m-%d %H:%M:%S')
            cutoff_time_str = cutoff_time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Execute price query with timeout handling
            try:
                price_df = pd.read_sql_query('''
                    SELECT * FROM price_history 
                    WHERE coin_id = ? AND timestamp > ? AND timestamp <= ?
                    ORDER BY timestamp
                    LIMIT 1000
                ''', conn, params=(coin_id, cutoff_time_str, end_time_str))
            except Exception as e:
                logger.error(f"Failed to fetch price data for {coin_id}: {e}")
                conn.close()
                return None
            
            # Execute news query with timeout handling
            try:
                news_df = pd.read_sql_query('''
                    SELECT timestamp, sentiment 
                    FROM news_history 
                    WHERE timestamp > ? AND timestamp <= ? AND currencies LIKE ?
                    ORDER BY timestamp
                    LIMIT 500
                ''', conn, params=(cutoff_time_str, end_time_str, f'%{coin_id.upper()}%'))
            except Exception as e:
                logger.error(f"Failed to fetch news data for {coin_id}: {e}")
                # Continue with empty news data if price data is available
                news_df = pd.DataFrame(columns=['timestamp', 'sentiment'])
            
            conn.close()
            
            if len(price_df) < 2:
                logger.warning(f"Insufficient price data for {coin_id}: {len(price_df)} records")
                return None
                    
            # Process data with error handling and memory optimization
            try:
                # Convert timestamps more efficiently
                if len(price_df) > 0:
                    price_df['timestamp'] = pd.to_datetime(price_df['timestamp'], 
                                                         format='%Y-%m-%d %H:%M:%S', 
                                                         errors='coerce')
                
                if len(news_df) > 0:
                    # Limit news data to prevent memory issues
                    if len(news_df) > 100:  # Limit to 100 most recent news items
                        news_df = news_df.tail(100)
                    
                    news_df['timestamp'] = pd.to_datetime(news_df['timestamp'], 
                                                        format='%Y-%m-%d %H:%M:%S', 
                                                        errors='coerce')
                    # Drop rows with invalid timestamps
                    news_df = news_df.dropna(subset=['timestamp'])
                
            except Exception as e:
                logger.error(f"Failed to parse timestamps: {e}")
                return None
                    
            # Calculate features with safe operations and bounds checking
            try:
                # Ensure we have valid data before calculations
                price_values = price_df['price'].dropna()
                volume_values = price_df['volume'].dropna()
                
                features = {
                    'price_volatility': float(price_values.std()) if len(price_values) > 1 else 0.0,
                    'volume_change': float(volume_values.pct_change().mean()) if len(volume_values) > 1 else 0.0,
                    'price_momentum': float(price_values.diff().mean()) if len(price_values) > 1 else 0.0,
                    'avg_sentiment': 0.0,
                    'sentiment_volatility': 0.0
                }
                
                # Only calculate sentiment if we have valid news data
                if len(news_df) > 0:
                    sentiment_values = news_df['sentiment'].dropna()
                    if len(sentiment_values) > 0:
                        features['avg_sentiment'] = float(sentiment_values.mean())
                        features['sentiment_volatility'] = float(sentiment_values.std()) if len(sentiment_values) > 1 else 0.0
                
                # Replace any remaining NaN or infinite values
                for key, value in features.items():
                    if pd.isna(value) or np.isnan(value) or np.isinf(value):
                        features[key] = 0.0
                
                logger.info(f"Generated features for {coin_id}: {features}")
                return features
                
            except Exception as e:
                logger.error(f"Failed to calculate features: {e}")
                return None
            
        except Exception as e:
            logger.error(f"Error in get_analysis_features for {coin_id}: {e}")
            return None

    def fit_scaler(self):
        """Fit the scaler with historical data"""
        try:
            conn = self.get_db_connection()
            
            price_df = pd.read_sql_query('''
                SELECT * FROM price_history
                ORDER BY timestamp DESC
                LIMIT 10000
            ''', conn)
            
            news_df = pd.read_sql_query('''
                SELECT * FROM news_history
                ORDER BY timestamp DESC
                LIMIT 5000
            ''', conn)
            
            conn.close()
            
            if len(price_df) == 0 or len(news_df) == 0:
                logger.warning("Not enough data to fit the scaler. Please collect more data.")
                return
            
            feature_data = []
            for coin_id in self.coin_ids:
                try:
                    features = self.get_analysis_features(coin_id, lookback_hours=24 * 7)  # Reduced from 30 days
                    if features:
                        feature_data.append([
                            features['price_volatility'],
                            features['volume_change'],
                            features['price_momentum'],
                            features['avg_sentiment'],
                            features['sentiment_volatility']
                        ])
                except Exception as e:
                    logger.warning(f"Failed to get features for {coin_id}: {e}")
                    continue
            
            if len(feature_data) == 0:
                logger.warning("No feature data available to fit the scaler.")
                return
            
            self.scaler.fit(feature_data)
            self.is_scaler_fitted = True
            logger.info("✅ Scaler fitted successfully.")
            
        except Exception as e:
            logger.error(f"Failed to fit scaler: {e}")
            raise

    def fit_model(self):
        """Fit the RandomForestClassifier with historical data - optimized for memory usage"""
        try:
            conn = self.get_db_connection()
            
            historical_data = []
            labels = []
            max_samples_per_coin = 20  # Drastically reduced to prevent timeout
            
            for coin_id in self.coin_ids:
                try:
                    # Get only recent data to reduce memory usage
                    price_df = pd.read_sql_query('''
                        SELECT * FROM price_history 
                        WHERE coin_id = ?
                        ORDER BY timestamp DESC
                        LIMIT 100
                    ''', conn, params=(coin_id,))
                    
                    if len(price_df) < 3:  # Need at least 3 records
                        logger.warning(f"Insufficient data for {coin_id}: {len(price_df)} records")
                        continue
                        
                    price_df['return'] = price_df['price'].pct_change()
                    price_df['label'] = np.where(price_df['return'] > 0.01, 1,
                                               np.where(price_df['return'] < -0.01, -1, 0))
                    
                    # Process only a small sample to prevent timeout
                    sample_size = min(max_samples_per_coin, len(price_df) - 1)
                    processed_count = 0
                    
                    for i in range(0, sample_size, 2):  # Skip every other sample
                        if processed_count >= 10:  # Hard limit per coin
                            break
                            
                        try:
                            # Use shorter lookback to speed up processing
                            features = self.get_analysis_features(coin_id, 
                                                               lookback_hours=6,  # Very short lookback
                                                               end_time=price_df.iloc[i]['timestamp'])
                            if features:
                                historical_data.append([
                                    features['price_volatility'],
                                    features['volume_change'],
                                    features['price_momentum'],
                                    features['avg_sentiment'],
                                    features['sentiment_volatility']
                                ])
                                labels.append(price_df.iloc[i+1]['label'])
                                processed_count += 1
                        except Exception as e:
                            logger.warning(f"Failed to process sample {i} for {coin_id}: {e}")
                            continue
                            
                    logger.info(f"Processed {processed_count} samples for {coin_id}")
                            
                except Exception as e:
                    logger.warning(f"Failed to process {coin_id}: {e}")
                    continue
            
            conn.close()
            
            if len(historical_data) < 5:  # Need minimum samples
                logger.warning(f"Insufficient training data: {len(historical_data)} samples")
                return False
            
            X = np.array(historical_data)
            y = np.array(labels)
            
            # Use a simpler model configuration for faster training
            self.model = RandomForestClassifier(
                n_estimators=50,  # Reduced from 100
                max_depth=10,     # Limit depth
                random_state=42,
                n_jobs=1          # Single thread to control memory
            )
            
            self.model.fit(X, y)
            logger.info(f"✅ Model trained successfully with {len(X)} samples.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to fit model: {e}")
            return False

    def predict_movement(self, coin_id):
        """Predict price movement for a given coin"""
        try:
            features = self.get_analysis_features(coin_id)
            if features is None:
                logger.warning(f"No features available for {coin_id}")
                return None
                
            feature_array = np.array([[
                features['price_volatility'],
                features['volume_change'],
                features['price_momentum'],
                features['avg_sentiment'],
                features['sentiment_volatility']
            ]])
            
            if not self.is_scaler_fitted:
                logger.info("Scaler not fitted. Fitting scaler with available data...")
                self.fit_scaler()
                if not self.is_scaler_fitted:
                    logger.error("Failed to fit scaler")
                    return None
            
            scaled_features = self.scaler.transform(feature_array)
            
            try:
                prediction = self.model.predict(scaled_features)[0]
                probabilities = self.model.predict_proba(scaled_features)[0]
            except Exception as e:
                logger.info("Model not trained. Training model with historical data...")
                if not self.fit_model():
                    logger.error("Failed to train model")
                    return None
                prediction = self.model.predict(scaled_features)[0]
                probabilities = self.model.predict_proba(scaled_features)[0]
            
            return {
                'prediction': int(prediction),  # -1: dump, 0: stable, 1: pump
                'confidence': float(max(probabilities)),
                'features': features
            }
            
        except Exception as e:
            logger.error(f"Failed to predict movement for {coin_id}: {e}")
            return None