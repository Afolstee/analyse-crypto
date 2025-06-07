from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from textblob import TextBlob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import json


class CryptoDataManager:
    def __init__(self, coin_ids, db_path='crypto_data.db'):
        self.coin_ids = coin_ids
        self.db_path = db_path
        self.setup_database()
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_scaler_fitted = False

    def setup_database(self):
        conn = sqlite3.connect(self.db_path)
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
        
        conn.commit()
        conn.close()

    def store_price_data(self, prices):
        conn = sqlite3.connect(self.db_path)
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

    def store_news_data(self, news):
        conn = sqlite3.connect(self.db_path)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for article in news:
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
        
        conn.commit()
        conn.close()

    def get_analysis_features(self, coin_id, lookback_hours=24, end_time=None):
        """Get analysis features up to a specific end time
        
        Args:
            coin_id (str): ID of the cryptocurrency
            lookback_hours (int): Number of hours to look back for analysis
            end_time (datetime|str|None): End time for analysis, defaults to current time
        """
        conn = sqlite3.connect(self.db_path)
        
        if end_time is None:
            end_time = datetime.now()
            
        if isinstance(end_time, str):
            end_time = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
                
        cutoff_time = end_time - timedelta(hours=lookback_hours)
        
        end_time_str = end_time.strftime('%Y-%m-%d %H:%M:%S')
        cutoff_time_str = cutoff_time.strftime('%Y-%m-%d %H:%M:%S')
        
        price_df = pd.read_sql_query('''
            SELECT * FROM price_history 
            WHERE coin_id = ? AND timestamp > ? AND timestamp <= ?
            ORDER BY timestamp
        ''', conn, params=(coin_id, cutoff_time_str, end_time_str))
        
        news_df = pd.read_sql_query('''
            SELECT timestamp, sentiment 
            FROM news_history 
            WHERE timestamp > ? AND timestamp <= ? AND currencies LIKE ?
            ORDER BY timestamp
        ''', conn, params=(cutoff_time_str, end_time_str, f'%{coin_id.upper()}%'))
        
        conn.close()
        
        if len(price_df) < 2:
            return None
                
        price_df['timestamp'] = pd.to_datetime(price_df['timestamp'], format='%Y-%m-%d %H:%M:%S')
        news_df['timestamp'] = pd.to_datetime(news_df['timestamp'])
                
        features = {
            'price_volatility': price_df['price'].std() if len(price_df) > 0 else 0,
            'volume_change': price_df['volume'].pct_change().mean() if len(price_df) > 0 else 0,
            'price_momentum': price_df['price'].diff().mean() if len(price_df) > 0 else 0,
            'avg_sentiment': news_df['sentiment'].mean() if len(news_df) > 0 else 0,
            'sentiment_volatility': news_df['sentiment'].std() if len(news_df) > 0 else 0
        }
        
        return features

    def fit_scaler(self):
        """Fit the scaler with historical data"""
        conn = sqlite3.connect(self.db_path)
        
        price_df = pd.read_sql_query('''
            SELECT * FROM price_history
        ''', conn)
        
        news_df = pd.read_sql_query('''
            SELECT * FROM news_history
        ''', conn)
        
        conn.close()
        
        if len(price_df) == 0 or len(news_df) == 0:
            print("⚠️ Not enough data to fit the scaler. Please collect more data.")
            return
        
        feature_data = []
        for coin_id in self.coin_ids:
            features = self.get_analysis_features(coin_id, lookback_hours=24 * 30)
            if features:
                feature_data.append([
                    features['price_volatility'],
                    features['volume_change'],
                    features['price_momentum'],
                    features['avg_sentiment'],
                    features['sentiment_volatility']
                ])
        
        if len(feature_data) == 0:
            print("⚠️ No feature data available to fit the scaler.")
            return
        
        self.scaler.fit(feature_data)
        self.is_scaler_fitted = True
        print("✅ Scaler fitted successfully.")

    def fit_model(self):
        """Fit the RandomForestClassifier with historical data"""
        conn = sqlite3.connect(self.db_path)
        
        historical_data = []
        labels = []
        
        for coin_id in self.coin_ids:
            price_df = pd.read_sql_query('''
                SELECT * FROM price_history 
                WHERE coin_id = ?
                ORDER BY timestamp
            ''', conn, params=(coin_id,))
            
            if len(price_df) < 2:
                continue
                
            price_df['return'] = price_df['price'].pct_change()
            
            price_df['label'] = np.where(price_df['return'] > 0.01, 1,
                                       np.where(price_df['return'] < -0.01, -1, 0))
            
            for i in range(len(price_df) - 1):
                features = self.get_analysis_features(coin_id, 
                                                   lookback_hours=24,
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
        
        conn.close()
        
        if len(historical_data) == 0:
            print("⚠️ No historical data available to train the model.")
            return False
        
        X = np.array(historical_data)
        y = np.array(labels)
        
        self.model.fit(X, y)
        print(f"✅ Model trained successfully with {len(X)} samples.")
        return True

    def predict_movement(self, coin_id):
        """Predict price movement for a given coin"""
        features = self.get_analysis_features(coin_id)
        if features is None:
            return None
            
        feature_array = np.array([[
            features['price_volatility'],
            features['volume_change'],
            features['price_momentum'],
            features['avg_sentiment'],
            features['sentiment_volatility']
        ]])
        
        if not self.is_scaler_fitted:
            print("⚠️ Scaler not fitted. Fitting scaler with available data...")
            self.fit_scaler()
            if not self.is_scaler_fitted:
                return None
        
        scaled_features = self.scaler.transform(feature_array)
        
        try:
            prediction = self.model.predict(scaled_features)[0]
            probabilities = self.model.predict_proba(scaled_features)[0]
        except Exception as e:
            print("⚠️ Model not trained. Training model with historical data...")
            if not self.fit_model():
                return None
            prediction = self.model.predict(scaled_features)[0]
            probabilities = self.model.predict_proba(scaled_features)[0]
        
        return {
            'prediction': int(prediction),  # -1: dump, 0: stable, 1: pump
            'confidence': float(max(probabilities)),
            'features': features
        }