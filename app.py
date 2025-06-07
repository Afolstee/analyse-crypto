from flask import Flask, jsonify, Response
from flask_cors import CORS
import requests
import json
import time
import os
from threading import Thread
from crypto_analyzer import CryptoDataManager

app = Flask(__name__)

# Environment-based CORS configuration
allowed_origins = os.environ.get('ALLOWED_ORIGINS', 'http://localhost:3000').split(',')
CORS(app, origins=allowed_origins)

COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/markets"
CRYPTOCOMPARE_NEWS_URL = "https://min-api.cryptocompare.com/data/v2/news/"

# Use environment variable for API key
API_KEY = os.environ.get('API_KEY', '095f087a120bf715fc109915bd8c6f237656caeef95300672772ab9bb5fea890')

coin_ids = ["bitcoin", "trump", "ethereum", "solana", "cardano", "dogecoin", "ripple", "polkadot", "litecoin", "chainlink", "uniswap"]

# Mapping for CryptoCompare symbols
COIN_SYMBOL_MAP = {
    "bitcoin": "BTC",
    "trump": "TRUMP",
    "ethereum": "ETH", 
    "solana": "SOL",
    "cardano": "ADA",
    "dogecoin": "DOGE",
    "ripple": "XRP",
    "polkadot": "DOT",
    "litecoin": "LTC",
    "chainlink": "LINK",
    "uniswap": "UNI"
}

# Initialize the analyzer
analyzer = CryptoDataManager(coin_ids=coin_ids)

def fetch_price_data():
    """Enhanced price fetching with better error handling and debugging"""
    params = {
        "vs_currency": "usd", 
        "ids": ",".join(coin_ids),
        "order": "market_cap_desc",
        "per_page": 11,
        "page": 1,
        "sparkline": False,
        "price_change_percentage": "24h"
    }
    
    try:
        print(f"🔄 Fetching price data from: {COINGECKO_URL}")
        print(f"📋 Parameters: {params}")
        
        response = requests.get(COINGECKO_URL, params=params, timeout=15)
        print(f"📊 Response status: {response.status_code}")
        
        if response.status_code == 429:
            print("⚠️ Rate limited by CoinGecko - waiting 60 seconds")
            time.sleep(60)
            response = requests.get(COINGECKO_URL, params=params, timeout=15)
        
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ Fetched price data successfully - {len(data)} coins")
        
        # Debug: Print first coin data structure
        if data and len(data) > 0:
            print(f"🔍 Sample coin data keys: {list(data[0].keys())}")
            print(f"🔍 Bitcoin price: ${data[0].get('current_price', 'N/A')}")
        
        # Validate data structure
        valid_data = []
        for coin in data:
            if isinstance(coin, dict) and 'current_price' in coin and coin['current_price'] is not None:
                valid_data.append(coin)
            else:
                print(f"⚠️ Invalid coin data: {coin.get('id', 'unknown')}")
        
        print(f"✅ Valid price records: {len(valid_data)}")
        return valid_data
        
    except requests.exceptions.Timeout:
        print("⏱️ CoinGecko API timeout - trying fallback")
        return fetch_fallback_prices()
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to fetch price data: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response text: {e.response.text[:200]}")
        return fetch_fallback_prices()
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
        return fetch_fallback_prices()
    except Exception as e:
        print(f"❌ Unexpected error fetching prices: {e}")
        return fetch_fallback_prices()

def fetch_fallback_prices():
    """Fallback price data when API fails"""
    print("🔄 Using fallback price data")
    fallback_prices = {
        "bitcoin": 43000,
        "ethereum": 2500,
        "solana": 85,
        "cardano": 0.45,
        "dogecoin": 0.08,
        "ripple": 0.55,
        "polkadot": 7.2,
        "litecoin": 75,
        "chainlink": 15,
        "uniswap": 6.5,
        "trump": 12.5
    }
    
    fallback_data = []
    for coin_id in coin_ids:
        fallback_data.append({
            'id': coin_id,
            'symbol': COIN_SYMBOL_MAP.get(coin_id, coin_id.upper()),
            'name': coin_id.title(),
            'current_price': fallback_prices.get(coin_id, 1.0),
            'market_cap': fallback_prices.get(coin_id, 1.0) * 1000000,
            'total_volume': fallback_prices.get(coin_id, 1.0) * 100000,
            'price_change_percentage_24h': round((hash(coin_id) % 200 - 100) / 10, 2),
            'market_cap_rank': coin_ids.index(coin_id) + 1,
            'last_updated': time.strftime('%Y-%m-%dT%H:%M:%S.000Z')
        })
    
    print(f"✅ Generated {len(fallback_data)} fallback price records")
    return fallback_data

def fetch_news():
    """Fetch news from CryptoCompare API"""
    # Get symbols for the coins we're tracking
    symbols = [COIN_SYMBOL_MAP.get(coin_id, coin_id.upper()) for coin_id in coin_ids]
    
    params = {
        "api_key": API_KEY,
        "categories": ",".join(symbols),
        "excludeCategories": "Sponsored",
        "sortOrder": "latest",
        "limit": 50
    }
    
    try:
        print(f"🔄 Fetching news from: {CRYPTOCOMPARE_NEWS_URL}")
        response = requests.get(CRYPTOCOMPARE_NEWS_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        # CryptoCompare can return "Success" or success message in "Message"
        if data.get("Response") == "Success" or "successfully returned" in data.get("Message", "").lower():
            news_data = data.get("Data", [])
            print(f"✅ Fetched {len(news_data)} news articles successfully")
            
            # Transform CryptoCompare news format to match our expected format
            transformed_news = []
            for article in news_data:
                transformed_article = {
                    'id': article.get('id', ''),
                    'title': article.get('title', ''),
                    'url': article.get('url', ''),
                    'published_at': article.get('published_on', 0),
                    'source': {
                        'title': article.get('source_info', {}).get('name', 'Unknown'),
                        'domain': article.get('source', '')
                    },
                    'summary': article.get('body', '')[:200] + '...' if len(article.get('body', '')) > 200 else article.get('body', ''),
                    'currencies': []  # CryptoCompare doesn't provide this directly
                }
                
                # Try to extract relevant currencies from categories or tags
                categories = article.get('categories', '')
                if categories:
                    # Match categories with our tracked coins
                    for coin_id, symbol in COIN_SYMBOL_MAP.items():
                        if symbol.lower() in categories.lower() or coin_id in categories.lower():
                            transformed_article['currencies'].append({
                                'code': symbol,
                                'title': coin_id.title()
                            })
                
                transformed_news.append(transformed_article)
            
            return transformed_news
        else:
            print(f"❌ CryptoCompare API error: {data.get('Message', 'Unknown error')}")
            return []
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to fetch news data: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Status code: {e.response.status_code}")
            print(f"Response: {e.response.text[:300]}...")
        return []

def create_analysis_response(prices, news, analyzer):
    """Create response with price, news, and analysis data"""
    print(f"🔧 Creating response with {len(prices)} prices and {len(news)} news articles")
    
    response = {
        'prices': prices,
        'news': news,
        'analysis': {},
        'model_ready': getattr(analyzer, 'is_scaler_fitted', True),  # Default to True for simple predictor
        'debug': {
            'price_count': len(prices),
            'news_count': len(news),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
    }
    
    # Add analysis for each coin if we have a working analyzer
    if hasattr(analyzer, 'predict_movement'):
        for price in prices:
            try:
                analysis = analyzer.predict_movement(price['id'])
                if analysis:
                    response['analysis'][price['id']] = analysis
            except Exception as e:
                print(f"Error analyzing {price['id']}: {e}")
    
    print(f"✅ Response created successfully")
    return response

def initialize_analyzer():
    """Initialize the analyzer with any available data"""
    retry_count = 0
    max_retries = 3
    
    while retry_count < max_retries:
        try:
            print(f"Attempting to initialize analyzer (attempt {retry_count + 1}/{max_retries})...")
            prices = fetch_price_data()
            news = fetch_news()
            
            if prices:
                if hasattr(analyzer, 'store_price_data'):
                    analyzer.store_price_data(prices)
                print(f"Stored {len(prices)} price records")
            if news:
                if hasattr(analyzer, 'store_news_data'):
                    analyzer.store_news_data(news)
                print(f"Stored {len(news)} news records")
            
            # Try to fit with whatever data we have
            if hasattr(analyzer, 'fit_scaler'):
                analyzer.fit_scaler()
            if hasattr(analyzer, 'fit_model'):
                analyzer.fit_model()
            
            print("✅ Initial analyzer setup complete")
            break
                
        except Exception as e:
            print(f"⚠️ Initial training incomplete (will retry in 10 seconds): {e}")
        
        retry_count += 1
        if retry_count < max_retries:
            time.sleep(10)
    
    if retry_count >= max_retries:
        print("❌ Failed to initialize analyzer after maximum retries, continuing with basic functionality")

@app.route("/crypto-data", methods=["GET"])
def get_crypto_data():
    try:
        print("🔄 /crypto-data endpoint called")
        prices = fetch_price_data()
        news = fetch_news()
        
        print(f"📊 Fetched {len(prices)} prices and {len(news)} news articles")
        
        # Store data for analysis
        if prices and hasattr(analyzer, 'store_price_data'):
            analyzer.store_price_data(prices)
        if news and hasattr(analyzer, 'store_news_data'):
            analyzer.store_news_data(news)
        
        # Generate response with analysis
        response = create_analysis_response(prices, news, analyzer)
        
        print(f"✅ Returning response with {len(response['prices'])} prices")
        return jsonify(response)
    except Exception as e:
        print(f"❌ Error in get_crypto_data: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'prices': [], 'news': []}), 500

@app.route("/crypto-stream")
def crypto_stream():
    def generate():
        error_count = 0
        max_errors = 10
        
        while error_count < max_errors:
            try:
                print("🔄 Streaming data...")
                prices = fetch_price_data()
                news = fetch_news()
                
                # Store and analyze data
                if prices and hasattr(analyzer, 'store_price_data'):
                    analyzer.store_price_data(prices)
                if news and hasattr(analyzer, 'store_news_data'):
                    analyzer.store_news_data(news)
                
                # Generate response with analysis
                response = create_analysis_response(prices, news, analyzer)
                yield f"data: {json.dumps(response)}\n\n"
                
                error_count = 0  # Reset error count on success
                time.sleep(30)  # Increased interval to reduce API load
                
            except Exception as e:
                error_count += 1
                print(f"❌ Error in stream (attempt {error_count}): {e}")
                yield f"data: {json.dumps({'error': str(e), 'prices': [], 'news': []})}\n\n"
                time.sleep(10)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Transfer-Encoding": "chunked",
            "Access-Control-Allow-Origin": "*"
        }
    )

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_ready': getattr(analyzer, 'is_scaler_fitted', True),
        'supported_coins': coin_ids,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route("/debug-prices", methods=["GET"])
def debug_prices():
    """Debug endpoint to test price fetching"""
    try:
        print("🔍 Debug endpoint called")
        prices = fetch_price_data()
        return jsonify({
            'success': True,
            'count': len(prices),
            'prices': prices,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        print(f"❌ Debug error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500
    
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        'message': 'Crypto Analysis API is running',
        'status': 'active',
        'endpoints': {
            'crypto_data': '/crypto-data',
            'crypto_stream': '/crypto-stream', 
            'health_check': '/health',
            'debug_prices': '/debug-prices'
        },
        'model_ready': getattr(analyzer, 'is_scaler_fitted', True),
        'supported_coins': coin_ids,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    })

if __name__ == "__main__":
    # Get port and debug settings from environment
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print("🚀 Initializing analyzer...")
    # Start analyzer initialization in a separate thread
    init_thread = Thread(target=initialize_analyzer)
    init_thread.daemon = True
    init_thread.start()
    
    # Start the Flask app
    print(f"🌐 Starting Flask app on port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True)