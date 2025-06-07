from flask import Flask, jsonify, Response
from flask_cors import CORS
import requests
import json
import time
from threading import Thread
from crypto_analyzer import CryptoDataManager

app = Flask(__name__)
CORS(app)

COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/markets"
CRYPTOCOMPARE_NEWS_URL = "https://min-api.cryptocompare.com/data/v2/news/"
API_KEY = "095f087a120bf715fc109915bd8c6f237656caeef95300672772ab9bb5fea890"

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
    params = {"vs_currency": "usd", "ids": ",".join(coin_ids)}
    try:
        response = requests.get(COINGECKO_URL, params=params, timeout=10)
        response.raise_for_status()
        print("✅ Fetched price data successfully")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to fetch price data: {e}")
        return []

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
    response = {
        'prices': prices,
        'news': news,
        'analysis': {},
        'model_ready': analyzer.is_scaler_fitted
    }
    
    if analyzer.is_scaler_fitted:
        # Add analysis for each coin
        for price in prices:
            try:
                analysis = analyzer.predict_movement(price['id'])
                if analysis:
                    response['analysis'][price['id']] = analysis
            except Exception as e:
                print(f"Error analyzing {price['id']}: {e}")
    
    return response

def initialize_analyzer():
    """Initialize the analyzer with any available data"""
    retry_count = 0
    max_retries = 5
    
    while not analyzer.is_scaler_fitted and retry_count < max_retries:
        try:
            print(f"Attempting to initialize analyzer (attempt {retry_count + 1}/{max_retries})...")
            prices = fetch_price_data()
            news = fetch_news()
            
            if prices:
                analyzer.store_price_data(prices)
                print(f"Stored {len(prices)} price records")
            if news:
                analyzer.store_news_data(news)
                print(f"Stored {len(news)} news records")
            
            # Try to fit with whatever data we have
            analyzer.fit_scaler()
            if analyzer.is_scaler_fitted:
                analyzer.fit_model()
                print("✅ Initial model training complete")
                break
            else:
                print("⚠️ Not enough data to fit scaler, retrying...")
                
        except Exception as e:
            print(f"⚠️ Initial training incomplete (will retry in 10 seconds): {e}")
        
        retry_count += 1
        time.sleep(10)
    
    if not analyzer.is_scaler_fitted:
        print("❌ Failed to initialize analyzer after maximum retries")

@app.route("/crypto-data", methods=["GET"])
def get_crypto_data():
    try:
        prices = fetch_price_data()
        news = fetch_news()
        
        # Store data for analysis
        if prices:
            analyzer.store_price_data(prices)
        if news:
            analyzer.store_news_data(news)
        
        # Generate response with analysis
        response = create_analysis_response(prices, news, analyzer)
        return jsonify(response)
    except Exception as e:
        print(f"Error in get_crypto_data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route("/crypto-stream")
def crypto_stream():
    def generate():
        error_count = 0
        max_errors = 10
        
        while error_count < max_errors:
            try:
                prices = fetch_price_data()
                news = fetch_news()
                
                # Store and analyze data
                if prices:
                    analyzer.store_price_data(prices)
                if news:
                    analyzer.store_news_data(news)
                
                # If model isn't fitted yet, try to fit it
                if not analyzer.is_scaler_fitted:
                    try:
                        analyzer.fit_scaler()
                        if analyzer.is_scaler_fitted:
                            analyzer.fit_model()
                            print("✅ Model training complete during stream")
                    except Exception as e:
                        print(f"⚠️ Model training incomplete: {e}")
                
                # Generate response with analysis
                response = create_analysis_response(prices, news, analyzer)
                yield f"data: {json.dumps(response)}\n\n"
                
                error_count = 0  # Reset error count on success
                time.sleep(30)  # Increased interval to reduce API load
                
            except Exception as e:
                error_count += 1
                print(f"Error in stream (attempt {error_count}): {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
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
        'model_ready': analyzer.is_scaler_fitted,
        'supported_coins': coin_ids
    })

if __name__ == "__main__":
    print("Initializing analyzer...")
    # Start analyzer initialization in a separate thread
    init_thread = Thread(target=initialize_analyzer)
    init_thread.daemon = True
    init_thread.start()
    
    # Start the Flask app
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)