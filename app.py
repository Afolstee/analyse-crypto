from flask import Flask, jsonify, Response, request
from flask_cors import CORS
from flask_compress import Compress
import requests
import json
import time
import os
import redis
from threading import Thread, Lock
from crypto_analyzer import CryptoDataManager
from datetime import datetime, timedelta

app = Flask(__name__)

# Enable gzip compression for better performance
Compress(app)

# Environment-based CORS configuration
allowed_origins = os.environ.get('ALLOWED_ORIGINS', 'http://localhost:3000,https://analyse-crypto-nine.vercel.app').split(',')

# Also add flexible CORS for development and production
CORS(app, 
     origins=allowed_origins,
     allow_headers=['Content-Type', 'Authorization', 'Accept'],
     methods=['GET', 'POST', 'OPTIONS'],
     supports_credentials=True)

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

# Enhanced caching system with Redis support
CACHE_DURATION_SECONDS = 300  # 5 minutes for API responses
NEWS_CACHE_DURATION_HOURS = 12  # 12 hours for news data
NEWS_REFRESH_INTERVAL = 12 * 60 * 60  # 12 hours in seconds

# Simple in-memory cache as fallback
memory_cache = {}

# Try Redis if available, fallback to memory
try:
    redis_client = redis.from_url(os.environ.get('REDIS_URL', 'redis://localhost:6379'))
    redis_client.ping()
    USE_REDIS = True
    print("✅ Redis connected successfully")
except Exception as e:
    USE_REDIS = False
    redis_client = None
    print(f"⚠️ Redis not available, using memory cache: {e}")

# News caching system (enhanced)
news_cache = {
    'data': [],
    'last_updated': None,
    'lock': Lock()
}

def get_cache(key):
    """Get data from cache (Redis or memory)"""
    try:
        if USE_REDIS:
            data = redis_client.get(key)
            if data:
                return json.loads(data)
            return None
        else:
            cache_entry = memory_cache.get(key, {})
            if cache_entry and time.time() - cache_entry.get('timestamp', 0) < CACHE_DURATION_SECONDS:
                return cache_entry.get('data')
            return None
    except Exception as e:
        print(f"❌ Cache get error for key {key}: {e}")
        return None

def set_cache(key, data, duration=CACHE_DURATION_SECONDS):
    """Set data in cache (Redis or memory)"""
    try:
        if USE_REDIS:
            redis_client.setex(key, duration, json.dumps(data, default=str))
        else:
            memory_cache[key] = {'data': data, 'timestamp': time.time()}
        print(f"✅ Cached data for key: {key}")
    except Exception as e:
        print(f"❌ Cache set error for key {key}: {e}")

def is_cache_expired():
    """Check if news cache is expired (older than 12 hours)"""
    if news_cache['last_updated'] is None:
        return True
    
    time_diff = datetime.now() - news_cache['last_updated']
    return time_diff.total_seconds() > (NEWS_CACHE_DURATION_HOURS * 3600)

def fetch_price_data():
    """Enhanced price fetching with caching and better error handling"""
    cache_key = "crypto_prices"
    
    # Try cache first
    cached_prices = get_cache(cache_key)
    if cached_prices:
        print(f"✅ Using cached price data ({len(cached_prices)} coins)")
        return cached_prices
    
    print("❌ Cache miss for prices, fetching fresh data...")
    
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
        
        response = requests.get(COINGECKO_URL, params=params, timeout=15)
        print(f"📊 Response status: {response.status_code}")
        
        if response.status_code == 429:
            print("⚠️ Rate limited by CoinGecko - waiting 60 seconds")
            time.sleep(60)
            response = requests.get(COINGECKO_URL, params=params, timeout=15)
        
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ Fetched price data successfully - {len(data)} coins")
        
        # Validate data structure
        valid_data = []
        for coin in data:
            if isinstance(coin, dict) and 'current_price' in coin and coin['current_price'] is not None:
                valid_data.append(coin)
            else:
                print(f"⚠️ Invalid coin data: {coin.get('id', 'unknown')}")
        
        print(f"✅ Valid price records: {len(valid_data)}")
        
        # Cache the valid data
        if valid_data:
            set_cache(cache_key, valid_data, CACHE_DURATION_SECONDS)
        
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

def fetch_news_from_api():
    """Fetch fresh news from CryptoCompare API with caching"""
    cache_key = "crypto_news_raw"
    
    # Check if we have recent news in cache
    cached_news = get_cache(cache_key)
    if cached_news:
        print(f"✅ Using cached news data ({len(cached_news)} articles)")
        return cached_news
    
    print("❌ Cache miss for news, fetching fresh data...")
    
    # Get symbols for the coins we're tracking
    symbols = [COIN_SYMBOL_MAP.get(coin_id, coin_id.upper()) for coin_id in coin_ids]
    
    params = {
        "api_key": API_KEY,
        "categories": ",".join(symbols),
        "excludeCategories": "Sponsored",
        "sortOrder": "latest",
        "limit": 100  # Fetch more to ensure we have 50+ recent articles
    }
    
    try:
        print(f"🔄 Fetching fresh news from: {CRYPTOCOMPARE_NEWS_URL}")
        response = requests.get(CRYPTOCOMPARE_NEWS_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        # CryptoCompare can return "Success" or success message in "Message"
        if data.get("Response") == "Success" or "successfully returned" in data.get("Message", "").lower():
            news_data = data.get("Data", [])
            print(f"✅ Fetched {len(news_data)} news articles successfully")
            
            # Transform CryptoCompare news format to match our expected format
            transformed_news = []
            current_time = int(time.time())
            twenty_four_hours_ago = current_time - (24 * 60 * 60)
            
            for article in news_data:
                published_time = article.get('published_on', 0)
                
                # Only include articles from last 24 hours for freshness
                if published_time < twenty_four_hours_ago:
                    continue
                    
                transformed_article = {
                    'id': article.get('id', ''),
                    'title': article.get('title', ''),
                    'url': article.get('url', ''),
                    'published_at': published_time,
                    'source': {
                        'title': article.get('source_info', {}).get('name', 'Unknown'),
                        'domain': article.get('source', '')
                    },
                    'summary': article.get('body', '')[:200] + '...' if len(article.get('body', '')) > 200 else article.get('body', ''),
                    'currencies': []
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
            
            # Sort by published time (most recent first)
            transformed_news.sort(key=lambda x: x['published_at'], reverse=True)
            
            # Keep only top 50 most recent articles
            final_news = transformed_news[:50]
            
            # Cache the processed news
            if final_news:
                set_cache(cache_key, final_news, NEWS_CACHE_DURATION_HOURS * 3600)
            
            return final_news
        else:
            print(f"❌ CryptoCompare API error: {data.get('Message', 'Unknown error')}")
            return []
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to fetch news data: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Status code: {e.response.status_code}")
            print(f"Response: {e.response.text[:300]}...")
        return []

def get_cached_news():
    """Get news from cache or fetch fresh if expired"""
    with news_cache['lock']:
        if is_cache_expired():
            print("🔄 News cache expired, fetching fresh news...")
            fresh_news = fetch_news_from_api()
            if fresh_news:
                news_cache['data'] = fresh_news
                news_cache['last_updated'] = datetime.now()
                print(f"✅ Updated news cache with {len(fresh_news)} articles")
            else:
                print("⚠️ Failed to fetch fresh news, using existing cache")
        else:
            print(f"✅ Using cached news ({len(news_cache['data'])} articles)")
        
        return news_cache['data'].copy()

def paginate_news(news_list, page=1, per_page=10):
    """Paginate news list with caching support"""
    # Limit per_page to prevent abuse
    per_page = min(per_page, 50)
    
    total_items = len(news_list)
    total_pages = (total_items + per_page - 1) // per_page  # Ceiling division
    
    if page < 1:
        page = 1
    elif page > total_pages:
        page = total_pages if total_pages > 0 else 1
    
    start_index = (page - 1) * per_page
    end_index = start_index + per_page
    
    paginated_news = news_list[start_index:end_index]
    
    return {
        'news': paginated_news,
        'pagination': {
            'current_page': page,
            'per_page': per_page,
            'total_items': total_items,
            'total_pages': total_pages,
            'has_next': page < total_pages,
            'has_prev': page > 1,
            'next_page': page + 1 if page < total_pages else None,
            'prev_page': page - 1 if page > 1 else None
        }
    }

def create_analysis_response(prices, news, analyzer):
    """Create response with price, news, and analysis data"""
    print(f"🔧 Creating response with {len(prices)} prices and {len(news)} news articles")
    
    response = {
        'prices': prices,
        'news': news,
        'analysis': {},
        'model_ready': getattr(analyzer, 'is_scaler_fitted', True),
        'news_cache_info': {
            'last_updated': news_cache['last_updated'].isoformat() if news_cache['last_updated'] else None,
            'is_fresh': not is_cache_expired(),
            'next_refresh': (news_cache['last_updated'] + timedelta(hours=12)).isoformat() if news_cache['last_updated'] else None
        },
        'cache_info': {
            'type': 'redis' if USE_REDIS else 'memory',
            'status': 'active'
        },
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

def news_refresh_worker():
    """Background worker to refresh news every 12 hours"""
    while True:
        try:
            time.sleep(NEWS_REFRESH_INTERVAL)  # Wait 12 hours
            print("🔄 Background news refresh triggered")
            get_cached_news()  # This will refresh if needed
        except Exception as e:
            print(f"❌ Error in news refresh worker: {e}")
            time.sleep(300)  # Wait 5 minutes before retrying

def initialize_analyzer():
    """Initialize the analyzer with any available data"""
    retry_count = 0
    max_retries = 3
    
    while retry_count < max_retries:
        try:
            print(f"Attempting to initialize analyzer (attempt {retry_count + 1}/{max_retries})...")
            prices = fetch_price_data()
            news = get_cached_news()  # Use cached news system
            
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
        
        # Get pagination parameters
        page = int(request.args.get('page', 1))
        per_page = min(int(request.args.get('per_page', 10)), 50)  # Limit to 50
        
        # Create cache key for this specific request
        cache_key = f"crypto_data:page:{page}:per_page:{per_page}"
        
        # Try cache first
        cached_data = get_cache(cache_key)
        if cached_data:
            print(f"✅ Cache hit for crypto-data: {cache_key}")
            return jsonify(cached_data)
        
        print(f"❌ Cache miss for crypto-data: {cache_key}")
        
        # Fetch fresh data
        prices = fetch_price_data()
        
        # Get cached news and paginate
        all_news = get_cached_news()
        paginated_result = paginate_news(all_news, page, per_page)
        
        print(f"📊 Fetched {len(prices)} prices and {len(paginated_result['news'])} paginated news articles")
        
        # Store data for analysis
        if prices and hasattr(analyzer, 'store_price_data'):
            analyzer.store_price_data(prices)
        if all_news and hasattr(analyzer, 'store_news_data'):
            analyzer.store_news_data(all_news)
        
        # Generate response with analysis
        response = create_analysis_response(prices, paginated_result['news'], analyzer)
        response['pagination'] = paginated_result['pagination']
        
        # Cache the response
        set_cache(cache_key, response, CACHE_DURATION_SECONDS)
        
        print(f"✅ Returning response with {len(response['prices'])} prices and pagination info")
        return jsonify(response)
    except Exception as e:
        print(f"❌ Error in get_crypto_data: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'prices': [], 'news': [], 'pagination': None}), 500

@app.route("/news", methods=["GET"])
def get_news_only():
    """Enhanced news endpoint with caching"""
    try:
        print("🔄 /news endpoint called")
        
        # Get pagination parameters
        page = int(request.args.get('page', 1))
        per_page = min(int(request.args.get('per_page', 10)), 50)  # Limit to 50
        
        # Create cache key for this specific request
        cache_key = f"news_only:page:{page}:per_page:{per_page}"
        
        # Try cache first
        cached_data = get_cache(cache_key)
        if cached_data:
            print(f"✅ Cache hit for news: {cache_key}")
            return jsonify(cached_data)
        
        print(f"❌ Cache miss for news: {cache_key}")
        
        # Get cached news and paginate
        all_news = get_cached_news()
        paginated_result = paginate_news(all_news, page, per_page)
        
        response = {
            'news': paginated_result['news'],
            'pagination': paginated_result['pagination'],
            'cache_info': {
                'last_updated': news_cache['last_updated'].isoformat() if news_cache['last_updated'] else None,
                'is_fresh': not is_cache_expired(),
                'next_refresh': (news_cache['last_updated'] + timedelta(hours=12)).isoformat() if news_cache['last_updated'] else None,
                'type': 'redis' if USE_REDIS else 'memory'
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Cache the response
        set_cache(cache_key, response, CACHE_DURATION_SECONDS)
        
        print(f"✅ Returning {len(paginated_result['news'])} news articles (page {page})")
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Error in get_news_only: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'news': [], 'pagination': None}), 500

@app.route("/crypto-stream")
def crypto_stream():
    def generate():
        error_count = 0
        max_errors = 10
        
        while error_count < max_errors:
            try:
                print("🔄 Streaming data...")
                prices = fetch_price_data()
                
                # For streaming, get first page of news
                all_news = get_cached_news()
                paginated_result = paginate_news(all_news, 1, 10)
                
                # Store and analyze data
                if prices and hasattr(analyzer, 'store_price_data'):
                    analyzer.store_price_data(prices)
                if all_news and hasattr(analyzer, 'store_news_data'):
                    analyzer.store_news_data(all_news)
                
                # Generate response with analysis
                response = create_analysis_response(prices, paginated_result['news'], analyzer)
                response['pagination'] = paginated_result['pagination']
                yield f"data: {json.dumps(response, default=str)}\n\n"
                
                error_count = 0  # Reset error count on success
                time.sleep(30)  # Increased interval to reduce API load
                
            except Exception as e:
                error_count += 1
                print(f"❌ Error in stream (attempt {error_count}): {e}")
                yield f"data: {json.dumps({'error': str(e), 'prices': [], 'news': [], 'pagination': None})}\n\n"
                time.sleep(10)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Transfer-Encoding": "chunked",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization, Accept",
            "Connection": "keep-alive"
        }
    )

@app.route("/health", methods=["GET"])
def health_check():
    """Enhanced health check with cache status"""
    cache_status = "healthy"
    if USE_REDIS:
        try:
            redis_client.ping()
        except:
            cache_status = "redis_error"
    
    return jsonify({
        'status': 'healthy',
        'model_ready': getattr(analyzer, 'is_scaler_fitted', True),
        'supported_coins': coin_ids,
        'cache_info': {
            'type': 'redis' if USE_REDIS else 'memory',
            'status': cache_status,
            'news_articles': len(news_cache['data']),
            'last_updated': news_cache['last_updated'].isoformat() if news_cache['last_updated'] else None,
            'is_fresh': not is_cache_expired()
        },
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
            'cache_type': 'redis' if USE_REDIS else 'memory',
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

@app.route("/refresh-news", methods=["POST"])
def force_refresh_news():
    """Manual endpoint to force news refresh"""
    try:
        print("🔄 Manual news refresh triggered")
        with news_cache['lock']:
            fresh_news = fetch_news_from_api()
            if fresh_news:
                news_cache['data'] = fresh_news
                news_cache['last_updated'] = datetime.now()
                
                # Clear related cache entries
                if USE_REDIS:
                    # Clear all news-related cache entries
                    for key in redis_client.scan_iter(match="news*"):
                        redis_client.delete(key)
                    for key in redis_client.scan_iter(match="crypto_data*"):
                        redis_client.delete(key)
                else:
                    # Clear memory cache entries that contain news
                    keys_to_delete = [k for k in memory_cache.keys() if 'news' in k or 'crypto_data' in k]
                    for key in keys_to_delete:
                        del memory_cache[key]
                
                return jsonify({
                    'success': True,
                    'message': f'News refreshed successfully with {len(fresh_news)} articles',
                    'articles_count': len(fresh_news),
                    'cache_cleared': True,
                    'timestamp': datetime.now().isoformat()
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'Failed to fetch fresh news'
                }), 500
    except Exception as e:
        print(f"❌ Error in force refresh: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route("/clear-cache", methods=["POST"])
def clear_cache():
    """Manual endpoint to clear all cache"""
    try:
        if USE_REDIS:
            redis_client.flushdb()
            message = "Redis cache cleared"
        else:
            memory_cache.clear()
            message = "Memory cache cleared"
        
        return jsonify({
            'success': True,
            'message': message,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        'message': 'Enhanced Crypto Analysis API is running',
        'status': 'active',
        'performance': {
            'caching_enabled': True,
            'cache_type': 'redis' if USE_REDIS else 'memory',
            'compression_enabled': True
        },
        'endpoints': {
            'crypto_data': '/crypto-data?page=1&per_page=10',
            'news_only': '/news?page=1&per_page=10',
            'crypto_stream': '/crypto-stream', 
            'health_check': '/health',
            'debug_prices': '/debug-prices',
            'refresh_news': '/refresh-news (POST)',
            'clear_cache': '/clear-cache (POST)'
        },
        'model_ready': getattr(analyzer, 'is_scaler_fitted', True),
        'supported_coins': coin_ids,
        'cache_info': {
            'type': 'redis' if USE_REDIS else 'memory',
            'news_articles': len(news_cache['data']),
            'last_updated': news_cache['last_updated'].isoformat() if news_cache['last_updated'] else None,
            'refresh_interval': f'{NEWS_CACHE_DURATION_HOURS} hours'
        },
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    })

@app.after_request
def after_request(response):
    """Add performance and security headers"""
    # Add cache headers for better performance
    if request.endpoint in ['get_news_only', 'get_crypto_data']:
        response.headers['Cache-Control'] = f'public, max-age={CACHE_DURATION_SECONDS}'
    
    # Security headers
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    
    return response

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': [
            '/crypto-data', '/news', '/crypto-stream', 
            '/health', '/debug-prices', '/refresh-news', '/clear-cache'
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500

@app.errorhandler(429)
def rate_limit_error(error):
    return jsonify({
        'error': 'Rate limit exceeded',
        'message': 'Too many requests, please try again later'
    }), 429

if __name__ == "__main__":
    print("🚀 Starting Enhanced Crypto Analysis API...")
    
    # Initialize the analyzer in the background
    init_thread = Thread(target=initialize_analyzer, daemon=True)
    init_thread.start()
    
    # Start background news refresh worker
    news_worker = Thread(target=news_refresh_worker, daemon=True)
    news_worker.start()
    
    # Get port from environment or default to 5000
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") == "development"
    
    print(f"✅ Server configuration:")
    print(f"   - Port: {port}")
    print(f"   - Debug mode: {debug}")
    print(f"   - Cache type: {'Redis' if USE_REDIS else 'Memory'}")
    print(f"   - CORS origins: {allowed_origins}")
    print(f"   - Supported coins: {len(coin_ids)} coins")
    print(f"   - News refresh interval: {NEWS_CACHE_DURATION_HOURS} hours")
    
    # Run the Flask app
    app.run(
        host="0.0.0.0",
        port=port,
        debug=debug,
        threaded=True,
        use_reloader=False  # Disable reloader to prevent duplicate background threads
    )