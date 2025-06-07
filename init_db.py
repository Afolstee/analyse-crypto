import requests
import time
from crypto_analyzer import CryptoDataManager

# API Configuration
COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/markets"
CRYPTOCOMPARE_NEWS_URL = "https://min-api.cryptocompare.com/data/v2/news/"
API_KEY = "095f087a120bf715fc109915bd8c6f237656caeef95300672772ab9bb5fea890"

# List of coins to track
COIN_IDS = [
    "bitcoin",
    "trump",
    "ethereum",
    "solana",
    "cardano",
    "dogecoin",
    "ripple",
    "polkadot",
    "litecoin",
    "chainlink",
    "uniswap",
]

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


def fetch_price_data():
    """Fetch price data from CoinGecko"""
    params = {"vs_currency": "usd", "ids": ",".join(COIN_IDS)}
    try:
        response = requests.get(COINGECKO_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        print(f"✅ Fetched price data successfully for {len(data)} coins")
        return data
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to fetch price data: {e}")
        return []


def fetch_news():
    """Fetch news from CryptoCompare API"""
    # Get symbols for the coins we're tracking
    symbols = [COIN_SYMBOL_MAP.get(coin_id, coin_id.upper()) for coin_id in COIN_IDS]
    
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


def test_api_endpoints():
    """Test API endpoints before starting data collection"""
    print("🔍 Testing API endpoints...")
    
    # Test CoinGecko
    print("Testing CoinGecko API...")
    prices = fetch_price_data()
    if prices:
        print(f"✅ CoinGecko API working - got {len(prices)} coins")
    else:
        print("❌ CoinGecko API failed")
        return False
    
    # Test CryptoCompare
    print("Testing CryptoCompare API...")
    news = fetch_news()
    if news:
        print(f"✅ CryptoCompare API working - got {len(news)} articles")
        # Show sample article for verification
        if len(news) > 0:
            sample = news[0]
            print(f"📰 Sample article: {sample.get('title', 'No title')[:50]}...")
    else:
        print("⚠️ CryptoCompare API failed - continuing without news data")
    
    return True


def initialize_database(data_points=12):
    """
    Initialize the database with multiple data points
    Args:
        data_points (int): Number of data points to collect (each 5 minutes apart)
    """
    print(f"🚀 Starting database initialization with {data_points} data points...")

    # Test APIs first
    if not test_api_endpoints():
        print("❌ API tests failed. Please check your configuration.")
        return None

    # Initialize the CryptoDataManager
    manager = CryptoDataManager(coin_ids=COIN_IDS)
    print("✅ CryptoDataManager initialized")

    # Collect initial data points
    successful_collections = 0
    for i in range(data_points):
        print(f"\n📊 Collecting data point {i+1}/{data_points}")

        # Fetch and store price data
        prices = fetch_price_data()
        if prices:
            manager.store_price_data(prices)
            print(f"✅ Stored price data for {len(prices)} coins")
            successful_collections += 1

        # Fetch and store news data
        news = fetch_news()
        if news:
            manager.store_news_data(news)
            print(f"✅ Stored {len(news)} news articles")

        # Wait between data points (except for last iteration)
        if i < data_points - 1:
            wait_time = 300  # 5 minutes
            print(f"⏳ Waiting {wait_time//60} minutes before next data collection...")
            time.sleep(wait_time)

    if successful_collections < 3:
        print(f"⚠️ Only {successful_collections} successful data collections. May not be enough for model training.")

    # Initialize the model
    print("\n🔧 Fitting scaler...")
    manager.fit_scaler()

    if manager.is_scaler_fitted:
        print("🤖 Training model...")
        success = manager.fit_model()
        if success:
            print("✅ Model training successful!")
        else:
            print("⚠️ Model training incomplete - may need more data")
    else:
        print("⚠️ Scaler fitting failed - not enough data")

    return manager


def quick_test():
    """Quick test with minimal data for development"""
    print("🚀 Running quick test initialization...")
    
    # Test APIs
    if not test_api_endpoints():
        return None
    
    # Initialize manager and collect just a few data points
    manager = CryptoDataManager(coin_ids=COIN_IDS)
    
    for i in range(3):  # Just 3 data points for testing
        print(f"\n📊 Test data point {i+1}/3")
        
        prices = fetch_price_data()
        if prices:
            manager.store_price_data(prices)
        
        news = fetch_news()
        if news:
            manager.store_news_data(news)
            
        if i < 2:
            time.sleep(10)  # Short wait for testing
    
    # Try to fit
    manager.fit_scaler()
    if manager.is_scaler_fitted:
        manager.fit_model()
    
    return manager


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("🧪 Running in test mode...")
        manager = quick_test()
    else:
        print("🚀 Starting full database initialization process...")
        # Initialize database with 12 data points (1 hour of data)
        manager = initialize_database(data_points=12)

    if manager:
        print("\n✅ Database initialization complete!")
        print("You can now start your Flask backend server.")
        print(f"Model ready: {manager.is_scaler_fitted}")
    else:
        print("\n❌ Database initialization failed!")
        print("Please check your API configuration and try again.")