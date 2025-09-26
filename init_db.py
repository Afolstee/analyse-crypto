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
        return response.json()
    except requests.exceptions.RequestException:
        return []


def fetch_news():
    """Fetch news from CryptoCompare API"""
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
        
        if data.get("Response") == "Success" or "successfully returned" in data.get("Message", "").lower():
            news_data = data.get("Data", [])
            
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
                    'currencies': []
                }
                
                categories = article.get('categories', '')
                if categories:
                    for coin_id, symbol in COIN_SYMBOL_MAP.items():
                        if symbol.lower() in categories.lower() or coin_id in categories.lower():
                            transformed_article['currencies'].append({
                                'code': symbol,
                                'title': coin_id.title()
                            })
                
                transformed_news.append(transformed_article)
            
            return transformed_news
        else:
            return []
            
    except requests.exceptions.RequestException:
        return []


def test_api_endpoints():
    """Test API endpoints before starting data collection"""
    prices = fetch_price_data()
    if not prices:
        return False
    
    fetch_news()  # Test news endpoint but don't require success
    return True


def initialize_database(data_points=12):
    """
    Initialize the database with multiple data points
    Args:
        data_points (int): Number of data points to collect (each 5 minutes apart)
    """
    if not test_api_endpoints():
        return None

    manager = CryptoDataManager(coin_ids=COIN_IDS)
    successful_collections = 0
    
    for i in range(data_points):
        prices = fetch_price_data()
        if prices:
            manager.store_price_data(prices)
            successful_collections += 1

        news = fetch_news()
        if news:
            manager.store_news_data(news)

        if i < data_points - 1:
            time.sleep(300)  # 5 minutes

    if successful_collections < 3:
        return None

    manager.fit_scaler()
    if manager.is_scaler_fitted:
        manager.fit_model()

    return manager


def quick_test():
    """Quick test with minimal data for development"""
    if not test_api_endpoints():
        return None
    
    manager = CryptoDataManager(coin_ids=COIN_IDS)
    
    for i in range(3):
        prices = fetch_price_data()
        if prices:
            manager.store_price_data(prices)
        
        news = fetch_news()
        if news:
            manager.store_news_data(news)
            
        if i < 2:
            time.sleep(10)
    
    manager.fit_scaler()
    if manager.is_scaler_fitted:
        manager.fit_model()
    
    return manager


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        manager = quick_test()
    else:
        manager = initialize_database(data_points=12)

    if not manager:
        exit(1)