import requests
import json

# CryptoPanic API Key (Replace with your own key)
CRYPTO_PANIC_API_KEY = '716e8f410469cf6d77baaa48dd00821822e4ab97'

# CoinGecko API Endpoint
COINGECKO_URL = 'https://api.coingecko.com/api/v3/coins/markets'

# Coins to track (Match CoinGecko's naming format)
coin_ids = ['bitcoin', 'ethereum', 'solana', 'cardano', 'dogecoin']


# Fetch Crypto News from CryptoPanic
def fetch_crypto_news():
    url = f'https://cryptopanic.com/api/v1/posts/?auth_token={CRYPTO_PANIC_API_KEY}&currencies={",".join([c.upper() for c in coin_ids])}'
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        articles = data.get('results', [])
        return articles
    else:
        print(f"Failed to fetch news. Status code: {response.status_code}")
        return []


# Fetch Price Data from CoinGecko
def fetch_price_data():
    params = {
        'vs_currency': 'usd',
        'ids': ','.join(coin_ids),
        'order': 'market_cap_desc',
        'per_page': len(coin_ids),
        'page': 1,
        'sparkline': False
    }

    response = requests.get(COINGECKO_URL, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch price data. Status code: {response.status_code}")
        return []


# Analyze Trends and Detect Alerts
def analyze_trends(news_data, price_data):
    trending_coins = {}

    # Count mentions of each coin in the news
    for article in news_data:
        for coin in coin_ids:
            if coin in article['title'].lower():
                trending_coins[coin] = trending_coins.get(coin, 0) + 1

    print("\n=== Trending Coins Based on Social Mentions ===")
    for coin, mentions in trending_coins.items():
        print(f"{coin.capitalize()}: {mentions} mentions")

    # Correlate with price data and detect significant changes
    print("\n=== Price and Social Activity Analysis ===")
    for coin_data in price_data:
        coin = coin_data['id']
        mentions = trending_coins.get(coin, 0)
        price_change = coin_data['price_change_percentage_24h']

        print(f"{coin.capitalize()}: {mentions} mentions, {price_change:.2f}% price change")

        # Alert for significant price movement or high social activity
        if mentions > 3 or abs(price_change) > 5:
            print(
                f"🚨 Alert: {coin.capitalize()} is trending! Mentions: {mentions}, Price Change: {price_change:.2f}% 🚨")


# Main Execution
if __name__ == "__main__":
    news_data = fetch_crypto_news()
    price_data = fetch_price_data()

    if news_data and price_data:
        analyze_trends(news_data, price_data)
