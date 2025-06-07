import requests
import json

# Replace with your CryptoPanic API key
API_KEY = '716e8f410469cf6d77baaa48dd00821822e4ab97'

# Define the API endpoint
url = f'https://cryptopanic.com/api/v1/posts/?auth_token={API_KEY}&currencies=BTC,ETH'

# Send a GET request to the API
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    data = response.json()
    print(f"Fetched {len(data['results'])} news articles.")

    # Save the data to a JSON file
    with open('crypto_news.json', 'w') as file:
        json.dump(data, file, indent=4)  # `indent=4` makes the file human-readable
    print("News data saved to 'crypto_news.json'.")

    # Print the first 5 news articles
    for news in data['results'][:5]:
        print(f"Title: {news['title']}")
        print(f"URL: {news['url']}")
        print(f"Published: {news['published_at']}")
        print("-" * 50)
else:
    print(f"Failed to fetch news. Status code: {response.status_code}")