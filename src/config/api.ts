export const API_CONFIG = {
  baseUrl: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000',
  endpoints: {
    stream: '/crypto-stream',
    data: '/crypto-data',
    news: '/news',           // Add this
    refreshNews: '/refresh-news'  // Add this
  }
};