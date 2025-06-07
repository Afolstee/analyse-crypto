export const API_CONFIG = {
  BACKEND_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000',
  SOCKET_URL: process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:5000',
  COIN_IDS: ['bitcoin', 'ethereum', 'solana', 'cardano', 'dogecoin', 'ripple', 'polkadot', 'litecoin', 'chainlink', 'uniswap']
};