// First, update your API configuration in a new file: src/config/api.ts
export const API_CONFIG = {
    baseUrl: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000',
    endpoints: {
      stream: '/crypto-stream',
      data: '/crypto-data'
    }
  };
  