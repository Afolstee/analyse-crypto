import { useState, useEffect } from 'react';
import { API_CONFIG } from '@/config/api';

interface SSEData {
  prices: {
    id: string;
    symbol: string;
    current_price: number;
    price_change_percentage_24h: number;
    total_volume: number;
  }[];
  news: {
    title: string;
    url: string;
    published_at: string;
    currencies: { code: string }[];
  }[];
  analysis: {
    [key: string]: {
      prediction: number;
      confidence: number;
      features: {
        price_volatility: number;
        volume_change: number;
        price_momentum: number;
        avg_sentiment: number;
        sentiment_volatility: number;
      };
    };
  };
}
export function useSSE() {
  const [isInitialLoading, setIsInitialLoading] = useState(true);
  const [lastUpdateTime, setLastUpdateTime] = useState<Date | null>(null);
  const [data, setData] = useState<SSEData | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    const eventSource = new EventSource(
      `${API_CONFIG.baseUrl}${API_CONFIG.endpoints.stream}`
    );

    eventSource.onopen = () => {
      setIsConnected(true);
      setError(null);
    };

    eventSource.onmessage = (event) => {
      try {
        const parsedData = JSON.parse(event.data);
        setData(parsedData);
        setLastUpdateTime(new Date());
        if (isInitialLoading) setIsInitialLoading(false);
      } catch (err) {
        setError(err instanceof Error ? err : new Error('Failed to parse SSE data'));
      }
    };

    eventSource.onerror = () => {
      setIsConnected(false);
      setError(new Error('SSE connection failed'));
    };

    return () => {
      eventSource.close();
      setIsConnected(false);
    };
  }, [isInitialLoading]); // Added isInitialLoading to dependencies array

  return { data, isConnected, error, isInitialLoading, lastUpdateTime };
}