"use client"

import { useEffect, useState } from 'react';
import { useSSE } from '@/hooks/useSSE';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';
import { AlertCircle, Loader2, TrendingUp, TrendingDown } from 'lucide-react';
import { API_CONFIG } from '@/config/api';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { motion } from 'framer-motion';

interface CryptoPrice {
  id: string;
  symbol: string;
  current_price: number;
  price_change_percentage_24h: number;
  total_volume: number;
}

interface NewsItem {
  title: string;
  url: string;
  published_at: string;
  currencies: { code: string }[];
}

interface ChartData {
  name: string;
  price: number;
}

interface StreamData {
  prices: CryptoPrice[];
  news: NewsItem[];
  analysis: { [key: string]: Analysis };
  model_ready: boolean;
}

interface Analysis {
  prediction: number;
  confidence: number;
  features: {
    price_volatility: number;
    volume_change: number;
    price_momentum: number;
    avg_sentiment: number;
    sentiment_volatility: number;
  };
}

// Add this type to extend the useSSE hook's return type
interface SSEResponse<T> {
  data: T | null;
  isConnected: boolean;
  error: Error | null;
  isInitialLoading: boolean;
}

const AnalysisCard = ({ coinId, analysis, modelReady }: { 
  coinId: string; 
  analysis?: Analysis;
  modelReady: boolean;
}) => {
  if (!modelReady) {
    return (
      <Card className="shadow-lg">
        <CardContent className="p-4">
          <div className="flex items-center justify-center space-x-2 py-4">
            <Loader2 className="h-4 w-4 animate-spin" />
            <p className="text-sm text-gray-500">Training model with initial data...</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!analysis) return null;

  const isPredictedUp = analysis.prediction > 0;
  const confidencePercentage = (analysis.confidence * 100).toFixed(1);

  return (
    <Card className="shadow-lg">
      <CardContent className="p-4">
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold text-lg">Market Analysis</h3>
            {isPredictedUp ? 
              <TrendingUp className="h-5 w-5 text-green-600" /> : 
              <TrendingDown className="h-5 w-5 text-red-600" />
            }
          </div>
          
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Predicted Movement</span>
              <span className={`font-medium ${isPredictedUp ? 'text-green-600' : 'text-red-600'}`}>
                {analysis.prediction.toFixed(2)}%
              </span>
            </div>
            
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Confidence</span>
              <span className="font-medium">{confidencePercentage}%</span>
            </div>
          </div>

          <div className="space-y-2">
            <h4 className="text-sm font-medium text-gray-700">Key Metrics</h4>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div>
                <p className="text-gray-600">Price Volatility</p>
                <p className="font-medium">{analysis.features.price_volatility.toFixed(2)}</p>
              </div>
              <div>
                <p className="text-gray-600">Volume Change</p>
                <p className="font-medium">{analysis.features.volume_change.toFixed(2)}%</p>
              </div>
              <div>
                <p className="text-gray-600">Price Momentum</p>
                <p className="font-medium">{analysis.features.price_momentum.toFixed(2)}</p>
              </div>
              <div>
                <p className="text-gray-600">Avg Sentiment</p>
                <p className="font-medium">{analysis.features.avg_sentiment.toFixed(2)}</p>
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

const CryptoPriceCard = ({ coin, chartData, analysis, modelReady }: { 
  coin: CryptoPrice; 
  chartData: ChartData[]; 
  analysis?: Analysis;
  modelReady: boolean;
}) => {
  const isPositive = coin.price_change_percentage_24h >= 0;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="space-y-4">
        <Card className="overflow-hidden shadow-lg hover:shadow-xl transition-shadow">
          <CardHeader className={`${isPositive ? 'bg-green-50' : 'bg-red-50'}`}>
            <CardTitle className="capitalize flex justify-between items-center">
              <span>{coin.id} ({coin.symbol.toUpperCase()})</span>
              {isPositive ? 
                <TrendingUp className="h-5 w-5 text-green-600" /> : 
                <TrendingDown className="h-5 w-5 text-red-600" />
              }
            </CardTitle>
          </CardHeader>
          <CardContent className="p-4">
            <div className="mb-4">
              <p className="text-2xl font-bold">${coin.current_price.toLocaleString()}</p>
              <p className={`text-sm ${isPositive ? 'text-green-500' : 'text-red-500'}`}>
                {coin.price_change_percentage_24h.toFixed(2)}% (24h)
              </p>
              <p className="text-xs text-gray-500 mt-1">
                Volume: ${(coin.total_volume/1000000).toFixed(2)}M
              </p>
            </div>
            {chartData && (
              <div className="h-48">
                <LineChart width={300} height={180} data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                  <XAxis 
                    dataKey="name" 
                    tick={{ fontSize: 12 }}
                    stroke="#888888"
                  />
                  <YAxis 
                    domain={['auto', 'auto']}
                    tick={{ fontSize: 12 }}
                    stroke="#888888"
                  />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: 'white',
                      border: '1px solid #e5e7eb'
                    }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="price" 
                    stroke={isPositive ? '#10B981' : '#EF4444'}
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                  />
                </LineChart>
              </div>
            )}
          </CardContent>
        </Card>
        {analysis !== undefined && (
          <AnalysisCard 
            coinId={coin.id}
            analysis={analysis}
            modelReady={modelReady}
          />
        )}
      </div>
    </motion.div>
  );
};

const NewsCard = ({ item }: { item: NewsItem }) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.5 }}
  >
    <Card className="shadow-md hover:shadow-lg transition-shadow">
      <CardContent className="p-6">
        <h3 className="font-bold mb-2 text-lg">
          <a 
            href={item.url} 
            target="_blank" 
            rel="noopener noreferrer" 
            className="text-blue-600 hover:text-blue-800 hover:underline"
          >
            {item.title}
          </a>
        </h3>
        <p className="text-sm text-gray-500 mb-3">
          Published: {new Date(item.published_at).toLocaleString()}
        </p>
        <div className="flex flex-wrap gap-2">
          {item.currencies.map(currency => (
            <span 
              key={currency.code} 
              className="bg-gray-100 text-gray-800 rounded-full px-3 py-1 text-sm font-medium"
            >
              {currency.code}
            </span>
          ))}
        </div>
      </CardContent>
    </Card>
  </motion.div>
);

export default function Home() {
  // Update the useSSE hook usage with the correct type
  const { data, isConnected, error, isInitialLoading } = useSSE() as SSEResponse<StreamData>;
  const [prices, setPrices] = useState<CryptoPrice[]>([]);
  const [news, setNews] = useState<NewsItem[]>([]);
  const [chartData, setChartData] = useState<{ [key: string]: ChartData[] }>({});

  useEffect(() => {
    if (data) {
      if (data.prices?.length > 0) {
        setPrices(data.prices);
      }
      
      if (data.news?.length > 0) {
        setNews(data.news);
      }
      
      setChartData(prevChartData => {
        const newChartData = { ...prevChartData };
        data.prices?.forEach((price: CryptoPrice) => {
          if (!newChartData[price.id]) {
            newChartData[price.id] = [];
          }
          newChartData[price.id] = [
            ...newChartData[price.id],
            {
              name: new Date().toLocaleTimeString(),
              price: price.current_price
            }
          ].slice(-30);
        });
        return newChartData;
      });
    }
  }, [data]);

  if (isInitialLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <Loader2 className="h-8 w-8 animate-spin" />
      </div>
    );
  }

  return (
    <div className="container mx-auto p-4">
      <motion.h1 
        className="text-3xl font-bold mb-6"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        Crypto Analytics Dashboard
      </motion.h1>
      
      {!isConnected && (
        <Alert 
          variant="destructive" 
          className="mb-4"
        >
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            Connection lost. Displaying last available data. Attempting to reconnect...
          </AlertDescription>
        </Alert>
      )}

      <Tabs defaultValue="prices" className="mb-6">
        <TabsList className="w-full border-b">
          <TabsTrigger value="prices" className="px-4 py-2">Prices</TabsTrigger>
          <TabsTrigger value="news" className="px-4 py-2">News</TabsTrigger>
        </TabsList>

        <TabsContent value="prices" className="mt-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {prices.map((coin, index) => (
              <motion.div
                key={coin.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              >
                <CryptoPriceCard 
                  coin={coin} 
                  chartData={chartData[coin.id] || []}
                  analysis={data?.analysis?.[coin.id]}
                  modelReady={data?.model_ready ?? false}
                />
              </motion.div>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="news" className="mt-4">
          <div className="space-y-4">
            {news.map((item, index) => (
              <NewsCard key={index} item={item} />
            ))}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}