"use client"

import { useEffect, useState } from 'react';
import { useSSE } from '@/hooks/useSSE';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';
import { AlertCircle, Loader2, TrendingUp, TrendingDown, Activity, X, HelpCircle } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { motion } from 'framer-motion';
import { API_CONFIG } from '@/config/api';

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
  timestamp?: number;
}

interface SSEResponse<T> {
  data: T | null;
  isConnected: boolean;
  error: Error | null;
  isInitialLoading: boolean;
}

// Custom tooltip component for explanations
const CustomTooltip = ({ text, children }: { text: string; children: React.ReactNode }) => {
  const [isVisible, setIsVisible] = useState(false);
  
  return (
    <div className="relative inline-block">
      <div
        onMouseEnter={() => setIsVisible(true)}
        onMouseLeave={() => setIsVisible(false)}
        onClick={() => setIsVisible(!isVisible)}
      >
        {children}
      </div>
      {isVisible && (
        <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-4 py-3 bg-gray-800 text-white text-xs rounded-lg shadow-lg z-50 max-w-xs min-w-[200px]">
          <div className="text-center leading-relaxed">{text}</div>
          <div className="absolute top-full left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-gray-800"></div>
        </div>
      )}
    </div>
  );
};

// Safe date formatter to prevent hydration mismatches
const formatDate = (date: Date | string | number) => {
  try {
    const d = new Date(date);
    return d.toLocaleString();
  } catch {
    return 'Invalid Date';
  }
};

// Safe time formatter
const formatTime = (date: Date | string | number) => {
  try {
    const d = new Date(date);
    return d.toLocaleTimeString();
  } catch {
    return 'Invalid Time';
  }
};

// Prediction Modal Component
const PredictionModal = ({ 
  isOpen, 
  onClose, 
  coin, 
  analysis, 
  chartData,
  modelReady 
}: { 
  isOpen: boolean;
  onClose: () => void;
  coin: CryptoPrice;
  analysis?: Analysis;
  chartData: ChartData[];
  modelReady: boolean;
}) => {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  if (!isOpen || !mounted) return null;

  if (!modelReady || !analysis) {
    return (
      <div 
        className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
        onClick={handleBackdropClick}
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.9 }}
          className="bg-white rounded-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="p-6">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-2xl font-bold capitalize">
                {coin.id} Price Prediction
              </h2>
              <button
                onClick={onClose}
                className="p-2 hover:bg-gray-100 rounded-full transition-colors"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            <div className="text-center py-8">
              <Loader2 className="h-12 w-12 animate-spin mx-auto mb-4 text-blue-600" />
              <h3 className="text-xl font-semibold mb-2">Preparing Analysis</h3>
              <p className="text-gray-600">
                {!modelReady 
                  ? 'Training AI model with market data...' 
                  : `Generating prediction for ${coin.id}...`
                }
              </p>
              <div className="mt-6 bg-blue-50 p-4 rounded-lg">
                <p className="text-sm text-blue-800">
                  Our AI is analyzing market trends, sentiment data, and price patterns to generate accurate predictions.
                </p>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    );
  }

  const isPredictedUp = analysis.prediction > 0;
  const lastUpdated = analysis.timestamp ? new Date(analysis.timestamp) : new Date();

  return (
    <div 
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
      onClick={handleBackdropClick}
    >
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.9 }}
        className="bg-white rounded-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="p-6">
          <div className="flex justify-between items-center mb-4">
            <div>
              <h2 className="text-2xl font-bold capitalize">
                {coin.id} Price Prediction
              </h2>
              <p className="text-sm text-gray-500 mt-1">
                (Last updated: {formatDate(lastUpdated)})
              </p>
            </div>
            <button
              onClick={onClose}
              className="p-2 hover:bg-gray-100 rounded-full transition-colors"
            >
              <X className="h-5 w-5" />
            </button>
          </div>

          <div className="space-y-6">
            {/* Current Price Info */}
            <Card>
              <CardContent className="p-4">
                <div className="flex justify-between items-center">
                  <div>
                    <p className="text-sm text-gray-600">Current Price</p>
                    <p className="text-2xl font-bold">${coin.current_price.toLocaleString()}</p>
                  </div>
                  <div className="text-right">
                    <p className="text-sm text-gray-600">24h Change</p>
                    <p className={`text-lg font-semibold ${coin.price_change_percentage_24h >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                      {coin.price_change_percentage_24h.toFixed(2)}%
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Prediction Details */}
            <Card className={`border-2 ${isPredictedUp ? 'border-green-200 bg-green-50' : 'border-red-200 bg-red-50'}`}>
              <CardContent className="p-4">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold">AI Prediction</h3>
                  {isPredictedUp ? 
                    <TrendingUp className="h-6 w-6 text-green-600" /> : 
                    <TrendingDown className="h-6 w-6 text-red-600" />
                  }
                </div>
                
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div>
                    <p className="text-sm text-gray-600">Predicted Movement</p>
                    <p className={`text-xl font-bold ${isPredictedUp ? 'text-green-600' : 'text-red-600'}`}>
                      {analysis.prediction.toFixed(2)}%
                    </p>
                  </div>
                  <div>
                    <div className="flex items-center space-x-1">
                      <p className="text-sm text-gray-600">Confidence Level</p>
                      <CustomTooltip text="How certain our AI model is about this prediction">
                        <HelpCircle className="h-3 w-3 text-gray-400 cursor-help" />
                      </CustomTooltip>
                    </div>
                    <p className="text-xl font-bold">{(analysis.confidence * 100).toFixed(1)}%</p>
                  </div>
                </div>

                <div className="space-y-3">
                  <h4 className="font-medium text-gray-700">Analysis Factors</h4>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-white p-3 rounded-lg">
                      <div className="flex items-center space-x-1 mb-1">
                        <p className="text-xs text-gray-600">Price Volatility</p>
                        <CustomTooltip text="Measures how much the price fluctuates over time">
                          <HelpCircle className="h-3 w-3 text-gray-400 cursor-help" />
                        </CustomTooltip>
                      </div>
                      <p className="font-semibold">{analysis.features.price_volatility.toFixed(3)}</p>
                    </div>
                    <div className="bg-white p-3 rounded-lg">
                      <div className="flex items-center space-x-1 mb-1">
                        <p className="text-xs text-gray-600">Volume Change</p>
                        <CustomTooltip text="Change in trading activity compared to previous periods">
                          <HelpCircle className="h-3 w-3 text-gray-400 cursor-help" />
                        </CustomTooltip>
                      </div>
                      <p className="font-semibold">{analysis.features.volume_change.toFixed(2)}%</p>
                    </div>
                    <div className="bg-white p-3 rounded-lg">
                      <div className="flex items-center space-x-1 mb-1">
                        <p className="text-xs text-gray-600">Price Momentum</p>
                        <CustomTooltip text="The speed and direction of recent price movements">
                          <HelpCircle className="h-3 w-3 text-gray-400 cursor-help" />
                        </CustomTooltip>
                      </div>
                      <p className="font-semibold">{analysis.features.price_momentum.toFixed(3)}</p>
                    </div>
                    <div className="bg-white p-3 rounded-lg">
                      <div className="flex items-center space-x-1 mb-1">
                        <p className="text-xs text-gray-600">Market Sentiment</p>
                        <CustomTooltip text="Overall market feeling based on news and social media analysis">
                          <HelpCircle className="h-3 w-3 text-gray-400 cursor-help" />
                        </CustomTooltip>
                      </div>
                      <p className="font-semibold">{analysis.features.avg_sentiment.toFixed(3)}</p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Price Chart */}
            {chartData.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle>Recent Price Movement</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-64">
                    <LineChart width={500} height={240} data={chartData}>
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
                        stroke={isPredictedUp ? '#10B981' : '#EF4444'}
                        strokeWidth={3}
                        dot={false}
                        isAnimationActive={false}
                      />
                    </LineChart>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </motion.div>
    </div>
  );
};

const CryptoPriceCard = ({ coin, chartData, analysis, modelReady }: { 
  coin: CryptoPrice; 
  chartData: ChartData[]; 
  analysis?: Analysis;
  modelReady: boolean;
}) => {
  const [showPrediction, setShowPrediction] = useState(false);
  const [mounted, setMounted] = useState(false);
  const isPositive = coin.price_change_percentage_24h >= 0;

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return (
      <Card className="overflow-hidden shadow-lg">
        <CardHeader className="bg-gray-50">
          <CardTitle className="capitalize flex justify-between items-center">
            <span>Loading...</span>
            <Loader2 className="h-5 w-5 animate-spin" />
          </CardTitle>
        </CardHeader>
        <CardContent className="p-4">
          <div className="animate-pulse">
            <div className="h-8 bg-gray-200 rounded mb-2"></div>
            <div className="h-4 bg-gray-200 rounded mb-4"></div>
            <div className="h-48 bg-gray-200 rounded"></div>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
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
            
            <div className="mb-4">
              <button
                onClick={() => setShowPrediction(true)}
                className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-lg transition-colors flex items-center justify-center space-x-2"
              >
                <Activity className="h-4 w-4" />
                <span>View AI Prediction</span>
              </button>
            </div>

            {chartData && chartData.length > 0 && (
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
      </motion.div>

      <PredictionModal
        isOpen={showPrediction}
        onClose={() => setShowPrediction(false)}
        coin={coin}
        analysis={analysis}
        chartData={chartData}
        modelReady={modelReady}
      />
    </>
  );
};

const NewsCard = ({ item }: { item: NewsItem }) => {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return (
      <Card className="shadow-md">
        <CardContent className="p-6">
          <div className="animate-pulse">
            <div className="h-6 bg-gray-200 rounded mb-2"></div>
            <div className="h-4 bg-gray-200 rounded mb-3"></div>
            <div className="flex space-x-2">
              <div className="h-6 bg-gray-200 rounded-full w-16"></div>
              <div className="h-6 bg-gray-200 rounded-full w-16"></div>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
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
            Published: {formatDate(item.published_at)}
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
};

export default function Home() {
  const { data, isConnected, error, isInitialLoading } = useSSE() as SSEResponse<StreamData>;
  const [prices, setPrices] = useState<CryptoPrice[]>([]);
  const [news, setNews] = useState<NewsItem[]>([]);
  const [chartData, setChartData] = useState<{ [key: string]: ChartData[] }>({});
  const [lastAnalysis, setLastAnalysis] = useState<{ [key: string]: Analysis }>({});
  const [activeTab, setActiveTab] = useState<string>("prices");
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (data) {
      if (data.prices?.length > 0) {
        setPrices(data.prices);
      }
      
      if (data.news?.length > 0) {
        setNews(data.news);
      }

      if (data.analysis) {
        const timestampedAnalysis: { [key: string]: Analysis } = {};
        Object.keys(data.analysis).forEach(coinId => {
          timestampedAnalysis[coinId] = {
            ...data.analysis[coinId],
            timestamp: Date.now()
          };
        });
        setLastAnalysis(prev => ({ ...prev, ...timestampedAnalysis }));
      }
      
      setChartData(prevChartData => {
        const newChartData = { ...prevChartData };
        data.prices?.forEach((price: CryptoPrice) => {
          if (!newChartData[price.id]) {
            newChartData[price.id] = [];
          }
          // Use a stable timestamp to prevent hydration issues
          const timestamp = formatTime(new Date());
          newChartData[price.id] = [
            ...newChartData[price.id],
            {
              name: timestamp,
              price: price.current_price
            }
          ].slice(-30); // Keep only last 30 data points
        });
        return newChartData;
      });
    }
  }, [data]);

  if (!mounted || isInitialLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4" />
          <p>Loading crypto dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-white">
      <div className="sticky top-0 z-40 bg-white border-b shadow-sm">
        <div className="container mx-auto p-4">
          <motion.h1 
            className="text-3xl font-bold mb-4"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            Crypto Analytics Dashboard
          </motion.h1>
          
          {!isConnected && (
            <Alert variant="destructive" className="mb-4">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                Connection lost. Displaying last available data. Attempting to reconnect...
              </AlertDescription>
            </Alert>
          )}

          <div className="flex justify-center">
            <div className="border-b bg-gray-100 rounded-lg">
              <button
                onClick={() => setActiveTab("prices")}
                className={`px-6 py-3 font-medium transition-colors rounded-l-lg ${
                  activeTab === "prices" 
                    ? "bg-green-500 text-white" 
                    : "text-gray-600 hover:text-gray-800 hover:bg-gray-200"
                }`}
              >
                Prices
              </button>
              <button
                onClick={() => setActiveTab("news")}
                className={`px-6 py-3 font-medium transition-colors rounded-r-lg ${
                  activeTab === "news" 
                    ? "bg-red-500 text-white" 
                    : "text-gray-600 hover:text-gray-800 hover:bg-gray-200"
                }`}
              >
                News
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="container mx-auto p-4">
        {activeTab === "prices" && (
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
                  analysis={lastAnalysis[coin.id] || data?.analysis?.[coin.id]}
                  modelReady={data?.model_ready ?? false}
                />
              </motion.div>
            ))}
          </div>
        )}

        {activeTab === "news" && (
          <div className="space-y-4">
            {news.map((item, index) => (
              <NewsCard key={`${item.url}-${index}`} item={item} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}