import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { AlertTriangle, ArrowDown, ArrowRight, ArrowUp } from 'lucide-react';

interface AnalysisFeatures {
  price_volatility: number;
  volume_change: number;
  price_momentum: number;
  avg_sentiment: number;
  sentiment_volatility: number;
}

interface Analysis {
  prediction: number;
  confidence: number;
  features: AnalysisFeatures;
}

interface AnalysisCardProps {
  coinId: string;
  analysis: Analysis;
}

const AnalysisCard = ({ coinId, analysis }: AnalysisCardProps) => {
  if (!analysis) return null;

  const getPredictionDisplay = () => {
    switch (analysis.prediction) {
      case 1:
        return {
          text: 'Potential Pump',
          icon: <ArrowUp className="h-5 w-5 text-green-600" />,
          color: 'text-green-600'
        };
      case -1:
        return {
          text: 'Potential Dump',
          icon: <ArrowDown className="h-5 w-5 text-red-600" />,
          color: 'text-red-600'
        };
      default:
        return {
          text: 'Stable Movement',
          icon: <ArrowRight className="h-5 w-5 text-yellow-600" />,
          color: 'text-yellow-600'
        };
    }
  };

  const predictionDisplay = getPredictionDisplay();

  return (
    <Card className="shadow-lg hover:shadow-xl transition-shadow">
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Analysis</span>
          {predictionDisplay.icon}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div>
            <h3 className={`text-lg font-bold ${predictionDisplay.color}`}>
              {predictionDisplay.text}
            </h3>
            <p className="text-sm text-gray-500">
              Confidence: {(analysis.confidence * 100).toFixed(1)}%
            </p>
          </div>
          
          <div className="space-y-2">
            <h4 className="font-medium">Key Metrics:</h4>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div>
                <p className="text-gray-500">Price Volatility:</p>
                <p>{analysis.features.price_volatility.toFixed(2)}</p>
              </div>
              <div>
                <p className="text-gray-500">Volume Change:</p>
                <p>{(analysis.features.volume_change * 100).toFixed(2)}%</p>
              </div>
              <div>
                <p className="text-gray-500">Price Momentum:</p>
                <p>{analysis.features.price_momentum.toFixed(2)}</p>
              </div>
              <div>
                <p className="text-gray-500">News Sentiment:</p>
                <p>{analysis.features.avg_sentiment.toFixed(2)}</p>
              </div>
            </div>
          </div>

          {analysis.confidence < 0.6 && (
            <div className="flex items-center gap-2 text-yellow-600 text-sm">
              <AlertTriangle className="h-4 w-4" />
              <p>Low confidence prediction - use caution</p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default AnalysisCard;