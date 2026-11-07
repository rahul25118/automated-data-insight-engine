export interface CsvColumnSummary {
  name: string;
  type: 'numeric' | 'string' | 'date';
  nullCount: number;
  uniqueCount: number;
  min?: number;
  max?: number;
  mean?: number;
  median?: number;
  mode?: number[];
  stdDev?: number;
  sampleStdDev?: number;
  populationStdDev?: number;
  sampleVariance?: number;
  populationVariance?: number;
  q1?: number;
  q3?: number;
  iqr?: number;
  percentiles?: Record<number, number>;
  coeffOfVariation?: number;
  skewness?: number;
  kurtosis?: number;
  sum?: number;
  sampleValues: (string | number)[];
  timeSeriesAnalysis?: TimeSeriesAnalysisResult;
}

export interface StatisticalInferenceResult {
  testName: string;
  statistic: number;
  pValue: number;
  degreesOfFreedom?: number;
  alpha: number;
  rejectNull: boolean;
  assumptions: string[];
  interpretation: string;
  limitations: string[];
}

export interface TimeSeriesAnalysisResult {
  isTemporal: boolean;
  trendSlope?: number;
  intercept?: number;
  rSquared?: number;
  mae?: number;
  rmse?: number;
  mape?: number;
  forecastNext3?: number[];
  walkForwardValidationR2?: number;
  interpretation?: string;
}

export interface CsvAnalysisResult {
  fileName: string;
  rowCount: number;
  columnCount: number;
  columns: CsvColumnSummary[];
  anomalies: {
    column: string;
    rowIndices: number[];
    reason: string;
    value: any;
    method?: 'IQR' | 'Z_SCORE' | 'MODIFIED_Z_SCORE';
  }[];
  groupAggregations?: Record<string, Record<string, number>>;
  correlations?: {
    var1: string;
    var2: string;
    pearsonR: number;
    pValue: number;
    significant: boolean;
  }[];
  hypothesisTests?: StatisticalInferenceResult[];
  generatedSummaryMarkdown: string;
  parsedAt: number;
}
