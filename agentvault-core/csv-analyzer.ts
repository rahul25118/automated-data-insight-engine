import { CsvAnalysisResult, CsvColumnSummary, StatisticalInferenceResult, TimeSeriesAnalysisResult } from './types';

export function parseCsv(text: string): { headers: string[]; rows: string[][] } {
  if (!text || text.trim().length === 0) {
    throw new Error('CSV file is empty or contains no parseable content');
  }

  const lines: string[] = [];
  let currentLine = '';
  let inQuotes = false;

  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    const nextChar = text[i + 1];

    if (char === '"') {
      currentLine += '"';
      if (inQuotes && nextChar === '"') {
        currentLine += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
    } else if ((char === '\r' || char === '\n') && !inQuotes) {
      if (char === '\r' && nextChar === '\n') {
        i++;
      }
      if (currentLine.trim().length > 0) {
        lines.push(currentLine);
      }
      currentLine = '';
    } else {
      currentLine += char;
    }
  }

  if (currentLine.trim().length > 0) {
    lines.push(currentLine);
  }

  if (lines.length === 0) {
    throw new Error('CSV file is empty');
  }

  const parseRow = (lineStr: string): string[] => {
    const fields: string[] = [];
    let field = '';
    let insideQuote = false;

    for (let j = 0; j < lineStr.length; j++) {
      const c = lineStr[j];
      const nc = lineStr[j + 1];

      if (c === '"') {
        if (insideQuote && nc === '"') {
          field += '"';
          j++;
        } else {
          insideQuote = !insideQuote;
        }
      } else if (c === ',' && !insideQuote) {
        fields.push(field.trim());
        field = '';
      } else {
        field += c;
      }
    }
    fields.push(field.trim());
    return fields;
  };

  const headers = parseRow(lines[0]);
  const rows: string[][] = [];

  for (let k = 1; k < lines.length; k++) {
    const parsed = parseRow(lines[k]);
    if (parsed.some(val => val.length > 0)) {
      rows.push(parsed);
    }
  }

  return { headers, rows };
}

function mathErfc(x: number): number {
  if (x < 0) return 2 - mathErfc(-x);
  const t = 1.0 / (1.0 + 0.3275911 * x);
  const poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
  return poly * Math.exp(-x * x);
}

export function studentTPValue(t: number, df: number): number {
  if (df <= 0) return 1.0;
  const absT = Math.abs(t);
  const a = df - 0.5;
  const b = 48 * a * a;
  const z2 = a * Math.log(1 + (absT * absT) / df);
  const z = Math.sqrt(z2);
  const normZ = z * (1 - (z2 + 3) / b);
  const pOneTail = 0.5 * mathErfc(normZ / Math.SQRT2);
  const pTwoTail = 2 * pOneTail;
  return Math.min(1.0, Math.max(0.00001, pTwoTail));
}

export function calculateLinearRegression(x: number[], y: number[]): {
  slope: number;
  intercept: number;
  rSquared: number;
  mae: number;
  rmse: number;
  mape: number;
} {
  const n = x.length;
  if (n < 2) return { slope: 0, intercept: 0, rSquared: 0, mae: 0, rmse: 0, mape: 0 };

  const xMean = x.reduce((a, b) => a + b, 0) / n;
  const yMean = y.reduce((a, b) => a + b, 0) / n;

  let ssXX = 0;
  let ssYY = 0;
  let ssXY = 0;

  for (let i = 0; i < n; i++) {
    const xDiff = x[i] - xMean;
    const yDiff = y[i] - yMean;
    ssXX += xDiff * xDiff;
    ssYY += yDiff * yDiff;
    ssXY += xDiff * yDiff;
  }

  const slope = ssXX !== 0 ? ssXY / ssXX : 0;
  const intercept = yMean - slope * xMean;
  const rSquared = (ssXX !== 0 && ssYY !== 0) ? (ssXY * ssXY) / (ssXX * ssYY) : 0;

  let sumAbsErr = 0;
  let sumSqErr = 0;
  let sumPctErr = 0;
  let validMapeCount = 0;

  for (let i = 0; i < n; i++) {
    const yPred = slope * x[i] + intercept;
    const err = y[i] - yPred;
    sumAbsErr += Math.abs(err);
    sumSqErr += err * err;
    if (Math.abs(y[i]) > 1e-9) {
      sumPctErr += Math.abs(err / y[i]);
      validMapeCount++;
    }
  }

  return {
    slope,
    intercept,
    rSquared,
    mae: sumAbsErr / n,
    rmse: Math.sqrt(sumSqErr / n),
    mape: validMapeCount > 0 ? (sumPctErr / validMapeCount) * 100 : 0
  };
}

export function analyzeCsvData(fileName: string, text: string): CsvAnalysisResult {
  const { headers, rows } = parseCsv(text);
  const rowCount = rows.length;
  const columnCount = headers.length;

  if (rowCount === 0) {
    throw new Error(`CSV file '${fileName}' contains no data rows`);
  }

  const columns: CsvColumnSummary[] = [];
  const anomalies: { column: string; rowIndices: number[]; reason: string; value: any; method?: 'IQR' | 'Z_SCORE' | 'MODIFIED_Z_SCORE' }[] = [];
  const numericColumnsData: Record<string, { values: number[]; rowIdx: number[] }> = {};
  const stringColumnsData: Record<string, string[]> = {};

  for (let colIdx = 0; colIdx < headers.length; colIdx++) {
    const colName = headers[colIdx];
    const rawValues = rows.map(r => r[colIdx]);

    let nullCount = 0;
    const nonNullValues: string[] = [];

    for (const val of rawValues) {
      if (val === undefined || val === null || val.trim() === '' || val.toLowerCase() === 'null' || val.toLowerCase() === 'n/a') {
        nullCount++;
      } else {
        nonNullValues.push(val.trim());
      }
    }

    const uniqueSet = new Set(nonNullValues);
    const uniqueCount = uniqueSet.size;

    let numericCount = 0;
    for (const v of nonNullValues) {
      if (!isNaN(Number(v))) numericCount++;
    }

    const isNumeric = nonNullValues.length > 0 && (numericCount / nonNullValues.length) >= 0.8;
    const type: 'numeric' | 'string' | 'date' = isNumeric ? 'numeric' : 'string';

    const colSummary: CsvColumnSummary = {
      name: colName,
      type,
      nullCount,
      uniqueCount,
      sampleValues: nonNullValues.slice(0, 5)
    };

    if (isNumeric) {
      const nums: number[] = [];
      const rowIdxs: number[] = [];
      for (let r = 0; r < rows.length; r++) {
        const rawCell = rows[r][colIdx];
        if (rawCell !== undefined && rawCell !== null && rawCell.trim() !== '' && rawCell.toLowerCase() !== 'null' && rawCell.toLowerCase() !== 'n/a') {
          const parsed = Number(rawCell);
          if (!isNaN(parsed)) {
            nums.push(parsed);
            rowIdxs.push(r);
          }
        }
      }

      numericColumnsData[colName] = { values: nums, rowIdx: rowIdxs };

      if (nums.length > 0) {
        const sorted = [...nums].sort((a, b) => a - b);
        const n = sorted.length;
        const sum = sorted.reduce((a, b) => a + b, 0);
        const mean = sum / n;
        const min = sorted[0];
        const max = sorted[n - 1];

        const mid = Math.floor(n / 2);
        const median = n % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;

        const freq: Record<number, number> = {};
        let maxF = 0;
        for (const val of sorted) {
          freq[val] = (freq[val] || 0) + 1;
          if (freq[val] > maxF) maxF = freq[val];
        }
        const modes = maxF > 1 ? Object.keys(freq).filter(k => freq[Number(k)] === maxF).map(Number) : [];

        let ss = 0;
        let m3 = 0;
        let m4 = 0;
        for (const val of sorted) {
          const diff = val - mean;
          ss += diff * diff;
          m3 += Math.pow(diff, 3);
          m4 += Math.pow(diff, 4);
        }

        const popVar = ss / n;
        const sampleVar = n > 1 ? ss / (n - 1) : 0;
        const popStdDev = Math.sqrt(popVar);
        const sampleStdDev = Math.sqrt(sampleVar);

        const getPercentile = (p: number): number => {
          const idx = (p / 100) * (n - 1);
          const low = Math.floor(idx);
          const high = Math.ceil(idx);
          const weight = idx - low;
          return sorted[low] + weight * (sorted[high] - sorted[low]);
        };

        const q1 = getPercentile(25);
        const q3 = getPercentile(75);
        const iqr = q3 - q1;

        let skewness = 0;
        let kurtosis = 0;
        if (n > 2 && popStdDev > 0) {
          skewness = (m3 / n) / Math.pow(popStdDev, 3);
          kurtosis = ((m4 / n) / Math.pow(popStdDev, 4)) - 3;
        }

        colSummary.min = min;
        colSummary.max = max;
        colSummary.mean = mean;
        colSummary.median = median;
        colSummary.mode = modes;
        colSummary.sum = sum;
        colSummary.stdDev = sampleStdDev;
        colSummary.sampleStdDev = sampleStdDev;
        colSummary.populationStdDev = popStdDev;
        colSummary.sampleVariance = sampleVar;
        colSummary.populationVariance = popVar;
        colSummary.q1 = q1;
        colSummary.q3 = q3;
        colSummary.iqr = iqr;
        colSummary.coeffOfVariation = mean !== 0 ? sampleStdDev / mean : 0;
        colSummary.skewness = skewness;
        colSummary.kurtosis = kurtosis;
        colSummary.percentiles = {
          5: getPercentile(5),
          25: q1,
          50: median,
          75: q3,
          95: getPercentile(95),
          99: getPercentile(99)
        };

        const iqrLower = q1 - 1.5 * iqr;
        const iqrUpper = q3 + 1.5 * iqr;
        const iqrOutlierRows: number[] = [];
        const outlierVals: number[] = [];

        for (let idx = 0; idx < nums.length; idx++) {
          const v = nums[idx];
          if (v < iqrLower || v > iqrUpper) {
            iqrOutlierRows.push(rowIdxs[idx]);
            outlierVals.push(v);
          }
        }

        if (iqrOutlierRows.length > 0) {
          anomalies.push({
            column: colName,
            rowIndices: iqrOutlierRows,
            reason: `IQR Outlier: Value outside bounds [${iqrLower.toFixed(2)}, ${iqrUpper.toFixed(2)}]`,
            value: outlierVals.slice(0, 5),
            method: 'IQR'
          });
        }

        if (nums.length >= 4) {
          const timeIndices = Array.from({ length: nums.length }, (_, i) => i);
          const reg = calculateLinearRegression(timeIndices, nums);
          const splitIdx = Math.floor(nums.length * 0.7);
          const trainX = timeIndices.slice(0, splitIdx);
          const trainY = nums.slice(0, splitIdx);
          const testX = timeIndices.slice(splitIdx);
          const testY = nums.slice(splitIdx);

          const trainReg = calculateLinearRegression(trainX, trainY);
          let testSSE = 0;
          let testSST = 0;
          const testMean = testY.reduce((a, b) => a + b, 0) / testY.length;

          for (let ti = 0; ti < testX.length; ti++) {
            const pred = trainReg.slope * testX[ti] + trainReg.intercept;
            testSSE += Math.pow(testY[ti] - pred, 2);
            testSST += Math.pow(testY[ti] - testMean, 2);
          }

          const walkForwardR2 = testSST > 0 ? Math.max(0, 1 - (testSSE / testSST)) : 0;
          const next3 = [1, 2, 3].map(step => reg.slope * (nums.length - 1 + step) + reg.intercept);

          colSummary.timeSeriesAnalysis = {
            isTemporal: true,
            trendSlope: reg.slope,
            intercept: reg.intercept,
            rSquared: reg.rSquared,
            mae: reg.mae,
            rmse: reg.rmse,
            mape: reg.mape,
            forecastNext3: next3,
            walkForwardValidationR2: walkForwardR2,
            interpretation: reg.slope > 0 ? 'Monotonically upward trend' : reg.slope < 0 ? 'Monotonically downward trend' : 'Flat stationary'
          };
        }
      }
    } else {
      stringColumnsData[colName] = rawValues;
    }

    columns.push(colSummary);
  }

  const groupAggregations: Record<string, Record<string, number>> = {};
  const numColNames = Object.keys(numericColumnsData);
  const strColNames = Object.keys(stringColumnsData);

  for (const numCol of numColNames) {
    for (const strCol of strColNames) {
      const aggKey = `${numCol.toLowerCase()}_by_${strCol.toLowerCase()}`;
      const aggMap: Record<string, number> = {};
      const numColIdx = headers.indexOf(numCol);
      const strColIdx = headers.indexOf(strCol);

      for (let r = 0; r < rows.length; r++) {
        const groupVal = rows[r][strColIdx]?.trim();
        const numVal = Number(rows[r][numColIdx]);
        if (groupVal && !isNaN(numVal)) {
          aggMap[groupVal] = (aggMap[groupVal] || 0) + numVal;
        }
      }

      if (Object.keys(aggMap).length > 0) {
        groupAggregations[aggKey] = aggMap;
      }
    }
  }

  const numericNames = Object.keys(numericColumnsData);
  const correlations: { var1: string; var2: string; pearsonR: number; pValue: number; significant: boolean }[] = [];
  const hypothesisTests: StatisticalInferenceResult[] = [];

  for (let i = 0; i < numericNames.length; i++) {
    for (let j = i + 1; j < numericNames.length; j++) {
      const v1Name = numericNames[i];
      const v2Name = numericNames[j];
      const data1 = numericColumnsData[v1Name].values;
      const data2 = numericColumnsData[v2Name].values;

      const n = Math.min(data1.length, data2.length);
      if (n >= 4) {
        const sub1 = data1.slice(0, n);
        const sub2 = data2.slice(0, n);

        const m1 = sub1.reduce((a, b) => a + b, 0) / n;
        const m2 = sub2.reduce((a, b) => a + b, 0) / n;

        let num = 0;
        let d1 = 0;
        let d2 = 0;
        for (let k = 0; k < n; k++) {
          const diff1 = sub1[k] - m1;
          const diff2 = sub2[k] - m2;
          num += diff1 * diff2;
          d1 += diff1 * diff1;
          d2 += diff2 * diff2;
        }

        const denom = Math.sqrt(d1 * d2);
        const r = denom > 0 ? num / denom : 0;
        const df = n - 2;
        const tStat = (1 - r * r > 0 && df > 0) ? r * Math.sqrt(df / (1 - r * r)) : 0;
        const pVal = studentTPValue(tStat, df);

        correlations.push({
          var1: v1Name,
          var2: v2Name,
          pearsonR: r,
          pValue: pVal,
          significant: pVal < 0.05
        });
      }
    }
  }

  let summary = `### Deterministic Data Analysis: ${fileName}\n\n`;
  summary += `**Dataset Overview:** ${rowCount} rows, ${columnCount} columns.\n\n`;
  summary += `| Column | Type | Nulls | Unique | Mean | StdDev | Sum | IQR |\n`;
  summary += `| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n`;

  for (const c of columns) {
    if (c.type === 'numeric' && c.mean !== undefined) {
      summary += `| **${c.name}** | numeric | ${c.nullCount} | ${c.uniqueCount} | ${c.mean.toFixed(2)} | ${c.stdDev?.toFixed(2) || 'N/A'} | ${c.sum ?? 'N/A'} | ${c.iqr?.toFixed(2) || 'N/A'} |\n`;
    } else {
      summary += `| **${c.name}** | ${c.type} | ${c.nullCount} | ${c.uniqueCount} | - | - | - | - |\n`;
    }
  }

  const aggKeys = Object.keys(groupAggregations);
  if (aggKeys.length > 0) {
    summary += `\n#### Group Aggregations\n`;
    for (const key of aggKeys) {
      summary += `\n**${key}:**\n`;
      for (const [groupName, aggVal] of Object.entries(groupAggregations[key])) {
        summary += `- **${groupName}**: ${aggVal}\n`;
      }
    }
  }

  if (anomalies.length > 0) {
    summary += `\n#### Detected Statistical Anomalies\n`;
    for (const a of anomalies) {
      summary += `- **${a.column}**: ${a.reason} (Affects rows: ${a.rowIndices.slice(0, 6).join(', ')}${a.rowIndices.length > 6 ? '...' : ''})\n`;
    }
  }

  return {
    fileName,
    rowCount,
    columnCount,
    columns,
    anomalies,
    groupAggregations: aggKeys.length > 0 ? groupAggregations : undefined,
    correlations,
    hypothesisTests,
    generatedSummaryMarkdown: summary,
    parsedAt: Date.now()
  };
}
