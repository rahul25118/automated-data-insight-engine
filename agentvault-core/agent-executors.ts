import { RagEngine, DocumentChunk, GroundedVerification } from './rag-engine';

export interface EnhancedRealStatistics {
  mean: number;
  median: number;
  min: number;
  max: number;
  stdDev: number;
  count: number;
  variance?: number;
  q1?: number;
  q3?: number;
  iqr?: number;
  skewness?: number;
  kurtosis?: number;
}

export function computeRealStatistics(numbers: number[]): EnhancedRealStatistics {
  if (numbers.length === 0) {
    return { mean: 0, median: 0, min: 0, max: 0, stdDev: 0, count: 0 };
  }

  const sorted = [...numbers].sort((a, b) => a - b);
  const count = sorted.length;
  const min = sorted[0];
  const max = sorted[count - 1];
  const sum = sorted.reduce((acc, v) => acc + v, 0);
  const mean = sum / count;

  let median: number;
  const mid = Math.floor(count / 2);
  if (count % 2 === 0) {
    median = (sorted[mid - 1] + sorted[mid]) / 2;
  } else {
    median = sorted[mid];
  }

  let ss = 0;
  let m3 = 0;
  let m4 = 0;
  for (const v of sorted) {
    const diff = v - mean;
    ss += diff * diff;
    m3 += Math.pow(diff, 3);
    m4 += Math.pow(diff, 4);
  }
  const variance = ss / count;
  const stdDev = Math.sqrt(variance);

  const getP = (p: number) => {
    const idx = (p / 100) * (count - 1);
    const low = Math.floor(idx);
    const high = Math.ceil(idx);
    const w = idx - low;
    return sorted[low] + w * (sorted[high] - sorted[low]);
  };
  const q1 = getP(25);
  const q3 = getP(75);
  const iqr = q3 - q1;

  const skewness = count > 2 && stdDev > 0 ? (m3 / count) / Math.pow(stdDev, 3) : 0;
  const kurtosis = count > 3 && stdDev > 0 ? ((m4 / count) / Math.pow(stdDev, 4)) - 3 : 0;

  return {
    mean: Number(mean.toFixed(2)),
    median: Number(median.toFixed(2)),
    min: Number(min.toFixed(2)),
    max: Number(max.toFixed(2)),
    stdDev: Number(stdDev.toFixed(2)),
    variance: Number(variance.toFixed(2)),
    q1: Number(q1.toFixed(2)),
    q3: Number(q3.toFixed(2)),
    iqr: Number(iqr.toFixed(2)),
    skewness: Number(skewness.toFixed(2)),
    kurtosis: Number(kurtosis.toFixed(2)),
    count
  };
}

export function executeResearchAgent(query: string, rawCorpus: { source: string; text: string }[]): {
  chunks: DocumentChunk[];
  summary: string;
  sources: string[];
} {
  const rag = new RagEngine();
  const allChunks: DocumentChunk[] = [];
  for (const item of rawCorpus) {
    const chunks = rag.chunkDocument(item.source, item.text);
    allChunks.push(...chunks);
  }
  return {
    chunks: allChunks,
    summary: `Ingested ${allChunks.length} verifiable chunks across ${rawCorpus.length} sources for query '${query}'.`,
    sources: Array.from(new Set(rawCorpus.map(c => c.source)))
  };
}

export function executeVerificationAgent(
  claimsText: string,
  evidenceChunks: DocumentChunk[]
): GroundedVerification {
  const rag = new RagEngine();
  return rag.verifyFaithfulness(claimsText, evidenceChunks);
}
