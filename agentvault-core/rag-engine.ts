export interface DocumentChunk {
  chunkId: string;
  sourceDoc: string;
  content: string;
  metadata: Record<string, any>;
  embedding?: number[];
}

export interface RetrievalResult {
  chunk: DocumentChunk;
  similarity: number;
}

export interface GroundedVerification {
  isGrounded: boolean;
  faithfulnessScore: number;
  supportedClaims: string[];
  unsupportedClaims: string[];
  citations: string[];
}

export class RagEngine {
  private chunks: DocumentChunk[] = [];

  public chunkDocument(sourceDoc: string, text: string, maxChunkLength: number = 400, overlap: number = 80): DocumentChunk[] {
    if (!text || text.trim().length === 0) return [];

    const paragraphs = text.split(/\n\s*\n/);
    const createdChunks: DocumentChunk[] = [];
    let currentChunk = '';
    let seq = 1;

    for (const paragraph of paragraphs) {
      if ((currentChunk + '\n\n' + paragraph).length <= maxChunkLength) {
        currentChunk = currentChunk ? currentChunk + '\n\n' + paragraph : paragraph;
      } else {
        if (currentChunk.trim().length > 0) {
          createdChunks.push({
            chunkId: `${sourceDoc}_part_${seq++}`,
            sourceDoc,
            content: currentChunk.trim(),
            metadata: { charLength: currentChunk.length, createdAt: Date.now() }
          });
        }
        const sliceStart = Math.max(0, currentChunk.length - overlap);
        currentChunk = currentChunk.substring(sliceStart) + '\n\n' + paragraph;
      }
    }

    if (currentChunk.trim().length > 0) {
      createdChunks.push({
        chunkId: `${sourceDoc}_part_${seq++}`,
        sourceDoc,
        content: currentChunk.trim(),
        metadata: { charLength: currentChunk.length, createdAt: Date.now() }
      });
    }

    this.chunks.push(...createdChunks);
    return createdChunks;
  }

  public static cosineSimilarity(vecA: number[], vecB: number[]): number {
    if (vecA.length !== vecB.length || vecA.length === 0) return 0;
    let dot = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < vecA.length; i++) {
      dot += vecA[i] * vecB[i];
      normA += vecA[i] * vecA[i];
      normB += vecB[i] * vecB[i];
    }
    const denom = Math.sqrt(normA) * Math.sqrt(normB);
    return denom > 0 ? dot / denom : 0;
  }

  public retrieve(queryEmbedding: number[], topK: number = 3, minThreshold: number = 0.5): RetrievalResult[] {
    const scored: RetrievalResult[] = [];

    for (const chunk of this.chunks) {
      if (!chunk.embedding) continue;
      const sim = RagEngine.cosineSimilarity(queryEmbedding, chunk.embedding);
      if (sim >= minThreshold) {
        scored.push({ chunk, similarity: sim });
      }
    }

    return scored.sort((a, b) => b.similarity - a.similarity).slice(0, topK);
  }

  public indexChunks(chunks: DocumentChunk[]): void {
    this.chunks = chunks;
  }

  public getAllChunks(): DocumentChunk[] {
    return this.chunks;
  }

  public verifyFaithfulness(generatedText: string, evidenceChunks: DocumentChunk[]): GroundedVerification {
    if (!generatedText || generatedText.trim().length === 0) {
      return { isGrounded: true, faithfulnessScore: 1.0, supportedClaims: [], unsupportedClaims: [], citations: [] };
    }

    const sentences = generatedText
      .split(/[.!?]+/)
      .map(s => s.trim())
      .filter(s => s.length > 15);

    if (sentences.length === 0) {
      return { isGrounded: true, faithfulnessScore: 1.0, supportedClaims: [], unsupportedClaims: [], citations: [] };
    }

    const supported: string[] = [];
    const unsupported: string[] = [];
    const citations: Set<string> = new Set();

    const corpus = evidenceChunks
      .map(c => c.content.toLowerCase().replace(/[-_]/g, ' '))
      .join(' ');

    for (const sentence of sentences) {
      const words = sentence
        .toLowerCase()
        .replace(/[-_]/g, ' ')
        .replace(/[^a-z0-9\s]/g, '')
        .split(/\s+/)
        .filter(w => w.length >= 3);

      if (words.length === 0) continue;

      let matchedWords = 0;
      for (const w of words) {
        if (corpus.includes(w)) {
          matchedWords++;
        }
      }

      const matchRatio = matchedWords / words.length;
      if (matchRatio >= 0.6) {
        supported.push(sentence);
        for (const ev of evidenceChunks) {
          if (words.some(w => ev.content.toLowerCase().includes(w))) {
            citations.add(ev.sourceDoc);
          }
        }
      } else {
        unsupported.push(sentence);
      }
    }

    const total = supported.length + unsupported.length;
    const faithfulnessScore = total > 0 ? supported.length / total : 1.0;

    return {
      isGrounded: faithfulnessScore >= 0.8 && unsupported.length === 0,
      faithfulnessScore,
      supportedClaims: supported,
      unsupportedClaims: unsupported,
      citations: Array.from(citations)
    };
  }
}
