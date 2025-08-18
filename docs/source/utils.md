# Toolkit

We provides some basic tools to help you build the RAG pipeline. But it's not limited to these
internal tools. You can use whatever you like.

You may need to install with extras:

```bash
pip install vechord[openai,spacy,cohere]
```

- Augment
    - {py:class}`~vechord.augment.GeminiAugmenter`: for contextual retrieval
- Chunk
    - {py:class}`~vechord.chunk.RegexChunker`: Regex based chunker
    - {py:class}`~vechord.chunk.SpacyChunker`: Spacy based chunker
    - {py:class}`~vechord.chunk.GeminiChunker`: Gemini based chunker
- Embedding
    - {py:class}`~vechord.embedding.GeminiDenseEmbedding`: Gemini embedding
    - {py:class}`~vechord.embedding.OpenAIDenseEmbedding`: OpenAI embedding
    - {py:class}`~vechord.embedding.JinaDenseEmbedding`: JinaAI embedding
    - {py:class}`~vechord.embedding.VoyageDenseEmbedding`: VoyageAI embedding
    - {py:class}`~vechord.embedding.SpacyDenseEmbedding`: Spacy embedding
- Evaluate
    - {py:class}`~vechord.evaluate.GeminiEvaluator`: Gemini based query generator
    - {py:class}`~vechord.evaluate.GeminiUMBRELAEvaluator`: Gemini UMBRELA evaluator
- GroundTruth
    - {py:class}`~vechord.groundtruth.GroundTruth`: generate ground truth
- Extract
    - {py:class}`~vechord.extract.SimpleExtractor`: Simple extractor
    - {py:class}`~vechord.extract.GeminiExtractor`: Gemini extractor
- Rerank
    - {py:class}`~vechord.rerank.CohereReranker`: Cohere reranker
    - {py:class}`~vechord.rerank.JinaReranker`: Jina MultiModal reranker
    - {py:class}`~vechord.rerank.ReciprocalRankFusion`: fuse function for hybrid retrieval
