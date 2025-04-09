# Toolkit

We provides some basic tools to help you build the RAG pipeline. But it's not limited to thses
internal tools. You can use whatever you like.

You may need to install with extras:

```bash
pip install vechord[gemini,openai,spacy,cohere]
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
    - {py:class}`~vechord.embedding.SpacyDenseEmbedding`: Spacy embedding
- Evaluate
    - {py:class}`~vechord.evaluate.GeminiEvaluator`: Gemini based evaluator
- Extract
    - {py:class}`~vechord.extract.SimpleExtractor`: Simple extractor
    - {py:class}`~vechord.extract.GeminiExtractor`: Gemini extractor
- Rerank
    - {py:class}`~vechord.rerank.CohereReranker`: Gemini based reranker
    - {py:class}`~vechord.rerank.ReciprocalRankFusion`: fuse function for hybrid retrieval
