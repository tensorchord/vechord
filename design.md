## Diagram

```mermaid
timeline
    title RAG
    section Ingestion
        Source: Local
              : Google Drive
              : Dropbox
              : Notion
        File: Document
            : Image
            : Audio
        Chunk: Text
             : Entities
             : Embedding
    section Query
        Analysis: Expansion
                : Keyword
                : Embedding
        Search: Vector Search
              : Full Text Search
              : Filter
        Rerank: ColBERT
    section Evaluation
        Metric: MAP
              : Recall
              : NDCG
```

## Pipeline

- request type
  - text
    - chunk
    - index: embedding | keyword | graph
  - image | pdf
    - ocr => text
    - multimodal embedding
    - recognize => graph

- index type
  - vector => vchord
  - keyword => vchord-bm25
  - graph
    - Entity: vector
    - Relation: vector

- search
  - vector
  - keyword
  - graph

