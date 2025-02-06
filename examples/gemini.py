from rich import print

from vechord import (
    GeminiAugmenter,
    GeminiDenseEmbedding,
    GeminiExtractor,
    LocalLoader,
    Pipeline,
    WordLlamaChunker,
    VectorChordClient,
)

if __name__ == "__main__":
    pipe = Pipeline(
        client=VectorChordClient(
            "local_pdf", "postgresql://postgres:postgres@172.17.0.1:5432/"
        ),
        loader=LocalLoader("data", include=[".pdf"]),
        extractor=GeminiExtractor(),
        chunker=WordLlamaChunker(),
        emb=GeminiDenseEmbedding(),
        augmenter=GeminiAugmenter(),
    )
    pipe.run()

    print(pipe.query("vector search"))
