from rich import print

from vechord import (
    LocalLoader,
    Pipeline,
    RegexChunker,
    SimpleExtractor,
    SpacyDenseEmbedding,
    VectorChordClient,
)

if __name__ == "__main__":
    pipe = Pipeline(
        client=VectorChordClient(
            "local_pdf", "postgresql://postgres:postgres@172.17.0.1:5432/"
        ),
        loader=LocalLoader("data", include=[".pdf"]),
        extractor=SimpleExtractor(),
        chunker=RegexChunker(),
        emb=SpacyDenseEmbedding(),
    )
    pipe.run()

    print(pipe.query("vector search"))
