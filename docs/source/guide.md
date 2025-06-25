# Guide

## Define the table

Inherite the {py:class}`~vechord.spec.Table` class and define the columns as attributes with the
type hints. Some advanced configuration can be done by using the {py:class}`typing.Annotated`.

### Choose a primary key

- {py:class}`~vechord.spec.PrimaryKeyAutoIncrease`: generate an auto-incrementing integer as the primary key
- {py:class}`~vechord.spec.PrimaryKeyUUID`: use `uuid7` as the primary key, suitable for distributed systems or general purposes
- `int` or `str`: insert the key manually

### Vector and Keyword search

- {py:class}`~vechord.spec.Vector`: define a vector column with dimensions, it's recommended to define something like `DenseVector = Vector[3072]` and use it in all tables. This accepts `list[float]` or `numpy.ndarray` as the input. For now, it only supports `f32` type.
  - for multivector, use `list[DenseVector]` as the type hint
- {py:class}`~vechord.spec.Keyword`: define a keyword column that the `str` will be tokenized and stored as the `bm25vector` type. This accepts `str` as the input.

### Configure the Index

The default index is suitable for small datasets (less than 100k). For larger datasets, you can
customize the index configuration by using the {py:class}`typing.Annotated` with:

- {py:class}`~vechord.spec.VectorIndex`: configure the `lists` and `distance` operators.
- {py:class}`~vechord.spec.MultiVectorIndex`: configure the `lists`.

```python
DenseVector = Vector[3072]

class MyTable(Table, kw_only=True):
    uid: PrimaryKeyUUID = msgspec.field(default_factory=PrimaryKeyUUID.factory)
    vec: Annotated[DenseVector, VectorIndex(lists=128)]
    text: str
```

:::{tip}
If you need to use a customized tokenizer, please refer to the [VectorChord-bm25 document](https://github.com/tensorchord/VectorChord-bm25/?tab=readme-ov-file#more-examples).
:::

### Use the foreign key to link tables

By default, the foreign key will add `REFERENCES ON DELETE CASCADE`.

```python
class SubTable(Table, kw_only=True):
    uid: PrimaryKeyUUID = msgspec.field(default_factory=PrimaryKeyUUID.factory)
    text: str
    mytable_uid: Annotated[UUID, ForeignKey[MyTable.uid]]
```

### JSONB

If you want to store a JSONB column, you can define like:

```python
from psycopg.types.json import Jsonb

class MyJsonTable(Table, kw_only=True):
    uid: PrimaryKeyUUID = msgspec.field(default_factory=PrimaryKeyUUID.factory)
    json: JSONB

item = MyJsonTable(json=Jsonb({"key": "value"}))
```

## Inject with decorator

The decorator {py:meth}`~vechord.registry.VechordRegistry.inject` can be used to load the
function arguments from the database and dump the return values to the database.

To use this decorator, you need to specify at least one of the `input` or `output` with
the table class you have defined.

- `input=Type[Table]`: will load the specified columns rom the database and inject the data to the decorated function arguments
  - if `input=None`, the function will need to pass the arguments manually
- `output=Type[Table]`: will dump the return values to the database (will also need to annotate the return type with the provided table class or a list of the table class)
  - if `output=None`, you can get the return value from the functiona call

The following example uses the pre-defined tables:

- {py:class}`~vechord.spec.DefaultDocument`
- {py:func}`~vechord.spec.create_chunk_with_dim`

```python
from uuid import UUID
import httpx
from vechord.registry import VechordRegistry
from vechord.extract import SimpleExtractor
from vechord.embedding import GeminiDenseEmbedding
from vechord.spec import DefaultDocument, create_chunk_with_dim

DefaultChunk = create_chunk_with_dim(3072)
vr = VechordRegistry(namespace="test", url="postgresql://postgres:postgres@127.0.0.1:5432/", tables=[DefaultDocument, DefaultChunk])
extractor = SimpleExtractor()
emb = GeminiDenseEmbedding()


@vr.inject(output=DefaultDocument)
async def add_document(url: str) -> DefaultDocument:
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        text = extractor.extract_html(resp.text)
        return DefaultDocument(title=url, text=text)


@vr.inject(input=Document, output=DefaultChunk)
async def add_chunk(uid: UUID, text: str) -> list[DefaultChunk]:
    chunks = text.split("\n")
    return [DefaultChunk(doc_id=uid, vec=await emb.vectorize_chunk(t), text=t) for t in chunks]


async def main():
    async with vr, emb:
        for url in ["https://paulgraham.com/best.html", "https://paulgraham.com/read.html"]:
            await add_document(url)
        await add_chunk()
```

### Select/Insert/Delete

We also provide some functions to select, insert and delete the data from the database.

- {py:meth}`~vechord.registry.VechordRegistry.select_by`
- {py:meth}`~vechord.registry.VechordRegistry.insert`
- {py:meth}`~vechord.registry.VechordRegistry.copy_bulk`
- {py:meth}`~vechord.registry.VechordRegistry.remove_by`

```python
docs = await vr.select_by(DefaultDocument.partial_init())
await vr.insert(DefaultDocument(text="hello world"))
await vr.copy_bulk([DefaultDocument(text="hello world"), DefaultDocument(text="hello vector")])
await vr.remove_by(DefaultDocument.partial_init())
```

## Transaction

Use the {py:class}`~vechord.registry.VechordPipeline` to run multiple functions in a transaction.

This also guarantees that the decorated functions will only load the data from the current
transaction instead of the whole table. So users can focus on the data processing part.

```python
pipeline = vr.create_pipeline([add_document, add_chunk])
await pipeline.run("https://paulgraham.com/best.html")
```

## Search

We provide search interface for different types of queries:

- {py:meth}`~vechord.registry.VechordRegistry.search_by_vector`
- {py:meth}`~vechord.registry.VechordRegistry.search_by_keyword`
- {py:meth}`~vechord.registry.VechordRegistry.search_by_multivec`

```python
await vr.search_by_vector(DefaultChunk, await emb.vectorize_query("hey"), topk=10)
```

## Access the cursor

If you need to change some settings or use the cursor directly:

```python
await vr.client.get_cursor().execute("SET vchordrq.probes = 100;")
```
