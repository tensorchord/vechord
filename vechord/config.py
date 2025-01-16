import msgspec


class EmbeddingConfig(msgspec.Struct, kw_only=True):
    model: str = "thenlper/gte-base"
    dim: int = 768
    api_key: str = "fake"
    api_url: str = "http://127.0.0.1:8000"
    timeout: int = 300


class DatabaseConfig(msgspec.Struct, kw_only=True):
    url: str = "postgresql://postgres:password@127.0.0.1:5432/"


class SourceConfig(msgspec.Struct, kw_only=True):
    local: str = msgspec.UNSET


class Config(msgspec.Struct, kw_only=True):
    embedding: EmbeddingConfig = msgspec.UNSET
    database: DatabaseConfig = DatabaseConfig()
    source: SourceConfig = SourceConfig()
