import dataclasses
import enum
import inspect
from abc import ABC, abstractmethod
from datetime import datetime
from types import UnionType
from typing import (
    Annotated,
    Any,
    Generic,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    runtime_checkable,
)
from uuid import UUID

import msgspec
import numpy as np
from psycopg.types.json import Jsonb
from uuid_utils.compat import uuid7

__all__ = [
    "DefaultDocument",
    "ForeignKey",
    "Keyword",
    "KeywordIndex",
    "MultiVectorIndex",
    "PrimaryKeyAutoIncrease",
    "PrimaryKeyUUID",
    "Table",
    "Vector",
    "VectorIndex",
    "create_chunk_with_dim",
]


@runtime_checkable
class VechordType(Protocol):
    @classmethod
    def schema(cls) -> str:
        pass


class VectorMeta(type):
    def __getitem__(self, dim: int):
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(
                f"dim must be a positive integer, not `{type(dim)}({dim})`"
            )
        return create_vector_type(dim)


V = TypeVar("V")


class Vector(Generic[V], metaclass=VectorMeta):
    """Vector type with fixed dimension.

    User can assign `np.ndarray` with `np.float32` type or `list[float]` type.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Use Vector[dim] to create a vector type")

    @classmethod
    def schema(cls) -> str:
        raise NotImplementedError("Should be implemented by the subclass: Vector[dim]")


def create_vector_type(dim: int) -> Type[Vector]:
    name = f"Vector[{dim}]"

    class SpecificVector(Vector, np.ndarray):
        nonlocal dim
        _dim: int = dim

        def __new__(cls, vec: list[float] | np.ndarray):
            if isinstance(vec, np.ndarray):
                if vec.shape != (dim,):
                    raise ValueError(f"expected shape ({dim},), got {vec.shape}")
            elif isinstance(vec, list):
                if len(vec) != dim:
                    raise ValueError(f"expected length {dim}, got {len(vec)}")
                vec = np.array(vec, dtype=np.float32)
            else:
                raise ValueError("expected list or np.ndarray")
            return np.asarray(vec, dtype=np.float32)

        @classmethod
        def schema(cls):
            return f"VECTOR({cls._dim})"

    SpecificVector.__name__ = name
    return SpecificVector


class ForeignKeyMeta(type):
    def __getitem__(self, ref):
        return create_foreign_key_type(ref)


F = TypeVar("F")


class ForeignKey(Generic[F], metaclass=ForeignKeyMeta):
    """Reference to another table's attribute as a foreign key.

    This should be used in the `Annotated[]` type hint.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Use ForeignKey[ref] to create a foreign key type")

    @classmethod
    def schema(cls) -> str:
        raise NotImplementedError(
            "Should be implemented by the subclass: ForeignKey[ref]"
        )


def create_foreign_key_type(ref) -> Type[ForeignKey]:
    name = f"ForeignKey[{ref}]"

    class SpecificForeignKey(ForeignKey):
        nonlocal ref
        _ref = ref

        def __init__(self, value):
            self.value = value

        @classmethod
        def schema(cls):
            ref_cls = cls._ref.__objclass__.__name__.lower()
            ref_val = cls._ref.__name__
            return f"REFERENCES {{namespace}}_{ref_cls}({ref_val}) ON DELETE CASCADE"

    SpecificForeignKey.__name__ = name
    return SpecificForeignKey


class PrimaryKeyAutoIncrease(int):
    """Primary key with auto-increment ID type. (wrap ``int``)"""

    @classmethod
    def schema(cls) -> str:
        return "BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY"


class PrimaryKeyUUID(UUID):
    """Primary key with UUID type. (wrap ``UUID``)

    This doesn't come with auto-generate, because PostgreSQL doesn't support UUID v7, while v4
    is purely random and not sortable.

    Choose this one over :class:`PrimaryKeyAutoIncrease` when you need universal uniqueness.

    We suggest to use:

    .. code-block:: python

        class MyTable(Table):
            uid: PrimaryKeyUUID = msgspec.field(default_factory=PrimaryKeyUUID.factory)
    """

    @classmethod
    def schema(cls) -> str:
        return "UUID PRIMARY KEY"

    @classmethod
    def factory(cls):
        return uuid7()


class Keyword(str):
    """Keyword type for text search. (wrap ``str``)

    User can assign the `str` type, it will be tokenized and converted to
    `bm25vector` in PostgreSQL.
    """

    _model: Literal["bert_base_uncased", "wiki_tocken"] = "bert_base_uncased"

    @classmethod
    def schema(cls) -> str:
        return "bm25vector"

    @classmethod
    def with_model(cls, model: Literal["bert_base_uncased", "wiki_tocken"]) -> Type:
        """TODO: test this"""
        cls._model = model
        return cls


TYPE_TO_PSQL = {
    int: "BIGINT",
    str: "TEXT",
    bytes: "BYTEA",
    float: "FLOAT 8",
    bool: "BOOLEAN",
    UUID: "UUID",
    datetime: "TIMESTAMPTZ",
    Jsonb: "JSONB",
}


def is_optional_type(typ: Type) -> bool:
    return (get_origin(typ) is Union or get_origin(typ) is UnionType) and type(
        None
    ) in get_args(typ)


def get_first_type_from_optional(typ: Type) -> Type:
    for arg in get_args(typ):
        if arg is not type(None):
            return arg
    raise ValueError(f"no non-None type in {typ}")


def is_list_of_vector_type(typ: Type) -> bool:
    return get_origin(typ) is list and issubclass(typ.__args__[0].__class__, VectorMeta)


def type_to_psql(typ) -> str:
    if is_optional_type(typ):
        typ = get_first_type_from_optional(typ)

    if get_origin(typ) is Annotated:
        meta = typ.__metadata__
        origin = typ.__origin__
        schema = [type_to_psql(origin)]
        for m in meta:
            if inspect.isclass(m) and issubclass(m, ForeignKey):
                schema.append(m.schema())
        return " ".join(schema)
    elif get_origin(typ) is list:
        return f"{type_to_psql(typ.__args__[0])}[]"
    if isinstance(typ, VechordType):
        return typ.schema()
    if typ in TYPE_TO_PSQL:
        return TYPE_TO_PSQL[typ]
    raise ValueError(f"unsupported type {typ}")


class VectorDistance(enum.Enum):
    L2 = "l2"
    COS = "cos"
    DOT = "dot"


class BaseIndex(ABC):
    name: str = ""
    index: str = ""
    op_name: str = ""
    op_symbol: str = ""

    def verify(self):
        """This will construct a new data-only struct to verify all the fields.

        This is to ensure that the `__post_init__` will not be called recursively.
        """
        fields = [
            (
                f.name,
                f.type,
                f.default
                if f.default is not f.default_factory
                else msgspec.field(default_factory=f.default_factory),
            )
            for f in dataclasses.fields(self)
        ]
        spec = msgspec.defstruct(f"Spec{self.__class__.__name__}", fields)
        instance = msgspec.convert(dataclasses.asdict(self), spec)
        for field in instance.__struct_fields__:
            setattr(self, field, getattr(instance, field))

    @abstractmethod
    def config(self) -> str:
        raise NotImplementedError


class IndexColumn(msgspec.Struct, frozen=True, order=True):
    name: str
    index: BaseIndex


@dataclasses.dataclass
class VectorIndex(BaseIndex):
    distance: VectorDistance = VectorDistance.L2
    lists: int = 1

    def __post_init__(self):
        self.verify()
        self.name = "vec_idx"
        self.index = "vchordrq"
        match self.distance:
            case VectorDistance.L2:
                self.op_symbol = "<->"
                self.op_name = "vector_l2_ops"
            case VectorDistance.COS:
                self.op_symbol = "<=>"
                self.op_name = "vector_cosine_ops"
            case VectorDistance.DOT:
                self.op_symbol = "<#>"
                self.op_name = "vector_ip_ops"

    def config(self):
        is_l2 = self.distance == VectorDistance.L2
        return f"""
residual_quantization = {"true" if is_l2 else "false"}
[build.internal]
lists = [{self.lists}]
spherical_centroids = {"false" if is_l2 else "true"}
"""


@dataclasses.dataclass
class MultiVectorIndex(BaseIndex):
    lists: Optional[int] = None

    def __post_init__(self):
        self.verify()
        self.name = "multivec_idx"
        self.index = "vchordrq"
        self.op_name = "vector_maxsim_ops"
        self.op_symbol = "@#"

    def config(self):
        return f"build.internal.lists = [{self.lists or ''}]"


@dataclasses.dataclass
class KeywordIndex(BaseIndex):
    model: Literal["bert_base_uncased", "wiki_tocken"] = "bert_base_uncased"

    def __post_init__(self):
        self.verify()
        self.name = "keyword_idx"
        self.index = "bm25"
        self.op_name = "bm25_ops"
        self.op_symbol = "<&>"

    def config(self):
        return ""


class Storage(msgspec.Struct):
    @classmethod
    def name(cls) -> str:
        return cls.__name__.lower()

    @classmethod
    def fields(cls) -> Sequence[str]:
        return cls.__struct_fields__

    @classmethod
    def partial_init(cls, **kwargs):
        fields = cls.fields()
        args = dict(zip(fields, [msgspec.UNSET] * len(fields), strict=False)) | kwargs
        return cls(**args)


class Table(Storage):
    """Base class for table definition."""

    @classmethod
    def table_schema(cls) -> Sequence[tuple[str, str]]:
        """Generate the table schema from the class attributes' type hints."""
        hints = get_type_hints(cls, include_extras=True)
        return tuple((name, type_to_psql(typ)) for name, typ in hints.items())

    @classmethod
    def vector_column(cls) -> Optional[IndexColumn]:
        """Get the vector column name."""
        for name, typ in get_type_hints(cls, include_extras=True).items():
            if issubclass(typ.__class__, VectorMeta):
                return IndexColumn(name, VectorIndex())
            elif get_origin(typ) is Annotated and issubclass(
                typ.__origin__.__class__, VectorMeta
            ):
                for m in typ.__metadata__:
                    if isinstance(m, VectorIndex):
                        return IndexColumn(name, m)
        return None

    @classmethod
    def multivec_column(cls) -> Optional[IndexColumn]:
        """Get the multivec column name."""
        for name, typ in get_type_hints(cls, include_extras=True).items():
            if is_list_of_vector_type(typ):
                return IndexColumn(name, MultiVectorIndex())
            elif get_origin(typ) is Annotated:
                inner_type = typ.__origin__
                if is_list_of_vector_type(inner_type):
                    for m in typ.__metadata__:
                        if isinstance(m, MultiVectorIndex):
                            return IndexColumn(name, m)
        return None

    @classmethod
    def keyword_column(cls) -> Optional[IndexColumn]:
        """Get the keyword column name."""
        for name, typ in get_type_hints(cls, include_extras=True).items():
            if typ is Keyword:
                return IndexColumn(name, KeywordIndex())
            elif get_origin(typ) is Annotated and typ.__origin__ is Keyword:
                for m in typ.__metadata__:
                    if isinstance(m, KeywordIndex):
                        return IndexColumn(name, m)
        return None

    @classmethod
    def non_vec_columns(cls) -> Sequence[str]:
        """Get the column names that are not vector or keyword."""
        exclude = (
            col.name if col else col
            for col in (
                cls.vector_column(),
                cls.keyword_column(),
                cls.multivec_column(),
            )
        )
        return tuple(field for field in cls.fields() if field not in exclude)

    @classmethod
    def keyword_tokenizer(cls) -> Optional[str]:
        """Get the keyword tokenizer."""
        for _, typ in get_type_hints(cls, include_extras=True).items():
            if typ is Keyword:
                return typ._model
            elif get_origin(typ) is Annotated and typ.__origin__ is Keyword:
                return typ.__origin__._model
        return None

    @classmethod
    def primary_key(cls) -> Optional[str]:
        """Get the primary key column name."""
        for name, typ in get_type_hints(cls, include_extras=True).items():
            typ_cls = (
                get_first_type_from_optional(typ) if is_optional_type(typ) else typ
            )
            if inspect.isclass(typ_cls) and issubclass(
                typ_cls, (PrimaryKeyUUID, PrimaryKeyAutoIncrease)
            ):
                return name
        return None

    def todict(self) -> dict[str, Any]:
        """Convert the table instance to a dictionary.

        This will ignore the values like:

        - `msgspec.UNSET`
        - default value is None and the value is also None (mainly for PrimaryKeyAutoIncrease)
        """
        defaults = getattr(self, "__struct_defaults__", None)
        fields = self.fields()
        if not defaults:
            return {
                k: v
                for k, v in zip(fields, (getattr(self, f) for f in fields), strict=True)
                if v is not msgspec.UNSET
            }
        # ignore default values
        res = {}
        for k, d in zip(fields, defaults, strict=False):
            v = getattr(self, k)
            if v is not msgspec.UNSET and (d is msgspec.NODEFAULT or v is not d):
                res[k] = v
        return res


class DefaultDocument(Table, kw_only=True):
    """Default Document table class."""

    uid: PrimaryKeyUUID = msgspec.field(default_factory=PrimaryKeyUUID.factory)
    title: str = ""
    text: str
    created_at: datetime = msgspec.field(default_factory=datetime.now)


class _DefaultChunk(Table, kw_only=True):
    """A placeholder for the chunk table class.

    This is not intended to be used directly, but rather as a base class for
    creating chunk table classes with specific vector dimensions.
    """

    uid: PrimaryKeyUUID = msgspec.field(default_factory=PrimaryKeyUUID.factory)
    doc_id: Annotated[UUID, ForeignKey[DefaultDocument.uid]]
    text: str
    vec: Vector[1]
    keyword: Keyword


def create_chunk_with_dim(dim: int) -> Type[_DefaultChunk]:
    """Create a chunk table class with a specific vector dimension.

    This comes with vector and keyword column. It also has a foreign key to the
    :class:`DefaultDocument` table. (If this is used, the :class:`DefaultDocument`
    table must be registered too.)

    Args:
        dim: vector dimension.
    """
    if not isinstance(dim, int) or dim <= 0:
        raise ValueError(f"dim must be a positive integer, not `{type(dim)}({dim})`")

    DenseVector = Vector[dim]

    class DefaultChunk(_DefaultChunk):
        """A chunk table class with a specific vector dimension."""

        vec: DenseVector  # type: ignore

    return DefaultChunk
