from datetime import datetime
from types import UnionType
from typing import (
    Annotated,
    Any,
    Generic,
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


class Vector(Generic[TypeVar("T")], metaclass=VectorMeta):
    """Vector type with fixed dimension."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Use Vector[dim] to create a vector type")

    @classmethod
    def schema(cls) -> str:
        raise NotImplementedError("Should be implemented by the subclass: Vector[dim]")


def create_vector_type(dim: int) -> Type[Vector]:
    class SpecificVector(Vector):
        nonlocal dim
        _dim: int = dim

        def __init__(self, vec: list[float] | np.ndarray):
            if isinstance(vec, np.ndarray):
                if vec.shape != (dim,):
                    raise ValueError(f"expected shape ({dim},), got {vec.shape}")
            elif isinstance(vec, list):
                if len(vec) != dim:
                    raise ValueError(f"expected length {dim}, got {len(vec)}")
            else:
                raise ValueError("expected list or np.ndarray")
            self.vec = vec

        @classmethod
        def schema(cls):
            return f"VECTOR({cls._dim})"

    SpecificVector.__name__ = f"Vector[{dim}]"
    return SpecificVector


class ForeignKeyMeta(type):
    def __getitem__(self, ref):
        return create_foreign_key_type(ref)


class ForeignKey(Generic[TypeVar("K")], metaclass=ForeignKeyMeta):
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

    SpecificForeignKey.__name__ = f"ForeignKey[{ref}]"
    return SpecificForeignKey


class PrimaryKeyAutoIncrease(int):
    """Primary key with auto-increment ID type."""

    @classmethod
    def schema(cls) -> str:
        return "BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY"


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


def is_optional_type(typ) -> bool:
    return (get_origin(typ) is Union or get_origin(typ) is UnionType) and type(
        None
    ) in get_args(typ)


def get_first_type_from_optional(typ) -> Type:
    for arg in get_args(typ):
        if arg is not type(None):
            return arg
    raise ValueError(f"no non-None type in {typ}")


def type_to_psql(typ) -> str:
    if is_optional_type(typ):
        typ = get_first_type_from_optional(typ)

    if get_origin(typ) is Annotated:
        meta = typ.__metadata__
        origin = typ.__origin__
        schema = [type_to_psql(origin)]
        for m in meta:
            if issubclass(m, ForeignKey):
                schema.append(m.schema())
        return " ".join(schema)
    if isinstance(typ, VechordType):
        return typ.schema()
    if typ in TYPE_TO_PSQL:
        return TYPE_TO_PSQL[typ]
    raise ValueError(f"unsupported type {typ}")


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
        return ((name, type_to_psql(typ)) for name, typ in hints.items())

    @classmethod
    def vector_column(cls) -> Optional[str]:
        """Get the vector column name."""
        for name, typ in get_type_hints(cls, include_extras=True).items():
            if issubclass(typ.__class__, VectorMeta):
                return name
        return None

    @classmethod
    def primary_key(cls) -> Optional[str]:
        """Get the primary key column name."""
        for name, typ in get_type_hints(cls, include_extras=True).items():
            typ_cls = (
                get_first_type_from_optional(typ) if is_optional_type(typ) else typ
            )
            if issubclass(typ_cls, PrimaryKeyAutoIncrease):
                return name
        return None

    def todict(self) -> dict[str, Any]:
        """Convert the table instance to a dictionary.

        This will ignore the default values.
        """
        defaults = getattr(self, "__struct_defaults__", None)
        fields = self.fields()
        if not defaults:
            return {k: getattr(self, k) for k in fields}
        # ignore default values
        res = {}
        for k, d in zip(fields, defaults, strict=False):
            v = getattr(self, k)
            if (d is msgspec.NODEFAULT or v != d) and v is not msgspec.UNSET:
                res[k] = v
        return res
