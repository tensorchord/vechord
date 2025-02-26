from collections import defaultdict
from datetime import datetime
from functools import wraps
from typing import (
    Annotated,
    Any,
    Generic,
    Optional,
    Protocol,
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

from vechord.client import VectorChordClient
from vechord.log import logger


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


class PrimaryKeyAutoIncrease:
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
    datetime: "TIMESTAMP",
}


def is_optional_type(typ) -> bool:
    return get_origin(typ) is Union and type(None) in get_args(typ)


def type_to_psql(typ):
    if is_optional_type(typ):
        typ = get_args(typ)[0]

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
    def fields(cls) -> list[str]:
        return cls.__struct_fields__

    @classmethod
    def partial_init(cls, **kwargs):
        fields = cls.fields()
        args = dict(zip(fields, [msgspec.UNSET] * len(fields))) | kwargs
        return cls(**args)


class Table(Storage):
    @classmethod
    def table_schema(cls) -> dict[str, str]:
        hints = get_type_hints(cls, include_extras=True)
        return ((name, type_to_psql(typ)) for name, typ in hints.items())

    @classmethod
    def vector_index(cls) -> Optional[str]:
        for name, typ in get_type_hints(cls, include_extras=True).items():
            if issubclass(typ.__class__, VectorMeta):
                return name

    def todict(self) -> dict[str, Any]:
        defaults = getattr(self, "__struct_defaults__", None)
        fields = self.fields()
        if not defaults:
            return {k: getattr(self, k) for k in fields}
        # ignore default values
        res = {}
        for k, d in zip(fields, defaults):
            v = getattr(self, k)
            if (d is msgspec.NODEFAULT or v != d) and v is not msgspec.UNSET:
                res[k] = v
        return res


class Memory(Storage):
    def todict(self) -> dict[str, Any]:
        res = {}
        for field in self.fields():
            value = getattr(self, field)
            if value is msgspec.UNSET:
                continue
            res[field] = value
        return res

    def select(self, select_fields: list[str]):
        fields = self.fields()
        return (getattr(self, f) for f in select_fields if f in fields)


class RunFunction(msgspec.Struct):
    func: callable
    args: Optional[list] = []


class VechordRegistry:
    def __init__(self, namespace: str, url: str):
        self.ns = namespace
        self.client = VectorChordClient(namespace, url)
        self.tables: list[Table] = []
        self.memory: dict[str, list[Memory]] = defaultdict(list)

    def register(self, klasses: list[Storage]):
        for cls in klasses:
            if issubclass(cls, Table):
                self.client.create_table_if_not_exists(cls.name(), cls.table_schema())
                if vector_index := cls.vector_index():
                    self.client.create_vector_index(cls.name(), vector_index)
                    logger.debug(
                        "create vector index for %s.%s", cls.name(), vector_index
                    )
                self.tables.append(cls)
            elif issubclass(cls, Memory):
                self.memory[cls.name()] = cls
            else:
                raise ValueError(f"unsupported class {cls}")

    def load_from_storage(self, cls: type[Storage], select_fields: list[str]):
        if issubclass(cls, Table):
            return self.client.select(cls.name(), select_fields)
        elif issubclass(cls, Memory):
            objs = self.memory[cls.name()]
            return [obj.select(select_fields) for obj in objs]
        else:
            raise ValueError(f"unsupported class {cls}")

    def dump_to_storage(self, cls: type[Storage], obj: Storage):
        if not isinstance(obj, cls):
            raise ValueError(f"expected {cls}, got {type(obj)}")

        if issubclass(cls, Table):
            self.client.insert(cls.name(), obj.todict())
        elif issubclass(cls, Memory):
            self.memory[cls.name()].append(obj)
        else:
            raise ValueError(f"unsupported class {cls}")

    def select_by(
        self, cls: type[Storage], obj: Storage, fields: Optional[list[str]] = None
    ):
        if not isinstance(obj, cls):
            raise ValueError(f"expected {cls}, got {type(obj)}")

        cls_fields = cls.fields()
        if fields is not None:
            if any(f not in cls_fields for f in fields):
                raise ValueError(f"unknown fields {fields}")
        else:
            fields = cls_fields

        kvs = obj.todict()
        if issubclass(cls, Memory):
            objs = self.memory[cls.name()]
            if kvs:
                return [
                    obj
                    for obj in objs
                    if all(getattr(obj, key) == value for key, value in kvs.items())
                ]
            else:
                return objs
        elif issubclass(cls, Table):
            res = self.client.select(cls.name(), fields, kvs)
            missing = dict(zip(cls_fields, [msgspec.UNSET] * len(cls_fields)))
            return [cls(**(missing | {k: v for k, v in zip(fields, r)})) for r in res]
        else:
            raise ValueError(f"unsupported class {cls}")

    def search(
        self,
        cls: type[Storage],
        vec,
        topk: int = 10,
        return_vector: bool = False,
    ):
        if issubclass(cls, Table):
            fields = list(cls.fields())
            vec_col = cls.vector_index()
            addition = {}
            if not return_vector:
                fields.remove(vec_col)
                addition[vec_col] = None
            res = self.client.query_vec(
                cls.name(),
                vec_col,
                vec,
                topk=topk,
                return_fields=fields,
            )
            return [cls(**{k: v for k, v in zip(fields, r)}, **addition) for r in res]
        elif issubclass(cls, Memory):
            raise ValueError(f"query is not supported for {cls}")
        else:
            raise ValueError(f"unsupported class {cls}")

    def remove_by(self, cls: type[Storage], obj):
        if not isinstance(obj, cls):
            raise ValueError(f"expected {cls}, got {type(obj)}")

        kvs = obj.todict()
        if not kvs:
            raise ValueError("empty object")
        if issubclass(cls, Table):
            self.client.delete(cls.name(), kvs)
        elif issubclass(cls, Memory):
            lst = self.memory[cls.name()]
            index = []
            kvs = obj.todict()
            for i, item in enumerate(lst):
                if all(
                    getattr(item, key, None) == value for (key, value) in kvs.items()
                ):
                    index.append(i)
            for i in reversed(index):
                lst.pop(i)
        else:
            raise ValueError(f"unsupported class {cls}")

    def inject(self, input: Optional[Storage] = None, output: Optional[Storage] = None):
        if input is None and output is None:
            return lambda func: func

        def decorator(func: callable):
            hints = get_type_hints(func)
            returns = hints.pop("return", None)
            columns = hints.keys()
            return_type = (
                returns.__args__[0] if get_origin(returns) is list else returns
            )
            if return_type is not output:
                raise ValueError(f"expected {output}, got {return_type} in {func}")

            if output is not None and returns is None:
                raise ValueError(
                    f"requires the return type for {func} if `output` is set"
                )

            @wraps(func)
            def wrapper(*args, **kwargs):
                arguments = [args]
                if input is not None:
                    arguments = self.load_from_storage(input, columns)

                if output is None:
                    for arg in arguments:
                        func(*arg, **kwargs)

                count = 0
                if get_origin(returns) is list:
                    for arg in arguments:
                        for ret in func(*arg):
                            self.dump_to_storage(output, ret)
                            count += 1
                else:
                    for arg in arguments:
                        ret = func(*arg)
                        self.dump_to_storage(output, ret)
                        count += 1
                logger.debug("inserted %d items to %s", count, output.name())

            return wrapper

        return decorator

    def clear_storage(self):
        for table in self.tables:
            self.client.drop(table.name())
        self.memory.clear()
