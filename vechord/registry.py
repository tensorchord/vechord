from collections import defaultdict
from datetime import datetime
from functools import wraps
from typing import Annotated, Any, Optional, Union, get_args, get_origin, get_type_hints
from uuid import UUID

import msgspec
import numpy as np

from vechord.client import VectorChordClient
from vechord.log import logger


class Vector:
    dim: int
    vec: Optional[np.ndarray] = None


class PrimaryKeyAutoIncrement:
    pass


TYPE_TO_PSQL = {
    int: "BIGINT",
    str: "TEXT",
    bytes: "BYTEA",
    float: "FLOAT 8",
    bool: "BOOLEAN",
    UUID: "UUID",
    datetime: "TIMESTAMP",
    Vector: "VECTOR",
    PrimaryKeyAutoIncrement: "BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY",
}


def is_optional_type(typ) -> bool:
    return get_origin(typ) is Union and type(None) in get_args(typ)


def type_to_psql(typ):
    if is_optional_type(typ):
        typ = get_args(typ)[0]

    if get_origin(typ) is Annotated:
        meta = typ.__metadata__
        origin = typ.__origin__
        if origin is Vector:
            assert len(meta) == 1, "only accept dim"
            return f"VECTOR({meta[0]})"
        raise ValueError(f"unsupported annotated type {typ}")
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


class Table(Storage):
    @classmethod
    def table_schema(cls) -> dict[str, str]:
        hints = get_type_hints(cls, include_extras=True)
        return ((name, type_to_psql(typ)) for name, typ in hints.items())

    @classmethod
    def vector_index(cls) -> Optional[str]:
        for name, typ in get_type_hints(cls, include_extras=True).items():
            if get_origin(typ) is Annotated and typ.__origin__ is Vector:
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
            if d is msgspec.NODEFAULT or v != d:
                res[k] = v
        return res


class Memory(Storage):
    def todict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.fields()}

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

    def run(self, funcs: list[RunFunction]):
        for func in funcs:
            func.func(*func.args) if func.args else func.func()

    def clear_storage(self):
        for table in self.tables:
            self.client.drop(table.name())
        self.memory.clear()
