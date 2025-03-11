from functools import wraps
from typing import (
    Callable,
    Generator,
    Iterator,
    Optional,
    get_origin,
    get_type_hints,
)

import msgspec

from vechord.client import (
    VectorChordClient,
    limit_to_transaction_buffer,
    select_transaction_buffer,
)
from vechord.log import logger
from vechord.spec import Table


def is_list_of_type(typ) -> bool:
    origin = get_origin(typ)
    if origin is None:
        return False
    if origin is list:
        return True
    return issubclass(origin, Iterator) or issubclass(origin, Generator)


class VechordRegistry:
    def __init__(self, namespace: str, url: str):
        self.ns = namespace
        self.client = VectorChordClient(namespace, url)
        self.tables: list[type[Table]] = []
        self.pipeline: list[Callable] = []

    def register(self, tables: list[type[Table]]):
        for table in tables:
            if not issubclass(table, Table):
                raise ValueError(f"unsupported class {table}")

            self.client.create_table_if_not_exists(table.name(), table.table_schema())
            logger.debug("create table %s if not exists", table.name())
            if vector_column := table.vector_column():
                self.client.create_vector_index(table.name(), vector_column)
                logger.debug(
                    "create vector index for %s.%s if not exists",
                    table.name(),
                    vector_column,
                )
            self.tables.append(table)

    def set_pipeline(self, pipeline: list[Callable]):
        self.pipeline = pipeline

    def run(self, *args, **kwargs):
        """Execute the pipeline in a transactional manner."""
        if not self.pipeline:
            raise RuntimeError("pipeline is not set")
        with self.client.transaction(), limit_to_transaction_buffer():
            # only the 1st one can accept input (could be empty)
            self.pipeline[0](*args, **kwargs)
            for func in self.pipeline[1:]:
                func()

    def select_by(
        self, cls: type[Table], obj: Table, fields: Optional[list[str]] = None
    ):
        if not isinstance(obj, cls):
            raise ValueError(f"expected {cls}, got {type(obj)}")
        if not issubclass(cls, Table):
            raise ValueError(f"unsupported class {cls}")

        cls_fields = cls.fields()
        if fields is not None:
            if any(f not in cls_fields for f in fields):
                raise ValueError(f"unknown fields {fields}")
        else:
            fields = cls_fields

        kvs = obj.todict()
        res = self.client.select(cls.name(), fields, kvs)
        missing = dict(zip(cls_fields, [msgspec.UNSET] * len(cls_fields), strict=False))
        return [
            cls(**(missing | {k: v for k, v in zip(fields, r, strict=False)}))
            for r in res
        ]

    def search(
        self,
        cls: type[Table],
        vec,
        topk: int = 10,
        return_vector: bool = False,
    ):
        if not issubclass(cls, Table):
            raise ValueError(f"unsupported class {cls}")
        fields = list(cls.fields())
        vec_col = cls.vector_column()
        if vec_col is None:
            raise ValueError(f"no vector column found in {cls}")
        if not return_vector:
            fields.remove(vec_col)
        res = self.client.query_vec(
            cls.name(),
            vec_col,
            vec,
            topk=topk,
            return_fields=fields,
        )
        return [
            cls.partial_init(**{k: v for k, v in zip(fields, r, strict=False)})
            for r in res
        ]

    def remove_by(self, cls: type[Table], obj):
        if not isinstance(obj, cls):
            raise ValueError(f"expected {cls}, got {type(obj)}")
        if not issubclass(cls, Table):
            raise ValueError(f"unsupported class {cls}")

        kvs = obj.todict()
        if not kvs:
            raise ValueError("empty object")
        self.client.delete(cls.name(), kvs)

    def insert(self, obj):
        if not isinstance(obj, Table):
            raise ValueError(f"unsupported class {type(obj)}")
        self.client.insert(obj.name(), obj.todict())

    def inject(
        self, input: Optional[type[Table]] = None, output: Optional[type[Table]] = None
    ):
        if input is None and output is None:
            return lambda func: func
        if input is not None and not issubclass(input, Table):
            raise ValueError(f"unsupported class {input}")
        if output is not None and not issubclass(output, Table):
            raise ValueError(f"unsupported class {output}")

        def decorator(func: Callable):
            hints = get_type_hints(func)
            returns = hints.pop("return", None)
            columns = hints.keys()
            return_type = returns.__args__[0] if is_list_of_type(returns) else returns
            if output and return_type is not output:
                raise ValueError(
                    f"expected {output}, got {return_type} in {func} output"
                )

            if output is not None and returns is None:
                raise ValueError(
                    f"requires the return type for {func} if `output` is set"
                )

            @wraps(func)
            def wrapper(*args, **kwargs):
                arguments = [args]
                if input is not None:
                    arguments = self.client.select(
                        input.name(),
                        columns,
                        from_buffer=select_transaction_buffer.get(),
                    )

                if output is None:
                    return [func(*arg, **kwargs) for arg in arguments]

                count = 0
                if is_list_of_type(returns):
                    for arg in arguments:
                        for ret in func(*arg, **kwargs):
                            self.insert(ret)
                            count += 1
                else:
                    for arg in arguments:
                        ret = func(*arg, **kwargs)
                        self.insert(ret)
                        count += 1
                logger.debug("inserted %d items to %s", count, output.name())

            return wrapper

        return decorator

    def clear_storage(self, drop_table: bool = False):
        for table in self.tables:
            if drop_table:
                self.client.drop(table.name())
            else:
                self.remove_by(table, table.partial_init())
