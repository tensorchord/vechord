from collections.abc import AsyncIterable, Iterable
from contextlib import AsyncExitStack
from functools import wraps
from inspect import isasyncgenfunction, iscoroutinefunction
from typing import (
    Callable,
    Optional,
    Sequence,
    TypeVar,
    get_origin,
    get_type_hints,
)

import numpy as np

from vechord.client import VechordClient, select_transaction_buffer_conn
from vechord.log import logger
from vechord.pipeline import VechordPipeline
from vechord.spec import Table, Vector

T = TypeVar("T", bound=Table)


def is_list_of_type(typ) -> bool:
    origin = get_origin(typ)
    if origin is None:
        return False
    return issubclass(origin, (Iterable, AsyncIterable))


def get_iterator_type(typ) -> type:
    if not is_list_of_type(typ):
        return typ
    return get_iterator_type(typ.__args__[0])


class VechordRegistry:
    """Create a registry for the given namespace and PostgreSQL URL.

    Args:
        namespace: the namespace for this registry, will be the prefix for all the
            tables registered.
        url: the PostgreSQL URL to connect to.
        tables: a list of Table classes to be registered.
        create_index: whether or not to create the index if not exists.
    """

    def __init__(
        self,
        namespace: str,
        url: str,
        *,
        tables: Iterable[type[Table]] = (),
        create_index: bool = True,
    ):
        self.ns = namespace
        self.client = VechordClient(namespace, url)
        self.tables: list[type[Table]] = []
        self.create_index = create_index
        self.pipeline: list[Callable] = []
        self._inited = False

        for table in tables:
            if not issubclass(table, Table):
                raise ValueError(f"unsupported class {table}")
            self.tables.append(table)

    async def __aenter__(self):
        if not self._inited:
            self._async_exit_stack = AsyncExitStack()
            await self._async_exit_stack.enter_async_context(self.client)
            await self.init_table_index()
            self._inited = True
        return self

    async def __aexit__(self, _exc_type, _exc_value, _traceback):
        await self._async_exit_stack.aclose()

    async def process_startup(self, scope, event):
        """Falcon ASGI middleware lifespan hook."""
        await self.__aenter__()

    async def process_shutdown(self, scope, event):
        """Falcon ASGI middleware lifespan hook."""
        await self.__aexit__(None, None, None)

    def reset_namespace(self, namespace: str):
        self.ns = namespace
        self.client.ns = namespace

    async def init_table_index(self, tables: Optional[Iterable[type[Table]]] = None):
        if tables is None:
            tables = self.tables

        for table in tables:
            await self.client.create_table_if_not_exists(
                table.name(), table.table_schema()
            )
            logger.debug("create table %s if not exists", table.name())

            if table.keyword_column() is not None:
                await self.client.create_tokenizer()

            if not self.create_index:
                continue

            # create index
            for index_column in (
                table.vector_column(),
                table.multivec_column(),
                table.keyword_column(),
                *table.unique_columns(),
            ):
                if index_column is None:
                    continue
                await self.client.create_index_if_not_exists(table.name(), index_column)
                logger.debug(
                    "create index for %s.%s if not exists",
                    table.name(),
                    index_column.name,
                )

    def create_pipeline(self, steps: list[Callable]) -> VechordPipeline:
        """Create the :class:`VechordPipeline` to run multiple functions in a transaction.

        Args:
            steps: a list of functions to be run in the pipeline.
        """
        return VechordPipeline(client=self.client, steps=steps)

    async def select_by(
        self,
        obj: T,
        fields: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> list[T]:
        """Retrieve the requested fields for the given object stored in the DB.

        Args:
            obj: the object to be retrieved, this should be generated from
                :meth:`Table.partial_init`, while the given values will be used
                for filtering (``=`` or ``is``).
            fields: the fields to be retrieved, if not set, all the fields will be
                retrieved.
            limit: the maximum number of results to be returned, if not set, all
                the results will be returned.
        """
        if not isinstance(obj, Table):
            raise ValueError(f"unsupported class {type(obj)}")

        cls_fields = obj.fields()
        cls = obj.__class__
        if fields is not None:
            if any(f not in cls_fields for f in fields):
                raise ValueError(f"unknown fields {fields}")
        else:
            fields = cls_fields

        kvs = obj.todict()
        res = await self.client.select(cls.name(), fields, kvs, limit=limit)
        return [
            cls.partial_init(**{k: v for k, v in zip(fields, r, strict=False)})
            for r in res
        ]

    async def search_by_vector(
        self,
        cls: type[T],
        vec: np.ndarray | Vector,
        topk: int = 10,
        return_fields: Optional[Sequence[str]] = None,
        probe: Optional[int] = None,
    ) -> list[T]:
        """Search the vector for the given `Table` class.

        Args:
            cls: the `Table` class to be searched.
            vec: the vector to be searched.
            topk: the number of results to be returned.
            return_fields: the fields to be returned, if not set, all the
                non-[vector,keyword] fields will be returned.
            probe: how many K-means clusters to probe for the `vec`.
        """
        if not issubclass(cls, Table):
            raise ValueError(f"unsupported class {cls}")
        fields = return_fields or cls.non_vec_columns()
        vec_col = cls.vector_column()
        if vec_col is None:
            raise ValueError(f"no vector column found in {cls}")
        res = await self.client.query_vec(
            cls.name(),
            vec_col,
            vec,
            topk=topk,
            return_fields=fields,
            probe=probe,
        )
        return [
            cls.partial_init(**{k: v for k, v in zip(fields, r, strict=False)})
            for r in res
        ]

    async def search_by_multivec(  # noqa: PLR0913
        self,
        cls: type[T],
        multivec: np.ndarray,
        topk: int = 10,
        return_fields: Optional[Sequence[str]] = None,
        maxsim_refine: int = 1000,
        probe: Optional[int] = None,
    ) -> list[T]:
        """Search the multivec for the given `Table` class.

        Args:
            cls: the `Table` class to be searched.
            multivec: the multivec to be searched.
            topk: the number of results to be returned.
            maxsim_refine: the maximum number of document vectors to be compute with
                full-precision for each vector in the `multivec`. 0 means all the
                distances are compute with bit quantization.
            probe: how many K-means clusters to probe for each vector in the `multivec`.
            return_fields: the fields to be returned, if not set, all the
                non-[vector,keyword] fields will be returned.
        """
        if not issubclass(cls, Table):
            raise ValueError(f"unsupported class {cls}")
        fields = return_fields or cls.non_vec_columns()
        multivec_col = cls.multivec_column()
        if multivec_col is None:
            raise ValueError(f"no multivec column found in {cls}")
        res = await self.client.query_multivec(
            cls.name(),
            multivec_col,
            multivec,
            maxsim_refine=maxsim_refine,
            probe=probe,
            topk=topk,
            return_fields=fields,
        )
        return [
            cls.partial_init(**{k: v for k, v in zip(fields, r, strict=False)})
            for r in res
        ]

    async def search_by_keyword(
        self,
        cls: type[T],
        keyword: str,
        topk: int = 10,
        return_fields: Optional[Sequence[str]] = None,
    ) -> list[T]:
        """Search the keyword for the given `Table` class.

        Args:
            cls: the `Table` class to be searched.
            keyword: the keyword to be searched.
            topk: the number of results to be returned.
            return_fields: the fields to be returned, if not set, all the
                non-[vector,keyword] fields will be returned.
        """
        if not issubclass(cls, Table):
            raise ValueError(f"unsupported class {cls}")
        fields = return_fields or cls.non_vec_columns()
        keyword_col = cls.keyword_column()
        if keyword_col is None:
            raise ValueError(f"no keyword column found in {cls}")
        res = await self.client.query_keyword(
            cls.name(),
            keyword_col,
            keyword,
            topk=topk,
            return_fields=fields,
            tokenizer=cls.keyword_tokenizer(),
        )
        return [
            cls.partial_init(**{k: v for k, v in zip(fields, r, strict=False)})
            for r in res
        ]

    async def remove_by(self, obj: Table):
        """Remove the given object from the DB.

        Args:
            obj: the object to be removed, this should be a `Table.partial_init()`
                instance, which means given values will be used for filtering.
        """
        if not isinstance(obj, Table):
            raise ValueError(f"unsupported class {type(obj)}")

        kvs = obj.todict()
        await self.client.delete(obj.__class__.name(), kvs)

    async def insert(self, obj: Table):
        """Insert the given object to the DB.

        Args:
            obj: the object to be inserted
        """
        if not isinstance(obj, Table):
            raise ValueError(f"unsupported class {type(obj)}")
        await self.client.insert(obj.name(), obj.todict())

    async def copy_bulk(self, objs: list[Table]):
        """Insert the given list of objects to the DB.

        This is more efficient than calling `insert` for each object.

        Args:
            objs: the list of objects to be inserted, needs to be of the same
                class and filled with the same fields. The class should be a
                subclass of `Table`.
        """
        if not objs:
            return
        cls = objs[0].__class__
        if not issubclass(cls, Table):
            raise ValueError(f"unsupported class {cls}")
        if not all(isinstance(obj, cls) for obj in objs):
            raise ValueError(f"not all the objects are {cls}")
        if cls.keyword_column() is not None:
            raise RuntimeError("`copy_bulk` does not support keyword column")

        name = objs[0].name()
        values = [obj.todict() for obj in objs]
        keys = set(values[0].keys())
        types = tuple(v for k, v in objs[0].table_psql_types() if k in keys)
        await self.client.copy_bulk(name, values, types)

    def inject(
        self, input: Optional[type[Table]] = None, output: Optional[type[Table]] = None
    ):
        """Decorator to inject the data for the function arguments & return value.

        This can be applied to both sync & async functions. But the decorated
        functions will all be async functions.

        Args:
            input: the input table to be retrieved from the DB. If not set, the function
                will require the input to be passed in the function call.
            output: the output table to store the return value. If not set, the return
                value will be return to the caller in a `list`.
        """
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
            return_type = get_iterator_type(returns)
            if output and return_type is not output:
                raise ValueError(
                    f"expected {output}, got {return_type} in {func} output"
                )

            if output is not None and returns is None:
                raise ValueError(
                    f"requires the return type for {func} if `output` is set"
                )

            _is_async = iscoroutinefunction(func)
            _is_async_gen = isasyncgenfunction(func)

            async def execute_func(*args, **kwargs):
                if _is_async:
                    return await func(*args, **kwargs)
                elif _is_async_gen:
                    # has to collect all items in an async generator
                    items = []
                    async for item in func(*args, **kwargs):
                        items.append(item)
                    return items
                return func(*args, **kwargs)

            @wraps(func)
            async def wrapper(*args, **kwargs):
                arguments = [args]
                if input is not None:
                    arguments = await self.client.select(
                        input.name(),
                        columns,
                        from_buffer=select_transaction_buffer_conn.get() is not None,
                    )

                if output is None:
                    return [await execute_func(*arg, **kwargs) for arg in arguments]

                count = 0
                use_copy = output.keyword_column() is None
                if is_list_of_type(returns):
                    for arg in arguments:
                        if use_copy:
                            rets = list(await execute_func(*arg, **kwargs))
                            await self.copy_bulk(rets)
                            count += len(rets)
                        else:
                            for ret in await execute_func(*arg, **kwargs):
                                await self.insert(ret)
                                count += 1
                elif use_copy:
                    # ref https://github.com/python/cpython/issues/76294
                    rets = [await execute_func(*arg, **kwargs) for arg in arguments]
                    await self.copy_bulk(rets)
                    count += len(rets)
                else:
                    for arg in arguments:
                        ret = await execute_func(*arg, **kwargs)
                        await self.insert(ret)
                        count += 1
                logger.debug("inserted %d items to %s", count, output.name())

            return wrapper

        return decorator

    async def clear_storage(self, drop_table: bool = False):
        """Clear the storage of the registry.

        Args:
            drop_table: whether to drop the table after removing all the data.
        """
        for table in self.tables:
            if drop_table:
                await self.client.drop(table.name())
            else:
                await self.remove_by(table.partial_init())
