from functools import wraps
from typing import (
    Callable,
    Generator,
    Iterator,
    Optional,
    Sequence,
    get_origin,
    get_type_hints,
)

import numpy as np

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
    """Create a registry for the given namespace and PostgreSQL URL.

    Args:
        namespace: the namespace for this registry, will be the prefix for all the
            tables registered.
        url: the PostgreSQL URL to connect to.
    """

    def __init__(self, namespace: str, url: str):
        self.ns = namespace
        self.client = VectorChordClient(namespace, url)
        self.tables: list[type[Table]] = []
        self.pipeline: list[Callable] = []

    def register(self, tables: list[type[Table]]):
        """Register the given tables to the registry.

        This will create the tables in the database if not exists.

        Args:
            tables: a list of Table classes to be registered.
        """
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
            if multivec_column := table.multivec_column():
                self.client.create_multivec_index(table.name(), multivec_column)
                logger.debug(
                    "create multivec index for %s.%s if not exists",
                    table.name(),
                    multivec_column,
                )
            if keyword_column := table.keyword_column():
                self.client.create_keyword_index(table.name(), keyword_column)
                logger.debug(
                    "create keyword index for %s.%s if not exists",
                    table.name(),
                    keyword_column,
                )
            self.tables.append(table)

    def set_pipeline(self, pipeline: list[Callable]):
        """Set the pipeline to be executed in the `run` method."""
        self.pipeline = pipeline

    def run(self, *args, **kwargs):
        """Execute the pipeline in a transactional manner.

        All the `args` and `kwargs` will be passed to the first function in the
        pipeline. The pipeline will run in *one* transaction, and all the `inject`
        can only see the data inserted in this transaction (to guarantee only the
        new inserted data will be processed in this pipeline).

        This will also return the final result of the last function in the pipeline.
        """
        if not self.pipeline:
            raise RuntimeError("pipeline is not set")
        with self.client.transaction(), limit_to_transaction_buffer():
            # only the 1st one can accept input (could be empty)
            self.pipeline[0](*args, **kwargs)
            for func in self.pipeline[1:-1]:
                func()
            return self.pipeline[-1]()

    def select_by(
        self,
        obj: Table,
        fields: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> list[Table]:
        """Retrieve the requested fields for the given object stored in the DB.

        Args:
            obj: the object to be retrieved, this should be a `Table.partial_init()`
                instance, which means given values will be used for filtering.
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
        res = self.client.select(cls.name(), fields, kvs, limit=limit)
        return [
            cls.partial_init(**{k: v for k, v in zip(fields, r, strict=False)})
            for r in res
        ]

    def search_by_vector(
        self,
        cls: type[Table],
        vec: np.ndarray,
        topk: int = 10,
        return_fields: Optional[Sequence[str]] = None,
    ) -> list[Table]:
        """Search the vector for the given `Table` class.

        Args:
            cls: the `Table` class to be searched.
            vec: the vector to be searched.
            topk: the number of results to be returned.
            return_fields: the fields to be returned, if not set, all the
                non-[vector,keyword] fields will be returned.
        """
        if not issubclass(cls, Table):
            raise ValueError(f"unsupported class {cls}")
        fields = cls.non_vec_columns() if return_fields is None else return_fields
        vec_col = cls.vector_column()
        if vec_col is None:
            raise ValueError(f"no vector column found in {cls}")
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

    def search_by_multivec(
        self,
        cls: type[Table],
        multivec: np.ndarray,
        topk: int = 10,
        return_fields: Optional[Sequence[str]] = None,
        max_maxsim_tuples: int = 1000,
    ) -> list[Table]:
        """Search the multivec for the given `Table` class.

        Args:
            cls: the `Table` class to be searched.
            multivec: the multivec to be searched.
            topk: the number of results to be returned.
            max_maxsim_tuples: the maximum number of tuples to be considered for
                the each vector in the multivec.
            return_fields: the fields to be returned, if not set, all the
                non-[vector,keyword] fields will be returned.
        """
        if not issubclass(cls, Table):
            raise ValueError(f"unsupported class {cls}")
        fields = cls.non_vec_columns() if return_fields is None else return_fields
        multivec_col = cls.multivec_column()
        if multivec_col is None:
            raise ValueError(f"no multivec column found in {cls}")
        res = self.client.query_multivec(
            cls.name(),
            multivec_col,
            multivec,
            max_maxsim_tuples=max_maxsim_tuples,
            topk=topk,
            return_fields=fields,
        )
        return [
            cls.partial_init(**{k: v for k, v in zip(fields, r, strict=False)})
            for r in res
        ]

    def search_by_keyword(
        self,
        cls: type[Table],
        keyword: str,
        topk: int = 10,
        return_fields: Optional[Sequence[str]] = None,
    ) -> list[Table]:
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
        fields = cls.non_vec_columns() if return_fields is None else return_fields
        keyword_col = cls.keyword_column()
        if keyword_col is None:
            raise ValueError(f"no keyword column found in {cls}")
        res = self.client.query_keyword(
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

    def remove_by(self, obj: Table):
        """Remove the given object from the DB.

        Args:
            obj: the object to be removed, this should be a `Table.partial_init()`
                instance, which means given values will be used for filtering.
        """
        if not isinstance(obj, Table):
            raise ValueError(f"unsupported class {type(obj)}")

        kvs = obj.todict()
        self.client.delete(obj.__class__.name(), kvs)

    def insert(self, obj: Table):
        """Insert the given object to the DB."""
        if not isinstance(obj, Table):
            raise ValueError(f"unsupported class {type(obj)}")
        self.client.insert(obj.name(), obj.todict())

    def inject(
        self, input: Optional[type[Table]] = None, output: Optional[type[Table]] = None
    ):
        """Decorator to inject the data for the function arguments & return value.

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
        """Clear the storage of the registry.

        Args:
            drop_table: whether to drop the table after removing all the data.
        """
        for table in self.tables:
            if drop_table:
                self.client.drop(table.name())
            else:
                self.remove_by(table.partial_init())
