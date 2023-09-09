# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import random
from typing import Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.testing._internal.common_dtype as common_dtype
from executorch.exir.dialects.edge.arg.model import (
    ArgMode,
    BaseArg,
    BaseKwarg,
    GenMode,
    get_callable,
)
from executorch.exir.dialects.edge.arg.type import ArgType


class DtypeRunner:
    def __init__(self):
        self.tensor_dtypes = list(common_dtype.all_types_and(torch.bool))
        self.scalar_dtypes = [torch.bool, torch.int, torch.float]

    @staticmethod
    def _get_types(inputs: Dict[str, List[BaseArg]]) -> List[ArgType]:
        """Given inputs, return a list of argument types."""
        return [arg.type for arg in inputs["args"] if arg.type.has_dtype()]

    @staticmethod
    def _get_args_kwargs(
        inputs: Dict[str, List[Union[BaseArg]]],
        dtypes: Tuple[torch.dtype],
        mode: ArgMode,
    ) -> Tuple[List[BaseArg], Dict[str, BaseKwarg]]:
        """Construct args and kwargs for op given dtypes."""
        args = []
        kwargs = {}
        counter = 0
        for arg in inputs["args"]:
            arg.mode = mode
            val = arg.get_val()
            if arg.type.has_dtype():
                val = arg.get_val_with_dtype(dtypes[counter])
                counter += 1
            if arg.kw and isinstance(arg, BaseKwarg):
                kwargs[arg.argname] = val
            else:
                args.append(val)
        return args, kwargs

    @staticmethod
    def _produce_dtype_tuple(
        types: List[ArgType], code_tuple: Tuple[int], ty: ArgType, dt: torch.dtype
    ) -> Optional[Tuple[torch.dtype]]:
        dtype_tuple = []
        for i, code in enumerate(code_tuple):
            same_group = [dt]
            if ty.is_scalar() and types[i].is_tensor():
                if dt == torch.bool or dt == torch.float:
                    same_group = list(common_dtype.floating_types())
                elif dt == torch.int:
                    same_group = list(common_dtype.integral_types())
                else:
                    same_group = [None]
            elif ty.is_tensor() and types[i].is_scalar():
                if dt == torch.bool:
                    same_group = [torch.bool]
                elif dt in common_dtype.integral_types():
                    same_group = [torch.int]
                elif dt in common_dtype.floating_types():
                    same_group = [torch.float]
                else:
                    same_group = [None]

            if code == 0:
                if dt is None and not types[i].is_optional():
                    return
                dtype_tuple.append(random.choice(same_group))
            else:
                all_types = common_dtype.all_types_and(torch.bool)
                diff_group = list(set(all_types) - set(same_group))
                dtype_tuple.append(random.choice(diff_group))
        return tuple(dtype_tuple)

    def _get_type_tuples(
        self, inputs: Dict[str, List[BaseArg]]
    ) -> List[List[torch.dtype]]:
        types = DtypeRunner._get_types(inputs)

        def mapping(t):
            if t.is_optional():
                return [None]
            elif t.is_scalar():
                return self.scalar_dtypes
            elif t.is_scalar_type() or t.is_tensor() or t.is_tensor_list():
                return self.tensor_dtypes
            else:
                raise ValueError("Type {t.name} does not have dtype")

        return list(map(mapping, types))

    def select_dtype_combinations(
        self, inputs: Dict[str, List[BaseArg]], genmode: GenMode
    ) -> Iterator[Tuple[torch.dtype]]:
        random.seed(0)

        def produce_code_tuples(n: int, i: int) -> Iterator[Tuple[int]]:
            codes = [(0,) if j == i else (0, 1) for j in range(n)]
            return itertools.product(*codes)

        type_tuples = self._get_type_tuples(inputs)
        if genmode == GenMode.All:
            return itertools.product(*type_tuples)  # noqa
        elif genmode == GenMode.Partial:
            dtype_tuples_set = set()
            types = DtypeRunner._get_types(inputs)
            n = len(types)
            for i in range(n):
                for dt in type_tuples[i]:
                    for code_tuple in produce_code_tuples(n, i):
                        dtype_tuple = DtypeRunner._produce_dtype_tuple(
                            types, code_tuple, types[i], dt
                        )
                        if (
                            dtype_tuple is not None
                            and dtype_tuple not in dtype_tuples_set
                        ):
                            yield dtype_tuple
                            dtype_tuples_set.add(dtype_tuple)

    def run_dtypes(
        self,
        name: str,
        inputs: Dict[str, List[BaseArg]],
        dtypes: Tuple[torch.dtype],
        argmode: ArgMode = ArgMode.RANDOM,
    ) -> Tuple[bool, str, Tuple[torch.dtype], List[BaseArg], Dict[str, BaseKwarg]]:
        args, kwargs = DtypeRunner._get_args_kwargs(inputs, dtypes, argmode)
        op = get_callable(name)
        try:
            op(*args, **kwargs)
            return (True, name, dtypes, args, kwargs)
        except Exception as e:
            if argmode == ArgMode.ONES:
                return (False, name, dtypes, args, kwargs)
            ones_args, ones_kwargs = DtypeRunner._get_args_kwargs(
                inputs, dtypes, ArgMode.ONES
            )
            try:
                op(*ones_args, **ones_kwargs)
                print(e)
                print(name, dtypes, args, kwargs)
                return (True, name, dtypes, ones_args, ones_kwargs)
            except Exception:
                return (False, name, dtypes, ones_args, ones_kwargs)

    def run(
        self, name: str, inputs: Dict[str, List[BaseArg]]
    ) -> List[
        Tuple[bool, str, Tuple[torch.dtype], List[BaseArg], Dict[str, BaseKwarg]]
    ]:
        results = []
        type_tuples = self._get_type_tuples(inputs)
        for element in itertools.product(*type_tuples):
            results.append(self.run_dtypes(name, inputs, element))
        return results
