# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.testing._internal.common_dtype as common_dtype
from executorch.exir.dialects.edge.arg.model import ArgMode, BaseArg, BaseKwarg
from executorch.exir.dialects.edge.arg.type import ArgType
from executorch.exir.dialects.edge.dtype.utils import extract_return_dtype
from executorch.exir.dialects.edge.op.api import get_callable


class DtypeRunner:
    def __init__(self):
        self.tensor_dtypes = list(common_dtype.all_types_and(torch.bool, torch.half))
        self.scalar_dtypes = [torch.bool, torch.int, torch.float]

    @staticmethod
    def _get_types(inputs: Dict[str, List[BaseArg]]) -> List[ArgType]:
        """Given inputs, return a list of argument types."""
        return [arg.type for arg in inputs["args"] if arg.type.has_dtype()]

    @staticmethod
    def _get_args_kwargs(
        inputs: Dict[str, List[BaseArg]],
        dtypes: Tuple[Optional[torch.dtype]],
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

    def _get_type_tuples(
        self, inputs: Dict[str, List[BaseArg]]
    ) -> List[List[Optional[torch.dtype]]]:
        types = DtypeRunner._get_types(inputs)

        def mapping(t):
            type_dtypes = []
            if t.is_optional():
                type_dtypes = [None]
            if t.is_scalar():
                return type_dtypes + self.scalar_dtypes
            elif t.is_scalar_type() or t.is_tensor() or t.is_tensor_list():
                return type_dtypes + self.tensor_dtypes
            else:
                raise ValueError("Type {t.name} does not have dtype")

        return list(map(mapping, types))

    def run_dtypes(
        self,
        name: str,
        inputs: Dict[str, List[BaseArg]],
        dtypes: Tuple[Optional[torch.dtype]],
        argmode: ArgMode = ArgMode.RANDOM,
    ) -> Tuple[
        bool, str, Tuple[Optional[torch.dtype]], List[BaseArg], Dict[str, BaseKwarg]
    ]:
        args, kwargs = DtypeRunner._get_args_kwargs(inputs, dtypes, argmode)
        op = get_callable(name)
        try:
            res = op(*args, **kwargs)
            ret_dtypes = ()
            if "returns" in inputs:
                ret_dtypes = tuple(extract_return_dtype(res, inputs["returns"]))
            return (True, name, dtypes + ret_dtypes, args, kwargs)
        except AssertionError as e:
            raise RuntimeError(
                f"opname: {name}, inputs: {inputs}, dtypes: {dtypes}, argmode {argmode}"
            ) from e
        except Exception as e:
            if argmode == ArgMode.ONES:
                return (False, name, dtypes, args, kwargs)
            ones_args, ones_kwargs = DtypeRunner._get_args_kwargs(
                inputs, dtypes, ArgMode.ONES
            )
            try:
                res = op(*args, **kwargs)
                ret_dtypes = ()
                if "returns" in inputs:
                    ret_dtypes = tuple(extract_return_dtype(res, inputs["returns"]))
                print(e)
                print(name, dtypes, args, kwargs)
                return (True, name, dtypes + ret_dtypes, ones_args, ones_kwargs)
            except Exception:
                return (False, name, dtypes, ones_args, ones_kwargs)

    def run(
        self,
        name: str,
        inputs: Dict[str, Any],
        argmode: ArgMode = ArgMode.ONES,
    ) -> List[
        Tuple[
            bool, str, Tuple[Optional[torch.dtype]], List[BaseArg], Dict[str, BaseKwarg]
        ]
    ]:
        results = []
        type_tuples = self._get_type_tuples(inputs)
        for element in itertools.product(*type_tuples):
            results.append(self.run_dtypes(name, inputs, element, argmode))
        return results
