# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
from functools import lru_cache
from typing import List, OrderedDict, Tuple

import torch
from facto.inputgen.argtuple.gen import ArgumentTupleGenerator
from facto.inputgen.specs.model import ConstraintProducer as cp
from facto.inputgen.variable.type import ScalarDtype
from facto.specdb.db import SpecDictDB

# seed to generate identical cases every run to reproduce from bisect
MAX_CASES = 50


def apply_tensor_contraints(op_name: str, index: int) -> list[object]:
    tensor_constraints = [
        cp.Dtype.In(lambda deps: [torch.int, torch.float]),
        cp.Dtype.NotIn(lambda deps: [torch.int64, torch.float64]),
        cp.Value.Ge(lambda deps, dtype, struct: -(2**4)),
        cp.Value.Le(lambda deps, dtype, struct: 2**4),
        cp.Rank.Ge(lambda deps: 1),
        cp.Size.Ge(lambda deps, r, d: 1),
        cp.Size.Le(lambda deps, r, d: 2**9),
    ]

    match op_name:
        case "where.self":
            if index == 0:  # condition
                tensor_constraints = [
                    cp.Dtype.In(lambda deps: [torch.bool]),
                    cp.Dtype.NotIn(lambda deps: [torch.int64, torch.float64]),
                    cp.Value.Ge(lambda deps, dtype, struct: -(2**4)),
                    cp.Value.Le(lambda deps, dtype, struct: 2**4),
                    cp.Rank.Ge(lambda deps: 1),
                    cp.Size.Ge(lambda deps, r, d: 1),
                    cp.Size.Le(lambda deps, r, d: 2**9),
                ]
            else:
                tensor_constraints = [
                    cp.Dtype.In(lambda deps: [torch.float, torch.int]),
                    cp.Dtype.NotIn(lambda deps: [torch.int64, torch.float64]),
                    cp.Value.Ge(lambda deps, dtype, struct: -(2**4)),
                    cp.Value.Le(lambda deps, dtype, struct: 2**4),
                    cp.Rank.Ge(lambda deps: 1),
                    cp.Size.Ge(lambda deps, r, d: 1),
                    cp.Size.Le(lambda deps, r, d: 2**9),
                ]
        case "sigmoid.default":
            tensor_constraints.extend(
                [
                    cp.Dtype.In(lambda deps: [torch.float]),
                    cp.Rank.Le(lambda deps: 2**2),
                    cp.Value.Ge(lambda deps, dtype, struct: -2),
                    cp.Value.Le(lambda deps, dtype, struct: 2),
                ]
            )
        case "rsqrt.default":
            tensor_constraints.extend(
                [
                    cp.Dtype.In(lambda deps: [torch.float]),
                    cp.Rank.Le(lambda deps: 2**2),
                    cp.Value.Gt(
                        lambda deps, dtype, struct: 0
                    ),  # only generate real numbers
                    cp.Value.Le(lambda deps, dtype, struct: 2**2),
                ]
            )
        case "mean.dim":
            tensor_constraints.extend(
                [
                    cp.Dtype.In(lambda deps: [torch.float]),
                    cp.Rank.Le(lambda deps: 2**2),
                ]
            )
        case "exp.default":
            tensor_constraints.extend(
                [
                    cp.Rank.Le(lambda deps: 2**3),
                    cp.Value.Ge(lambda deps, dtype, struct: -(2**2)),
                    cp.Value.Le(lambda deps, dtype, struct: 2**2),
                ]
            )
        case "slice_copy.Tensor":
            tensor_constraints.extend(
                [
                    cp.Rank.Le(lambda deps: 2),
                    cp.Value.Ge(lambda deps, dtype, struct: 1),
                    cp.Value.Le(lambda deps, dtype, struct: 2),
                ]
            )
        case _:
            tensor_constraints.extend(
                [
                    cp.Rank.Le(lambda deps: 2**2),
                ]
            )
    return tensor_constraints


def apply_scalar_contraints(op_name: str) -> list[ScalarDtype]:
    match op_name:
        case "add.Scalar" | "sub.Scalar" | "mul.Scalar" | "div.Scalar":
            return [ScalarDtype.int]

        case _:
            return [ScalarDtype.float, ScalarDtype.int]


@lru_cache(maxsize=None)
def facto_testcase_gen(op_name: str) -> List[Tuple[List[str], OrderedDict[str, str]]]:
    # minimal example to test add.Tensor using FACTO
    spec = SpecDictDB[op_name]

    for index, in_spec in enumerate(copy.deepcopy(spec.inspec)):
        if in_spec.type.is_scalar():
            if in_spec.name != "alpha":
                spec.inspec[index].constraints.extend(
                    [
                        cp.Dtype.In(lambda deps: apply_scalar_contraints(op_name)),
                        cp.Value.Ge(lambda deps, dtype: -(2**8)),
                        cp.Value.Le(lambda deps, dtype: 2**2),
                        cp.Size.Ge(lambda deps, r, d: 1),
                        cp.Size.Le(lambda deps, r, d: 2**2),
                    ]
                )
            if in_spec.name == "max_val":  # hardtanh
                spec.inspec[index].deps = [0, 1]
                spec.inspec[index].constraints.extend(
                    [cp.Value.Ge(lambda deps, _: deps[1])]
                )
            else:
                spec.inspec[index].constraints.extend(
                    [
                        cp.Value.Gt(lambda deps, dtype: 0),
                        cp.Value.Le(lambda deps, dtype: 2),
                    ]
                )
        elif in_spec.type.is_scalar_type():
            spec.inspec[index].constraints.extend(
                [
                    cp.Dtype.In(lambda deps: apply_scalar_contraints(op_name)),
                ]
            )
        elif in_spec.type.is_tensor():
            spec.inspec[index].constraints.extend(
                apply_tensor_contraints(op_name, index)
            )
        elif in_spec.type.is_dim_list():
            spec.inspec[index].constraints.extend(
                [
                    cp.Length.Ge(lambda deps: 1),
                    cp.Optional.Eq(lambda deps: False),
                ]
            )
        elif in_spec.type.is_bool():
            spec.inspec[index].constraints.extend(
                [
                    cp.Dtype.In(lambda deps: [torch.bool]),
                ]
            )

    return [
        (posargs, inkwargs)
        for posargs, inkwargs, _ in ArgumentTupleGenerator(spec).gen()
    ][:MAX_CASES]
