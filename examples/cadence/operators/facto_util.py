# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import copy
from typing import List, OrderedDict, Tuple

import torch
from facto.inputgen.argtuple.gen import ArgumentTupleGenerator
from facto.inputgen.specs.model import ConstraintProducer as cp
from facto.inputgen.utils.random_manager import random_manager
from facto.inputgen.variable.type import ScalarDtype
from facto.specdb.db import SpecDictDB

# seed to generate identical cases every run to reproduce from bisect
random_manager.seed(1729)


def apply_tensor_contraints(op_name: str, tensor_constraints: list[object]) -> None:
    match op_name:
        case "sigmoid.default" | "rsqrt.default":
            tensor_constraints.extend(
                [
                    cp.Dtype.In(lambda deps: [torch.float]),
                    cp.Rank.Le(lambda deps: 2**2),
                    cp.Value.Ge(lambda deps, dtype, struct: -2),
                    cp.Value.Le(lambda deps, dtype, struct: 2),
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
        case _:
            tensor_constraints.extend(
                [
                    cp.Rank.Le(lambda deps: 2**2),
                ]
            )
    tensor_constraints.extend(
        [
            cp.Dtype.In(lambda deps: [torch.int, torch.float]),
            cp.Dtype.NotIn(lambda deps: [torch.int64, torch.float64]),
            cp.Value.Ge(lambda deps, dtype, struct: -(2**4)),
            cp.Value.Le(lambda deps, dtype, struct: 2**4),
            cp.Rank.Ge(lambda deps: 1),
            cp.Size.Ge(lambda deps, r, d: 1),
            cp.Size.Le(lambda deps, r, d: 2**9),
        ]
    )


def apply_scalar_contraints(op_name: str) -> list[ScalarDtype]:
    match op_name:
        case "add.Scalar" | "sub.Scalar" | "mul.Scalar" | "div.Scalar":
            return [ScalarDtype.int]
        case _:
            return [ScalarDtype.float, ScalarDtype.int]


def facto_testcase_gen(op_name: str) -> List[Tuple[List[str], OrderedDict[str, str]]]:
    # minimal example to test add.Tensor using FACTO
    spec = SpecDictDB[op_name]
    tensor_constraints = []
    # common tensor constraints
    apply_tensor_contraints(op_name, tensor_constraints)

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
            spec.inspec[index].constraints.extend(tensor_constraints)
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
    ]
