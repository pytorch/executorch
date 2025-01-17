# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import copy
from typing import List, OrderedDict, Tuple

import torch
from inputgen.argtuple.gen import ArgumentTupleGenerator
from inputgen.specs.model import ConstraintProducer as cp
from inputgen.utils.random_manager import random_manager
from inputgen.variable.type import ScalarDtype
from specdb.db import SpecDictDB

# seed to generate identical cases every run to reproduce from bisect
random_manager.seed(1729)


def apply_tensor_contraints(op_name: str, tensor_constraints: list[object]) -> None:
    match op_name:
        case (
            "sigmoid.default"
            | "_softmax.default"
            | "rsqrt.default"
            | "exp.default"
            | "mul.Tensor"
            | "div.Tensor"
        ):
            tensor_constraints.append(
                cp.Dtype.In(lambda deps: [torch.float]),
            )
        case (
            "add.Tensor"
            | "sub.Tensor"
            | "add.Scalar"
            | "sub.Scalar"
            | "mul.Scalar"
            | "div.Scalar"
        ):
            tensor_constraints.append(
                cp.Dtype.In(lambda deps: [torch.float, torch.int]),
            )
        case _:
            tensor_constraints.append(
                cp.Dtype.In(lambda deps: [torch.float, torch.int]),
            )
    tensor_constraints.extend(
        [
            cp.Value.Ge(lambda deps, dtype, struct: -(2**8)),
            cp.Value.Le(lambda deps, dtype, struct: 2**8),
            cp.Rank.Ge(lambda deps: 1),
            cp.Rank.Le(lambda deps: 2**2),
            cp.Size.Ge(lambda deps, r, d: 1),
            cp.Size.Le(lambda deps, r, d: 2**2),
        ]
    )


def facto_testcase_gen(op_name: str) -> List[Tuple[List[str], OrderedDict[str, str]]]:
    # minimal example to test add.Tensor using FACTO
    spec = SpecDictDB[op_name]

    for index, in_spec in enumerate(copy.deepcopy(spec.inspec)):
        if in_spec.type.is_scalar():
            if in_spec.name != "alpha":
                spec.inspec[index].constraints.extend(
                    [
                        cp.Dtype.In(lambda deps: [ScalarDtype.float, ScalarDtype.int]),
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
        elif in_spec.type.is_tensor():
            tensor_constraints = []
            # common tensor constraints
            apply_tensor_contraints(op_name, tensor_constraints)
            spec.inspec[index].constraints.extend(tensor_constraints)

    return [
        (posargs, inkwargs)
        for posargs, inkwargs, _ in ArgumentTupleGenerator(spec).gen()
    ]
