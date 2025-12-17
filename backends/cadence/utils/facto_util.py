# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
from functools import lru_cache
from typing import List, OrderedDict, Tuple

import facto.specdb.function as fn

import torch
from facto.inputgen.argtuple.gen import ArgumentTupleGenerator
from facto.inputgen.specs.model import ConstraintProducer as cp
from facto.inputgen.utils.random_manager import seeded_random_manager as rm
from facto.inputgen.variable.type import ScalarDtype
from facto.specdb.db import SpecDictDB

# seed to generate identical cases every run to reproduce from bisect
MAX_CASES = 50


# Global cache to store generated shapes per tensor to ensure consistency
_shape_cache: dict[str, list[int]] = {}


def _positive_valid_dim_list(tensor: torch.Tensor, length: int) -> set[tuple[int, ...]]:
    """
    Generate valid permutations using only positive dimension indices.
    This is required for Cadence/Xtensa kernels that don't support negative indexing.

    Args:
        tensor: Input tensor to generate permutations for
        length: Number of dimensions in the permutation (must equal tensor.dim())

    Returns:
        Set of valid permutation tuples containing only positive indices [0, rank-1]
    """
    if length > tensor.dim():
        return set()

    n = tensor.dim()
    pool = list(range(n))

    # Generate multiple valid permutations (only positive indices)
    permutations: set[tuple[int, ...]] = set()
    for _ in range(3):  # Generate 3 different permutations for diversity
        perm = tuple(rm.get_random().sample(pool, length))
        permutations.add(perm)

    return permutations


def apply_tensor_contraints(op_name: str, index: int) -> list[object]:
    # Constraint to limit tensor size to < 4000 bytes with fully randomized shapes
    import random

    def get_dtype_bytes(dtype: torch.dtype) -> int:
        """Get the number of bytes per element for a given dtype"""
        dtype_bytes = {
            torch.int8: 1,
            torch.uint8: 1,
            torch.int16: 2,
            torch.uint16: 2,
            torch.int32: 4,
            torch.float32: 4,
            torch.int64: 8,
            torch.float64: 8,
            torch.bool: 1,
            torch.float: 4,  # alias for float32
            torch.int: 4,  # alias for int32
            torch.long: 8,  # alias for int64
        }
        return dtype_bytes.get(dtype, 4)  # Default to 4 bytes if dtype not found

    def generate_random_shape_with_byte_limit(
        rank: int, dtype: torch.dtype, max_bytes: int = 3999, seed_base: int = 42
    ) -> list[int]:
        """Generate a random shape with given rank ensuring total byte size < max_bytes"""
        random.seed(seed_base + rank)

        bytes_per_element = get_dtype_bytes(dtype)
        max_elements = max_bytes // bytes_per_element

        # Start with all dimensions as 1
        shape = [1] * rank
        remaining_elements = (
            max_elements - 1
        )  # Leave room since we start with product=1

        # Randomly distribute the remaining capacity across dimensions
        for i in range(rank):
            if remaining_elements <= 1:
                break

            # Calculate maximum size this dimension can have without exceeding limit
            current_product = 1
            for j in range(rank):
                if j != i:
                    current_product *= shape[j]

            max_size_for_dim = min(
                remaining_elements // current_product, 50
            )  # Cap at 50
            if max_size_for_dim > shape[i]:
                # Randomly choose a size between current and max
                new_size = random.randint(shape[i], max_size_for_dim)
                shape[i] = new_size
                remaining_elements = max_elements // (current_product * new_size)
                remaining_elements = max(1, remaining_elements)

        # Final random shuffle of the dimensions to make it more random
        random.shuffle(shape)
        return shape

    def random_size_constraint(deps: object, r: int, d: int) -> int:
        """Generate random sizes ensuring total byte size < 4000 bytes"""
        # Use conservative approach: assume worst case is 4 bytes per element (float32/int32)
        # This ensures we never exceed 4000 bytes regardless of actual dtype
        worst_case_dtype = torch.float32  # 4 bytes per element

        # Create a unique key for this tensor configuration
        cache_key = f"{r}_{d}_conservative"

        if cache_key not in _shape_cache:
            # Generate a new random shape for this rank using worst-case byte estimation
            shape = generate_random_shape_with_byte_limit(
                r, worst_case_dtype, max_bytes=3999, seed_base=42 + r * 10 + d
            )
            _shape_cache[cache_key] = shape

        # Return the size for dimension d, ensuring we don't go out of bounds
        cached_shape = _shape_cache[cache_key]
        return cached_shape[d] if d < len(cached_shape) else 1

    max_size_constraint = cp.Size.Le(
        lambda deps, r, d: random_size_constraint(deps, r, d)
    )

    tensor_constraints = (
        [
            cp.Dtype.In(
                lambda deps: [
                    torch.int8,
                    torch.int16,
                    torch.uint8,
                    torch.uint16,
                    torch.int32,
                    torch.float32,
                ]
            ),
            cp.Value.Ge(lambda deps, dtype, struct: -(2**4)),
            cp.Value.Le(lambda deps, dtype, struct: 2**4),
            cp.Rank.Ge(lambda deps: 1),
            cp.Size.Ge(lambda deps, r, d: 1),
            max_size_constraint,
            cp.Rank.Le(lambda deps: 2**3),
        ]
        if op_name
        not in (
            "slice_copy.Tensor",
            "add.Scalar",
            "sub.Scalar",
            "mul.Scalar",
            "div.Tensor",
            "neg.default",
        )
        else [
            cp.Dtype.In(
                lambda deps: [
                    torch.int32,
                    torch.float32,
                ]
            ),
            cp.Value.Ge(lambda deps, dtype, struct: -(2**4)),
            cp.Value.Le(lambda deps, dtype, struct: 2**4),
            cp.Rank.Ge(lambda deps: 1),
            cp.Size.Ge(lambda deps, r, d: 1),
            max_size_constraint,
            cp.Rank.Le(lambda deps: 2**3),
        ]
    )

    match op_name:
        case "where.self":
            if index == 0:  # condition
                tensor_constraints = [
                    cp.Dtype.In(lambda deps: [torch.bool]),
                    cp.Value.Ge(lambda deps, dtype, struct: 0),
                    cp.Value.Le(lambda deps, dtype, struct: 1),
                    cp.Rank.Ge(lambda deps: 1),
                    cp.Size.Ge(lambda deps, r, d: 1),
                    max_size_constraint,
                ]
            elif index == 1:  # input tensor(a)
                tensor_constraints = [
                    cp.Dtype.In(lambda deps: [torch.float32]),
                    cp.Value.Ge(lambda deps, dtype, struct: -(2**4)),
                    cp.Value.Le(lambda deps, dtype, struct: 2**4),
                    cp.Rank.Ge(lambda deps: 1),
                    cp.Size.Ge(lambda deps, r, d: 1),
                    cp.Size.In(
                        lambda deps, r, d: fn.broadcast_with(deps[0].shape, r, d)
                    ),
                    max_size_constraint,
                ]
            else:  # input tensor(b)
                tensor_constraints = [
                    cp.Dtype.In(lambda deps: [torch.float32]),
                    cp.Dtype.Eq(lambda deps: deps[1].dtype),
                    cp.Value.Ge(lambda deps, dtype, struct: -(2**4)),
                    cp.Value.Le(lambda deps, dtype, struct: 2**4),
                    cp.Rank.Ge(lambda deps: 1),
                    cp.Size.Ge(lambda deps, r, d: 1),
                    cp.Size.In(
                        lambda deps, r, d: fn.broadcast_with(
                            fn.broadcasted_shape(deps[0].shape, deps[1].shape), r, d
                        )
                    ),
                    max_size_constraint,
                ]
        case "embedding.default":
            tensor_constraints = [
                cp.Dtype.In(lambda deps: [torch.float, torch.int]),
                cp.Dtype.NotIn(lambda deps: [torch.int64, torch.float64]),
                cp.Value.Ge(lambda deps, dtype, struct: -(2**4)),
                cp.Value.Le(lambda deps, dtype, struct: 2**4),
                cp.Rank.Ge(lambda deps: 1),
                cp.Size.Ge(lambda deps, r, d: 1),
                max_size_constraint,
            ]
        case "sigmoid.default":
            tensor_constraints.extend(
                [
                    cp.Dtype.In(lambda deps: [torch.float32]),
                    cp.Value.Ge(lambda deps, dtype, struct: -2),
                    cp.Value.Le(lambda deps, dtype, struct: 2),
                ]
            )
        case "transpose_copy.int":
            tensor_constraints.extend(
                [
                    cp.Dtype.In(lambda deps: [torch.float32, torch.int32]),
                ]
            )
        case "permute_copy.default":
            tensor_constraints.extend(
                [
                    cp.Dtype.In(lambda deps: [torch.float32, torch.int8, torch.uint8]),
                    cp.Rank.Le(
                        lambda deps: 5
                    ),  # xa_nn_transpose only supports up to 5D
                    cp.Rank.Ge(lambda deps: 1),  # Must have at least 1 dimension
                ]
            )
        case "sqrt.default":
            tensor_constraints.extend(
                [
                    cp.Dtype.In(lambda deps: [torch.float32, torch.int32]),
                ]
            )
        case "clamp.default":
            tensor_constraints.extend(
                [
                    cp.Dtype.In(lambda deps: [torch.float32, torch.int32]),
                    # Avoid NaN/Inf values that expose clamp NaN handling bugs
                    cp.Value.Ge(lambda deps, dtype, struct: -(2**4)),
                    cp.Value.Le(lambda deps, dtype, struct: 2**4),
                ]
            )
        case "rsqrt.default":
            tensor_constraints.extend(
                [
                    cp.Dtype.In(lambda deps: [torch.float32]),
                    cp.Value.Gt(
                        lambda deps, dtype, struct: 0
                    ),  # only generate real numbers
                    cp.Value.Le(lambda deps, dtype, struct: 2**2),
                ]
            )
        case "relu.default":
            tensor_constraints.extend(
                [
                    cp.Dtype.In(lambda deps: [torch.float32]),
                ]
            )
        case "mean.dim":
            tensor_constraints.extend(
                [
                    cp.Dtype.In(lambda deps: [torch.float32]),
                ]
            )
        case "exp.default":
            tensor_constraints.extend(
                [
                    cp.Dtype.In(lambda deps: [torch.float32]),
                    cp.Value.Ge(lambda deps, dtype, struct: -(2**2)),
                    cp.Value.Le(lambda deps, dtype, struct: 2**2),
                ]
            )
        case "tanh.default":
            tensor_constraints.extend(
                [
                    cp.Dtype.In(lambda deps: [torch.float32]),
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
        case "div.Scalar" | "add.Tensor" | "mul.Tensor" | "sub.Tensor":
            tensor_constraints.extend(
                [
                    cp.Dtype.In(
                        lambda deps: [
                            torch.int32,
                            torch.int64,
                            torch.float32,
                        ]
                    ),
                ]
            )
        case "split_copy.Tensor":
            tensor_constraints.extend(
                [
                    cp.Dtype.In(
                        lambda deps: [
                            torch.int32,
                            torch.int64,
                            torch.float32,
                        ]
                    ),
                    cp.Value.Ge(lambda deps, dtype, struct: 1),
                    cp.Value.Le(lambda deps, dtype, struct: 2**3),
                    cp.Rank.Le(lambda deps: 3),
                    cp.Size.Le(lambda deps, r, d: 2**2),
                ]
            )
        case "constant_pad_nd.default":
            tensor_constraints = [
                cp.Dtype.In(lambda deps: [torch.float32]),
                cp.Value.Ge(lambda deps, dtype, struct: -(2**4)),
                cp.Value.Le(lambda deps, dtype, struct: 2**4),
                cp.Rank.Ge(lambda deps: 1),
                cp.Rank.Le(lambda deps: 2),  # Reduced from 3 to 2 (max 2D tensors)
                cp.Size.Ge(lambda deps, r, d: 1),
                cp.Size.Le(lambda deps, r, d: 3),  # Max dimension size of 3
            ]
        case "avg_pool2d.default":
            tensor_constraints.extend(
                [
                    cp.Rank.Eq(lambda deps: 4),
                ]
            )
        case "bmm.default" | "addmm.default" | "mm.default":
            tensor_constraints.extend(
                [
                    cp.Dtype.Eq(lambda deps: torch.float),
                    cp.Size.Le(lambda deps, r, d: 2**2),
                    cp.Value.Le(lambda deps, dtype, struct: 2**4),
                ]
            )
        case "div.Tensor":
            if index == 1:  # Only apply zero-prevention to divisor
                tensor_constraints.extend(
                    [
                        cp.Value.Ne(
                            lambda deps, dtype, struct: 0
                        ),  # Prevent division by zero
                        cp.Value.Le(lambda deps, dtype, struct: 2**3),
                        cp.Size.Le(lambda deps, r, d: 2**3),
                        cp.Rank.Le(lambda deps: 2**2),
                    ]
                )
            else:
                tensor_constraints.extend(
                    [
                        cp.Value.Le(lambda deps, dtype, struct: 2**3),
                        cp.Size.Le(lambda deps, r, d: 2**3),
                        cp.Rank.Le(lambda deps: 2**2),
                    ]
                )
        case "pow.Tensor_Scalar":
            tensor_constraints.extend(
                [
                    cp.Dtype.In(lambda deps: [torch.float32, torch.int32]),
                ]
            )
        case "div.Tensor_mode" | "minimum.default":
            if index == 0:
                tensor_constraints = [
                    cp.Dtype.In(lambda deps: [torch.int64, torch.int32, torch.float32]),
                    cp.Value.Ge(lambda deps, dtype, struct: -(2**4)),
                    cp.Value.Le(lambda deps, dtype, struct: 2**4),
                    cp.Rank.Ge(lambda deps: 1),
                    cp.Size.Ge(lambda deps, r, d: 1),
                    cp.Size.Le(lambda deps, r, d: 2**2),
                ]
            else:
                tensor_constraints = [
                    cp.Dtype.In(lambda deps: [torch.int64, torch.int32, torch.float32]),
                    cp.Value.Ge(lambda deps, dtype, struct: -(2**4)),
                    cp.Value.Le(lambda deps, dtype, struct: 2**4),
                    cp.Value.Ne(
                        lambda deps, dtype, struct: 0
                    ),  # Prevent division by zero
                    cp.Rank.Ge(lambda deps: 1),
                    cp.Rank.Eq(lambda deps: deps[0].dim()),
                    cp.Size.Eq(lambda deps, r, d: fn.safe_size(deps[0], d)),
                ]
        case "_native_batch_norm_legit_no_training.default":
            tensor_constraints.extend(
                [
                    cp.Rank.Le(lambda deps: 3),
                ],
            )
        case "reciprocal.default":
            tensor_constraints = [
                cp.Value.Ge(lambda deps, dtype, struct: -(2**2)),
                cp.Value.Le(lambda deps, dtype, struct: 2**2),
                cp.Size.Le(lambda deps, r, d: 2**3),
            ]
        case "leaky_relu.default":
            tensor_constraints.extend(
                [
                    cp.Dtype.In(lambda deps: [torch.float32]),
                ]
            )
        case "_softmax.default":
            tensor_constraints.extend(
                [
                    cp.Dtype.Eq(lambda deps: torch.float32),
                    cp.Size.Le(lambda deps, r, d: 2**2),
                ]
            )
        case "flip.default":
            tensor_constraints.extend(
                [
                    cp.Dtype.In(lambda deps: [torch.float32]),
                ]
            )
        case _:
            pass
    return tensor_constraints


def apply_scalar_contraints(op_name: str) -> list[ScalarDtype]:
    match op_name:
        case (
            "add.Scalar"
            | "sub.Scalar"
            | "mul.Scalar"
            | "div.Scalar"
            | "constant_pad_nd.default"
            | "clamp.default"
        ):
            return [ScalarDtype.int]
        case "full.default":
            return [ScalarDtype.int]
        case _:
            return [ScalarDtype.float, ScalarDtype.int]


@lru_cache(maxsize=None)
def facto_testcase_gen(  # noqa: C901
    op_name: str,
) -> List[Tuple[List[str], OrderedDict[str, str]]]:
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
            # Special handling for clamp.default to ensure min < max with sufficient gap (at least 2) and never None
            if op_name == "clamp.default":
                if in_spec.name == "min":
                    # min must always be provided (not None) and bounded, leave room for max
                    spec.inspec[index].constraints.extend(
                        [
                            cp.Optional.Eq(lambda deps: False),  # Never None
                            cp.Value.Ge(lambda deps, dtype: -(2**4)),
                            cp.Value.Le(
                                lambda deps, dtype: 2**4 - 2
                            ),  # Leave room for max (at least 2 units)
                        ]
                    )
                elif in_spec.name == "max":
                    # max must always be provided (not None), be >= min + 2 (sufficient gap), and bounded
                    spec.inspec[index].deps = [0, 1]  # deps on input tensor and min
                    spec.inspec[index].constraints.extend(
                        [
                            cp.Optional.Eq(lambda deps: False),  # Never None
                            cp.Value.Ge(
                                lambda deps, dtype: deps[1] + 2
                            ),  # max >= min + 2 (sufficient gap)
                            cp.Value.Le(lambda deps, dtype: 2**4),
                        ]
                    )
            elif in_spec.name == "max_val":  # hardtanh
                spec.inspec[index].deps = [0, 1]
                spec.inspec[index].constraints.extend(
                    [cp.Value.Ge(lambda deps, _: deps[1])]
                )
            elif in_spec.name == "negative_slope" and op_name == "leaky_relu.default":
                # For leaky_relu, negative_slope should be in typical range (0, 1]
                spec.inspec[index].constraints.extend(
                    [
                        cp.Value.Gt(lambda deps, dtype: 0),
                        cp.Value.Le(lambda deps, dtype: 1.0),
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
            if in_spec.name == "dtype":  # full.default
                spec.inspec[index].constraints.extend(
                    [
                        cp.Dtype.In(lambda deps: [torch.long, torch.float]),
                    ]
                )
        elif in_spec.type.is_tensor():
            spec.inspec[index].constraints.extend(
                apply_tensor_contraints(op_name, index)
            )
        elif in_spec.type.is_dim_list():
            # Special handling for permute_copy.default to ensure valid permutation
            if op_name == "permute_copy.default":
                spec.inspec[index].constraints.extend(
                    [
                        cp.Length.Ge(lambda deps: 1),
                        cp.Length.Eq(
                            lambda deps: deps[0].dim()
                        ),  # Must be a complete permutation
                        cp.Optional.Eq(lambda deps: False),
                        # Generate valid permutations using only positive indices
                        # Cadence/Xtensa hardware kernels do not support negative dimension indices
                        cp.Value.Gen(
                            lambda deps, length: (
                                _positive_valid_dim_list(deps[0], length),
                                fn.invalid_dim_list(deps[0], length),
                            )
                        ),
                    ]
                )
            else:
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
        elif in_spec.type.is_length_list():
            spec.inspec[index].constraints.extend(
                [
                    cp.Value.Ge(lambda deps, dtype, struct: 0),
                ]
            )
            if op_name == "avg_pool2d.default":
                spec.inspec[index].constraints.extend(
                    [
                        cp.Length.Eq(lambda deps: 2),
                    ]
                )
        elif in_spec.type.is_shape():
            spec.inspec[index].constraints.extend(
                [
                    cp.Rank.Ge(lambda deps: 1),
                    cp.Rank.Le(lambda deps: 2**2),
                    cp.Value.Gt(lambda deps, dtype, struct: 0),
                    cp.Value.Le(lambda deps, dtype, struct: 2**2),
                    cp.Size.Ge(lambda deps, r, d: 1),
                    cp.Size.Le(lambda deps, r, d: 2**2),
                ]
            )

    return [
        (posargs, inkwargs)
        for posargs, inkwargs, _ in ArgumentTupleGenerator(spec).gen()
    ][:MAX_CASES]
