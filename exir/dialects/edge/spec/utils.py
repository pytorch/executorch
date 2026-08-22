# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from functools import partial
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from executorch.exir.dialects.edge.arg.model import BaseArg, BaseKwarg, Return
from executorch.exir.dialects.edge.op.sample_input import SAMPLE_INPUT

from torch._ops import OpOverloadPacket


def is_tensor_arg(arg: torch._C.Argument) -> bool:
    """Check if a given argument is a Tensor argument."""
    arg_type_str: str = str(arg.type)
    return arg_type_str in ["Tensor", "Optional[Tensor]", "List[Tensor]"]


def is_tensor_val(arg: Any) -> bool:
    """Check if a given value is a Tensor-like value.
    Please make sure the legal value in this function should be same as is_tensor_arg"""
    if isinstance(arg, torch.Tensor):
        return True
    if isinstance(arg, list) or isinstance(arg, tuple):
        return len(arg) > 0 and all(isinstance(v, torch.Tensor) for v in arg)
    return False


def is_essential_tensor_arg(arg: torch._C.Argument) -> bool:
    """Check if a given argument is a Tensor argument."""
    arg_type_str: str = str(arg.type)
    return arg_type_str in ["Tensor", "List[Tensor]"]


def is_optional_tensor_arg(arg: torch._C.Argument) -> bool:
    """Check if a given argument is a Tensor argument."""
    arg_type_str: str = str(arg.type)
    return arg_type_str in ["Optional[Tensor]"]


def get_tensor_variable_names(
    func_schema: torch._C.FunctionSchema,
) -> Tuple[List[str], List[str], List[str]]:
    """Get names of essential tensor variables, optional tensor
    variables and all tensor variables from given function schema.
    The tensor variables here include both input tensors and output tensors."""

    essential_tensor_arg_names: List[str] = [
        arg.name for arg in func_schema.arguments if is_essential_tensor_arg(arg)
    ]
    optional_tensor_arg_names: List[str] = [
        arg.name for arg in func_schema.arguments if is_optional_tensor_arg(arg)
    ]
    all_tensor_arg_names: List[str] = [
        arg.name for arg in func_schema.arguments if is_tensor_arg(arg)
    ]

    return_tensor_variable_names: List[str] = []

    ret_name_base = "__ret_"
    ret_id = 0
    for ret in func_schema.returns:
        name = ret.name if ret.name else f"{ret_name_base}{ret_id}"
        if is_tensor_arg(ret):
            return_tensor_variable_names.append(name)
            ret_id += 1
    return (
        essential_tensor_arg_names + return_tensor_variable_names,
        optional_tensor_arg_names,
        all_tensor_arg_names + return_tensor_variable_names,
    )


def get_args_rets(op_name: str) -> List[BaseArg]:
    args_rets: List[BaseArg] = []
    args_rets.extend(SAMPLE_INPUT[op_name].get("args", []))
    args_rets.extend(SAMPLE_INPUT[op_name].get("returns", []))
    return args_rets


def get_names_for_args_with_dtype(
    op_name: str, func_schema: torch._C.FunctionSchema
) -> List[str]:
    """Dtype runner is returning dtypes for more arguments than edge dialect cares about.
    This function returns a list of booleans to select the dtypes matter to edge dialect.
    """
    args_rets: List[BaseArg] = get_args_rets(op_name)
    names = []
    arguments, returns = func_schema.arguments, func_schema.returns
    args, kwargs, rets = [], [], []
    for arg in args_rets:
        if isinstance(arg, Return):
            rets.append(arg)
        elif isinstance(arg, BaseKwarg):
            kwargs.append(arg)
        else:
            args.append(arg)
    names.extend(
        [
            schema.name
            for sample, schema in zip(args, arguments)
            if sample.type.has_dtype()
        ]
    )
    names.extend([sample.argname for sample in kwargs if sample.type.has_dtype()])
    ret_name_base = "__ret_"
    for ret_id, (_, schema) in enumerate(zip(rets, returns)):
        names.append(schema.name if schema.name else f"{ret_name_base}{ret_id}")
    return names


def get_torch_op_overload(
    namespace: str, opname: str, overload: Optional[str]
) -> torch._ops.OpOverload:
    packet: OpOverloadPacket = getattr(getattr(torch.ops, namespace), opname)
    if overload:
        return getattr(packet, overload)
    else:
        return packet.default


def group_by_format(
    all_combinations: Set[Tuple[str]],
) -> List[Tuple[int, Tuple[Tuple[str]]]]:
    """Taking combinations that having same format of all_combinations as a group.
    Two combinations having same format here means one and only one of their
    corresponding input tensors is different. e.g. {Tensor(0), Tensor(0), Tensor(1)}
    shares same format with {Tensor(0), Tensor(0), Tensor(0)},
    but not {Tensor(0), Tensor(1), Tensor(0)}.
    """

    grouped_combinations: Set[Tuple[int, Tuple[Tuple[str]]]] = set()

    def almost_same_except(b: Tuple[str], combination: Tuple[str], index: int):
        """Check if a and b share same format"""
        for i, (aa, bb) in enumerate(zip(combination, b)):
            if (i == index and aa == bb) or (i != index and aa != bb):
                return False
        return True

    for combination in all_combinations:
        # filter out combinations that only differ at index
        has_same_comb: bool = False
        for index in range(len(combination)):
            filtered: Set[Tuple[str]] = set()
            filtered.add(combination)
            combo_filter = partial(
                almost_same_except, combination=combination, index=index
            )
            filtered.update(set(filter(combo_filter, all_combinations)))
            if len(filtered) > 1:
                has_same_comb = True
                grouped_combinations.add((index, tuple(sorted(filtered))))
        if not has_same_comb:
            grouped_combinations.add((-1, (combination,)))
    return list(grouped_combinations)


def update_type_alias(type_alias: Dict[Tuple[str], int], new_key: Tuple[str]) -> None:
    """Update type_alias with new type alias"""
    if new_key not in type_alias:
        type_alias[new_key] = len(type_alias)


def gen_index_pairs_to_types_mapping(
    type_alias: Dict[Tuple[str], int], type_constraint: List[List[int]]
) -> Dict[Tuple[int], List[str]]:
    """Generate mapping from index pairs to types. For example, given type_constraint [0, 0], [1, 1]
    type_alias ('Double',): 0, ('Int',): 1, output will be {(0, 1): ['Double', 'Int', 'Double', 'Int']}.
    """

    def gen(x: List[int]):
        """Generate all possible pairs of elements in the list."""
        for i in range(len(x) - 1):
            for j in range(i + 1, len(x)):
                yield (x[i], x[j])

    reverse: Dict[Tuple[int], Set[str]] = defaultdict(set)
    for constraint in type_constraint:
        # collect indices of elements with the same value. Value is a list of indices.
        positions: Dict[int, List[int]] = defaultdict(list)
        for i, val in enumerate(constraint):
            positions[val].append(i)
        for key, val in positions.items():
            # key is type_alias value which is alias index
            alias = next(k for k, v in type_alias.items() if v == key)
            # only care about pairs for now. Key to reverse is the pair of indices where elements are the same. Value is the list of types.
            for pair in gen(val):
                reverse[pair].update(alias)
    return {k: sorted(v) for k, v in reverse.items()}


def check_new_alias_fit_constraints(
    type_alias: Dict[Tuple[str], int],
    type_constraint: List[List[int]],
    new_alias: Tuple[str],
) -> bool:
    """Check whether new type alias fits the existing constraints.
    For example, for existing aliases ('Float'): 0, ('Int'): 1, a new alias of ('Float, Int') and type_constraint is [[0, 0]]
    This new alias doesn't fit because we need [[0, 0], [0, 1]] to be satisfied.
    """
    constraint_set: Set[Tuple[int]] = {
        tuple(constraint) for constraint in type_constraint
    }
    length = len(type_constraint[0])
    subset: Set[Tuple[int]] = {
        tuple([type_alias[(type_info,)]] * length) for type_info in new_alias
    }
    return subset.issubset(constraint_set)


def aggregate_if_two_types_being_the_same(
    type_alias: Dict[Tuple[str], int], type_constraint: List[List[int]]
) -> Tuple[List[Tuple[str]], List[Tuple[int]]]:
    """aggregate the type constraints that has two types being the same, at the same position.
    For example, [0, 0] and [1, 1] where ('Double',): 0, ('Int',): 1 can be aggregated into
    [2, 2] where ('Double', 'Int'): 3.
    """

    reverse: Dict[Tuple[int], List[str]] = gen_index_pairs_to_types_mapping(
        type_alias, type_constraint
    )

    idx_to_update: Set[int] = set()
    for alias in reverse.values():
        alias_tuple = tuple(alias)
        if alias_tuple in type_alias or not check_new_alias_fit_constraints(
            type_alias, type_constraint, alias_tuple
        ):
            continue
        idx_to_update.update(
            v for k, v in type_alias.items() if {*k}.issubset({*alias_tuple})
        )
        # update type_alias to include new type alias.
        type_alias[alias_tuple] = len(type_alias)
        # replace indices within alias to be new alias index.
        for i in range(len(type_constraint)):
            for j, a in enumerate(type_constraint[i]):
                if a in idx_to_update:
                    type_constraint[i][j] = type_alias[alias_tuple]

    # remove unused aliases
    type_alias = {k: v for k, v in type_alias.items() if v not in idx_to_update}
    sorted_keys = sorted(type_alias.keys())
    # map indices back to start from 0 contiguous
    index_map = {type_alias[sorted_keys[i]]: i for i in range(len(sorted_keys))}
    # remove duplicate constraints
    constraint_set: Set[Tuple[int]] = {
        tuple(index_map[i] for i in c) for c in type_constraint
    }

    return list(sorted_keys), sorted(constraint_set)


def aggregate_grouped_type_combinations(
    grouped_combinations: List[Tuple[int, Tuple[Tuple[str]]]],
) -> Tuple[Dict[Tuple[str], int], List[List[int]]]:
    """Aggregate grouped type combinations."""
    type_alias: Dict[Tuple[str], int] = {}
    type_constraint: List[List[int]] = []
    for distinct_id, same_format_combinations in grouped_combinations:
        comb_iter = iter(same_format_combinations)
        if len(same_format_combinations) == 1:
            # can not combine with others; each type in the comb is am individual type alias.
            comb: Tuple[str] = next(comb_iter)
            temp_type_constraint: List[int] = []
            for type_str in comb:
                update_type_alias(type_alias, (type_str,))
                temp_type_constraint.append(type_alias[(type_str,)])
            type_constraint.append(temp_type_constraint)
        else:
            # gather different types in each combinations together as a list
            # make the list as a separate type alias
            all_distinct_types: Tuple[str] = tuple(
                sorted({sf_comb[distinct_id] for sf_comb in same_format_combinations})
            )

            update_type_alias(type_alias, all_distinct_types)

            comb: Tuple[str] = next(comb_iter)
            temp_type_constraint: List[int] = []
            # assign each type of the format to a single type alias
            for i, type_str in enumerate(comb):
                if i == distinct_id:
                    temp_type_constraint.append(type_alias[all_distinct_types])
                else:
                    update_type_alias(type_alias, (type_str,))
                    temp_type_constraint.append(type_alias[(type_str,)])

            type_constraint.append(temp_type_constraint)
    return type_alias, type_constraint


def type_aggregrate(
    allow_types: Set[Tuple[str]],
) -> Tuple[List[Tuple[str]], List[Tuple[int]]]:
    """
    This function aims to aggreate the enumerate combinations of supported types into type alias format.
    E.g. input: [["Float", "Float", "Float"], ["Half", "Half", "Half"], ["Char", "Char", "Int"]]
            output: [["Float", "Half"], ["Char"], ["Int"]], [[0, 0, 0], [1, 1, 2]]

            for i-dx list in the type_constraint, any j in [0, len(self.tensor_variable_names)) self.tensor_variable_names[j],
            can be in one of the types in type_alias[type_constraint[i][j]]; also self.tensor_variable_names[k] and
            self.tensor_variable_names[l] shoule be same if type_constraint[i][k] == type_constraint[i][l].

    NOTE: This is not the optimum way to aggregate types. It generates correct but not the optimum representation.
    TODO(gasoonjia): continue update aggregrate algorithm.
    """

    # group combinations with the same format
    grouped_combinations: List[Tuple[int, Tuple[Tuple[str]]]] = group_by_format(
        allow_types
    )

    type_alias, type_constraint = aggregate_grouped_type_combinations(
        grouped_combinations
    )

    sorted_type_alias, sorted_type_constraint = aggregate_if_two_types_being_the_same(
        type_alias, type_constraint
    )

    return sorted_type_alias, sorted_type_constraint
