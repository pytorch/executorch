# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

r"""
Handle the following op convertions:
- convert a functional op to an out variant op
- convert an out variant op to a scratch op.

We assume there is already a functionalization pass being done that removes aliases and inplace variants.

For the to_out_variant convertion, The functional variant will be represented
as qualified op name plus the overload name. The returned out variant constains
the following information
- the OpOverload for the out variant
- the list of keyward arguments names that are out variables. There should be
  at least one out variables. Some ops may also have multiple out variables,
  e.g. aten::topk.values returns both values and indices for the topk elements.

"""

import dataclasses
import logging
from typing import Dict, Optional, Tuple

import torch
from torch._ops import OpOverload
from torchgen.model import FunctionSchema, SchemaKind

# cache the FunctionSchema so we don't need to parse everytime>
# Use OpOverload as hash key. We can not use torch._C.FunctionSchema as key since
# it's not hashable.
_op_overload_to_schema_cache: Dict[OpOverload, FunctionSchema] = {}

# Value type is Optional so we can cache None if an op does not have
# out variant/scratch op. This way, we don't need to confuse the op not
# existing case with cache miss.
_func_to_out_variant_map: Dict[OpOverload, Optional[OpOverload]] = {}
_out_variant_to_scratch_map: Dict[OpOverload, Optional[OpOverload]] = {}
_mutable_to_out_variant_map: Dict[OpOverload, Optional[OpOverload]] = {}

# We've found a functional and an out variant with the same name, but their
# schemas mismatch. This map collects all of these cases and provides proper
# error message to user. The key is an `OpOverload` of a functional variant.
_schema_mismatch_map: Dict[OpOverload, Optional[FunctionSchema]] = {}


def _pybind_schema_to_native_schema(
    pybind_schema: torch._C.FunctionSchema,
) -> Optional[FunctionSchema]:
    """
    We have 2 FunctionSchema definitions in python.
    One is defined in torchgen (call it native FunctionSchema), another is a
    pybind of c10::FunctionSchema (call it pybind FunctionSchema).
    Because we want to leverage torchgen to handle out variant, we will
    convert any pybind FunctionSchema to native FunctionSchema.
    """
    native_schema = None
    try:
        native_schema = FunctionSchema.parse(str(pybind_schema))
    except (RuntimeError, AssertionError, ValueError):
        # Need catch AssertionError since parsing prim ops like:
        #   aten::to.prim_other(Tensor(a) self, bool non_blocking=False, bool copy=False) -> Tensor(a|b)
        # cause an asertion error in torchgen when parsiong annotation 'a|b'.
        # We should ignore it. Hopefully one day the C++ FunctionSchema parsing
        # is 100% consistent with Python FunctionSchema parsing, then we don't need
        # catch these exceptions any more.

        # We also need catch ValueError for schema like:
        #   aten::copy.Dict_str(Dict(str, t)(a) self) -> Dict(str, t)
        # torchgen throws ValueError since it does not expect the type string
        # containing commas. Ignore those schemas for now.
        logging.debug(f"Fail to parse function schema: {str(pybind_schema)}")
        # ignore failure and return None. There are some schemas defined as
        # prim ops that can not be parsed by torchgen. E.g.:
        #   https://www.fburl.com/code/1vvzhssa
        # We should be safe to ignore them since PyE are not using these ops.
    return native_schema


def _get_overload_schema(op_overload: OpOverload) -> Optional[FunctionSchema]:
    native_schema = _op_overload_to_schema_cache.get(op_overload)
    if not native_schema:
        native_schema = _pybind_schema_to_native_schema(op_overload._schema)
        _op_overload_to_schema_cache[op_overload] = native_schema  # pyre-ignore
    return native_schema


def get_out_args_from_opoverload(op_overload: OpOverload) -> Tuple[str]:
    return get_out_args_from_schema(_get_overload_schema(op_overload))  # pyre-ignore


def get_out_args_from_schema(out_var_schema: FunctionSchema) -> Tuple[str]:
    """
    Assume the input is the schema for an out variant.
    Return the name list of the out arguments.
    """
    assert (
        out_var_schema.is_out_fn()
    ), f"Expect an out variant, but get: {out_var_schema}"
    return tuple(arg.name for arg in out_var_schema.arguments.out)


def parse_qualified_opname(qualified_opname: str) -> Tuple[str, str]:
    """
    Given a qualified opname like aten::add, return a tuple for namespace
    (aten here) and op name (add here)
    """
    ns_and_opname = qualified_opname.split("::")
    if len(ns_and_opname) != 2:
        raise RuntimeError(f"Invalid qualified_opname {qualified_opname}")
    return tuple(ns_and_opname)


def get_op_overload(qualified_opname: str, overload: str) -> OpOverload:
    """
    Arguments:
        qualified_opname: string like {namespace}::{op name}
        overload: the overload string of the op
    """
    ns, opname = parse_qualified_opname(qualified_opname)
    if not overload:
        overload = "default"
    return getattr(getattr(getattr(torch.ops, ns), opname), overload)


def schema_to_opoverload(schema: FunctionSchema) -> OpOverload:
    qualified_name = str(schema.name.name)
    overload = schema.name.overload_name
    return get_op_overload(qualified_name, overload)


def set_mapping_for_op(op: OpOverload) -> None:
    """
    op can either be a functional op, mutable op, or out variant op.
    This method is only called if
    1. either op is a functional op and it's missing in the _func_to_out_variant_map cache.
    2. or op is a out variant op and it's missing in the _out_variant_to_scratch_map cache.

    Setup entries in _func_to_out_variant_map and _out_variant_to_scratch_map for all ops sharing the same
    op name as the passed in OpOverload.
    """
    native_schema = _pybind_schema_to_native_schema(op._schema)
    # pyre-fixme[16]: `Optional` has no attribute `kind`.
    assert native_schema.kind() in (
        SchemaKind.functional,
        SchemaKind.out,
        SchemaKind.mutable,
    )
    assert not (
        native_schema.kind() == SchemaKind.functional and op in _func_to_out_variant_map
    )
    assert not (
        native_schema.kind() == SchemaKind.out and op in _out_variant_to_scratch_map
    )
    assert not (
        native_schema.kind() == SchemaKind.mutable and op in _mutable_to_out_variant_map
    )
    qualified_opname = str(op._schema.name)

    all_schemas = [
        _pybind_schema_to_native_schema(pybind_schema)
        for pybind_schema in torch._C._jit_get_schemas_for_operator(qualified_opname)
    ]

    # skip the schema that we can not be parsed by torchgen
    all_schemas = [schema for schema in all_schemas if schema is not None]

    group_by_signature: Dict[str, Dict[SchemaKind, FunctionSchema]] = {}

    for schema in all_schemas:
        signature = schema.signature()
        # override the return type to an empty tuple. Otherwise,  for ops like
        # aten.slice.Tensor_out that returns a Tensor list,
        # the signature of the schema does not match the one for the functional
        # op aten.slice.Tensor because of different return type.
        # Schema for aten.slice.Tensor_out:
        #   split.Tensor_out(Tensor(a -> *) self, int split_size, int dim=0, *, Tensor(a!)[] out) -> ()
        # Schema for aten.slice.Tensor
        #   split.Tensor(Tensor(a -> *) self, int split_size, int dim=0) -> Tensor(a)[]
        # The reason of the above inconsistency is explained in: https://github.com/pytorch/pytorch/pull/76049
        signature = dataclasses.replace(signature, returns=())

        kind = schema.kind()
        # pyre-fixme[6]: For 1st argument expected `str` but got `FunctionSchema`.
        group_by_kind = group_by_signature.setdefault(signature, {})
        assert (
            kind not in group_by_kind
        ), f"Schema of kind {kind} already exist for {schema}"
        group_by_kind[kind] = schema

    # add all the functional op -> out variant op pairs to the cache
    for group_by_kind in group_by_signature.values():
        func_op_schema = group_by_kind.get(SchemaKind.functional)
        out_var_schema = group_by_kind.get(SchemaKind.out)
        mutable_op_schema = group_by_kind.get(SchemaKind.mutable)
        scratch_schema = group_by_kind.get(SchemaKind.scratch)

        # update the map even if out_var_schema is None to cache the negative
        # case
        if func_op_schema:
            _func_to_out_variant_map[schema_to_opoverload(func_op_schema)] = (
                schema_to_opoverload(out_var_schema) if out_var_schema else None
            )
            # out variant schema missing from group_by_kind
            if out_var_schema is None:
                # find the out variant with a schema different than the functional variant
                mismatched_out_schema: Optional[FunctionSchema] = next(
                    (s for s in all_schemas if s.kind() == SchemaKind.out), None
                )
                _schema_mismatch_map[schema_to_opoverload(func_op_schema)] = (
                    mismatched_out_schema
                )

        # update hte map even if scratch_schema is None to cache the negative
        # case
        if out_var_schema:
            _out_variant_to_scratch_map[schema_to_opoverload(out_var_schema)] = (
                schema_to_opoverload(scratch_schema) if scratch_schema else None
            )
        if mutable_op_schema:
            _mutable_to_out_variant_map[schema_to_opoverload(mutable_op_schema)] = (
                schema_to_opoverload(out_var_schema) if out_var_schema else None
            )


def to_out_variant(op_overload: OpOverload) -> Tuple[OpOverload, Tuple[str]]:
    r"""
    Convert the passed in OpOverload to its out variant. Raise an exception if
    on return the op_overload is not guaranteed to be an out variant.

    If a conversion is found, return the out variant OpOverload alongwith the name of out
    arguments.
    """
    schema = _get_overload_schema(op_overload)
    if schema.is_out_fn():  # pyre-ignore
        return op_overload, get_out_args_from_schema(schema)  # pyre-ignore[6]

    # should be a functionalish op here
    assert (
        schema.kind() == SchemaKind.functional  # pyre-ignore[16]
        or schema.kind() == SchemaKind.mutable
    ), f"Expect a functionalish op, but get {schema.kind()} {schema}"

    if (
        op_overload not in _func_to_out_variant_map
        and op_overload not in _mutable_to_out_variant_map
    ):
        # setup out_var
        set_mapping_for_op(op_overload)

    if op_overload in _mutable_to_out_variant_map:
        out_var = _mutable_to_out_variant_map[op_overload]
    else:
        out_var = _func_to_out_variant_map.get(op_overload)

    if not out_var:
        msg = f"Missing out variant for functional op: {schema} . Make sure you have loaded your custom operator library for compiler. E.g., custom_ops_generated_lib"
        if op_overload in _schema_mismatch_map:
            if _schema_mismatch_map[op_overload]:
                msg += (
                    f"\nFound an out variant for operator name {op_overload.name()} but its schema mismatched with functional op."
                    f"\nfunctional op schema:\t{schema}"
                    f"\nout variant op schema:\t{_schema_mismatch_map[op_overload]}"
                )
        raise RuntimeError(msg)

    return out_var, get_out_args_from_opoverload(out_var)


def to_scratch_op(op_overload: OpOverload) -> Optional[OpOverload]:
    schema = _get_overload_schema(op_overload)

    # If the op is not an out variant, then we must have ignored some failure in to_out_var
    # pass. Return immediately rather than throwing an exception since the user must have ignores
    # errors for some reason (e.g. desigin some special unit tests, or unblock new
    # use cases).
    if schema.kind() != SchemaKind.out:  # pyre-ignore
        logging.debug(f"Expect an out variant op as input, got: {schema.kind()}")
        return None

    if op_overload not in _out_variant_to_scratch_map:
        set_mapping_for_op(op_overload)
    scratch_op = _out_variant_to_scratch_map.get(op_overload)

    # scratch_op can be None
    return scratch_op


def is_out_variant(qualified_opname: str, overload: str) -> bool:
    op_overload = get_op_overload(qualified_opname, overload)
    schema = _get_overload_schema(op_overload)
    if schema is None:
        return False
    return schema.is_out_fn()


def is_inplace_variant(qualified_opname: str, overload: str) -> bool:
    op_overload = get_op_overload(qualified_opname, overload)
    schema = _get_overload_schema(op_overload)
    if schema is None:
        return False
    return schema.kind() == SchemaKind.inplace
