# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
APIs to help lowering edge dialect ops to other dialects.
"""
import dataclasses
import logging
from typing import List, Optional

import torch

from executorch.exir.operator.convert import _pybind_schema_to_native_schema
from torch._ops import OpOverload, OpOverloadPacket
from torchgen.model import FunctionSchema, SchemaKind


def get_torch_op_overload(
    namespace: str, opname: str, overload: Optional[str]
) -> torch._ops.OpOverload:
    packet: OpOverloadPacket = getattr(getattr(torch.ops, namespace), opname)
    if overload:
        return getattr(packet, overload)
    else:
        return packet.default


def get_callable(name) -> torch._ops.OpOverload:
    main, suffix = name.split(".")
    return get_torch_op_overload("aten", main, suffix)


def to_variant(op: OpOverload, variant: SchemaKind) -> OpOverload:
    """Given an operator overload, return its corresponding variant. Currently
    only supports functional variant and out variant.
    Argument:
        op (OpOverload): operator overload instance.
        variant (SchemaKind): the variant we are looking for.
    Returns:
        OpOverload: The matched variant operator.
    Example:
        torch.ops.aten.add.Tensor, SchemaKind.out -> torch.ops.aten.add.out
        torch.ops.aten.add.out, SchemaKind.functional -> torch.ops.aten.add.Tensor
    """
    assert (
        variant == SchemaKind.functional or variant == SchemaKind.out
    ), f"Only support out variant and functional variant, got {variant}"
    # first check if the current operator is the target variant
    native_schema: Optional[FunctionSchema] = _pybind_schema_to_native_schema(
        op._schema
    )
    assert (
        native_schema is not None
    ), f"Schema: {op._schema} cannot be converted to torch.FunctionSchema"

    # get all overloads
    torch_packet = getattr(
        getattr(torch.ops, op.namespace), op._schema.name.split("::")[1]
    )
    schemas: List[torch._C.FunctionSchema] = [
        getattr(torch_packet, o)._schema
        for o in torch._C._jit_get_operation(op._schema.name)[1]
    ]
    # compare the signature of out variant overload with the signature of the original overload
    signature = dataclasses.replace(native_schema.signature(), returns=())
    for schema in schemas:
        native_s: Optional[FunctionSchema] = _pybind_schema_to_native_schema(schema)
        if native_s is None:
            logging.warning(
                f"Schema: {schema} cannot be converted to torch.FunctionSchema"
            )
            continue
        if (
            native_s.kind() == variant
            and dataclasses.replace(native_s.signature(), returns=()) == signature
        ):
            op_variant = get_torch_op_overload(
                op.namespace, schema.name.split("::")[1], schema.overload_name
            )
            return op_variant
    raise RuntimeError(
        f"{variant} variant of operator {op.name()} can't be found. We've found the schemas of all the overloads: {[str(s) for s in schemas]}"
    )
