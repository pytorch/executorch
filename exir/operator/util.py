# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401
from torchgen.model import FunctionSchema, SchemaKind
from torchgen.native_function_generation import (
    functional_to_out_signature,
    mutable_to_out_signature,
    self_to_out_signature,
)
from torchgen.utils import NamespaceHelper


def gen_out_variant_schema(func_op_schema: str) -> str:
    """
    Generate schema for the out= variant of a given functional operator schema.
    """
    # Parse the operator schema
    namespace_helper = NamespaceHelper.from_namespaced_entity(
        namespaced_entity=func_op_schema, max_level=1
    )
    func = FunctionSchema.parse(namespace_helper.entity_name)

    namespace = namespace_helper.get_cpp_namespace(default="")
    # Convert it to out variant schema
    if func.kind() == SchemaKind.inplace:
        schema = str(self_to_out_signature(func))
    elif func.kind() == SchemaKind.functional:
        schema = str(functional_to_out_signature(func))
    elif func.kind() == SchemaKind.mutable:
        schema = str(mutable_to_out_signature(func))
    elif func.kind() == SchemaKind.out:
        schema = str(func)
    else:
        raise RuntimeError(f"SchemaKind: {func.kind()} is not supported")

    return f"{namespace}::{schema}" if namespace else schema


# TODO: move to torchao
_QUANT_PRIMITIVES = [
    torch.ops.quantized_decomposed.dequantize_per_channel.default,
    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
    torch.ops.quantized_decomposed.dequantize_per_tensor.tensor,
    torch.ops.quantized_decomposed.convert_element_type.no_fuse,
    torch.ops.quantized_decomposed.quantize_per_tensor.default,
    torch.ops.quantized_decomposed.quantize_per_tensor.tensor,
    torch.ops.quantized_decomposed.quantize_per_channel.default,
    torch.ops.quantized_decomposed.choose_qparams.tensor,
]
try:
    import torchao  # noqa: F401

    _QUANT_PRIMITIVES.extend(
        [
            torch.ops.torchao.dequantize_affine.default,
            torch.ops.torchao.quantize_affine.default,
            torch.ops.torchao.choose_qparams_affine.default,
        ]
    )
except ImportError:
    pass
