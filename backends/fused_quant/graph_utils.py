# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Optional

import torch
from executorch.backends.fused_quant.ops import QuantParamsStruct
from executorch.backends.transforms.permute_pass_utils import get_arg
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from torch import fx
from torch._export.utils import _detect_fake_mode_from_gm
from torch._ops import OpOverload
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.export import ExportedProgram
from torch.export.graph_signature import (
    ExportGraphSignature,
    InputKind,
    InputSpec,
    TensorArgument,
)
from torch.fx import map_arg


_OpOverload = EdgeOpOverload | OpOverload
_QPARAM_FIELDS = ("scale", "zero_point", "dtype", "quant_min", "quant_max")


def split_fused_arg_names(target: _OpOverload) -> tuple[list[str], list[str]]:
    """Split a fused op's schema into quantized inputs and extra arguments."""
    names = [argument.name for argument in target._schema.arguments]
    try:
        first_qparam = next(
            index for index, name in enumerate(names) if name.endswith("_scale")
        )
    except StopIteration as error:
        raise ValueError(f"Fused op {target} has no qparam blocks") from error

    input_names = names[:first_qparam]
    cursor = first_qparam
    for prefix in (*input_names, "out"):
        expected = [f"{prefix}_{field}" for field in _QPARAM_FIELDS]
        actual = names[cursor : cursor + len(_QPARAM_FIELDS)]
        if actual != expected:
            raise ValueError(
                f"Fused op {target} does not follow the qparam argument convention: "
                f"expected {expected}, got {actual}"
            )
        cursor += len(_QPARAM_FIELDS)
    return input_names, names[cursor:]


def is_quantize_node(node: fx.Node) -> bool:
    return node.op == "call_function" and node.target in (
        torch.ops.quantized_decomposed.quantize_per_tensor.default,
        torch.ops.quantized_decomposed.quantize_per_channel.default,
        torch.ops.torchao.quantize_affine.default,
    )


def is_dequantize_node(node: fx.Node) -> bool:
    return node.op == "call_function" and node.target in (
        torch.ops.quantized_decomposed.dequantize_per_tensor.default,
        torch.ops.quantized_decomposed.dequantize_per_channel.default,
        torch.ops.torchao.dequantize_affine.default,
    )


def is_affine_quant_node(node: fx.Node) -> bool:
    """Check if a node is a torchao affine quantize/dequantize op.

    Affine nodes carry an explicit ``block_size`` and a full-rank scale/zero_point
    whose shape already encodes the block layout -- exactly QuantParamsStruct's
    convention, covering per-tensor/channel/group/blockwise uniformly with no axis.
    """
    return node.target in (
        torch.ops.torchao.quantize_affine.default,
        torch.ops.torchao.dequantize_affine.default,
    )


def is_per_tensor_quant_node(node: fx.Node) -> bool:
    """Check if a quantize/dequantize node is per-tensor."""
    return node.target in (
        torch.ops.quantized_decomposed.quantize_per_tensor.default,
        torch.ops.quantized_decomposed.dequantize_per_tensor.default,
    )


def is_per_channel_quant_node(node: fx.Node) -> bool:
    """Check if a quantize/dequantize node is per-channel."""
    return node.target in (
        torch.ops.quantized_decomposed.quantize_per_channel.default,
        torch.ops.quantized_decomposed.dequantize_per_channel.default,
    )


def get_qparams_from_node(
    node: fx.Node, prefix: str
) -> Optional[QuantParamsStruct[fx.Node]]:
    """Extract QuantParamsStruct from a node's flat args by prefix.

    Returns None if both scale and zero_point are None (unquantized).
    Raises ValueError if only one of scale/zero_point is None.
    """
    scale = get_arg(node, f"{prefix}_scale", Optional[fx.Node])
    zero_point = get_arg(node, f"{prefix}_zero_point", Optional[fx.Node])
    if scale is None and zero_point is None:
        return None
    if (scale is None) != (zero_point is None):
        raise ValueError(
            f"{prefix}_scale and {prefix}_zero_point must both be None or both be "
            f"provided on node {node.name}"
        )
    assert scale is not None and zero_point is not None
    return QuantParamsStruct(
        scale=scale,
        zero_point=zero_point,
        dtype=get_arg(node, f"{prefix}_dtype", torch.dtype),
        quant_min=get_arg(node, f"{prefix}_quant_min", int),
        quant_max=get_arg(node, f"{prefix}_quant_max", int),
    )


def get_qparams_flat(node: fx.Node, prefix: str) -> list[torch.fx.node.Argument]:
    """Get the 5 flat qparams values from a node for pass-through to another op."""
    return [
        get_arg(node, f"{prefix}_scale"),
        get_arg(node, f"{prefix}_zero_point"),
        get_arg(node, f"{prefix}_dtype"),
        get_arg(node, f"{prefix}_quant_min"),
        get_arg(node, f"{prefix}_quant_max"),
    ]


def compute_meta_val(node: fx.Node) -> object:
    """Compute the ``meta['val']`` for a freshly created (or just-mutated)
    ``call_function`` node by executing its target on its inputs' fake vals under
    the graph's fake mode.

    Handles single-output, multi-output (a tuple of fake tensors), and
    literal-only ops (e.g. ``aten.full``, whose result is faked because it runs
    under the fake mode) uniformly. Callers assign the return to
    ``node.meta['val']``.
    """
    gm = node.graph.owning_module
    assert gm is not None, "node's graph has no owning module"
    fake_mode = _detect_fake_mode_from_gm(gm)
    assert fake_mode is not None, "graph has no fake tensor mode"
    target = node.target
    assert callable(target), f"expected a callable target, got {target}"
    args = map_arg(node.args, lambda n: n.meta["val"])
    kwargs = map_arg(node.kwargs, lambda n: n.meta["val"])
    with fake_mode:
        return target(*args, **kwargs)


def get_constant(ep: ExportedProgram, node: fx.Node) -> Optional[torch.Tensor]:
    """Get the constant tensor value backing a placeholder node.

    Handles parameters (state_dict), persistent buffers (state_dict),
    non-persistent buffers (constants), and lifted tensor constants (constants).
    Returns None if the node is not a constant placeholder.
    """
    if node.op != "placeholder":
        return None

    sig = ep.graph_signature
    name = node.name

    if name in sig.inputs_to_parameters:
        fqn = sig.inputs_to_parameters[name]
        return ep.state_dict[fqn].data

    if name in sig.inputs_to_buffers:
        fqn = sig.inputs_to_buffers[name]
        if fqn in sig.non_persistent_buffers:
            return ep.constants[fqn]
        return ep.state_dict[fqn]

    if name in sig.inputs_to_lifted_tensor_constants:
        fqn = sig.inputs_to_lifted_tensor_constants[name]
        return ep.constants[fqn]

    return None


def get_scale(ep: ExportedProgram, qp: "QuantParamsStruct[fx.Node]") -> float:
    """Read a per-tensor scale value from its qparams.

    Per-tensor qparams are 0-dim lifted constants (placeholders) post-export;
    resolve the backing tensor and return its Python scalar.
    """
    value = get_constant(ep, qp.scale)
    assert value is not None, (
        f"expected a lifted-constant qparam for {qp.scale.name}, got op={qp.scale.op}"
    )
    return float(value.item())


def get_zero_point(ep: ExportedProgram, qp: "QuantParamsStruct[fx.Node]") -> int:
    """Read a per-tensor zero_point value from its qparams. See get_scale."""
    value = get_constant(ep, qp.zero_point)
    assert value is not None, (
        f"expected a lifted-constant qparam for {qp.zero_point.name}, "
        f"got op={qp.zero_point.op}"
    )
    return int(value.item())


def set_constant(ep: ExportedProgram, node: fx.Node, value: torch.Tensor) -> None:
    """Set the constant tensor value for a placeholder node.

    Writes to the correct backing store (state_dict for parameters and
    persistent buffers, constants for non-persistent buffers and lifted
    tensor constants). Parameters are wrapped in torch.nn.Parameter.

    Raises ValueError if the node is not a constant placeholder.
    """
    if node.op != "placeholder":
        raise ValueError(f"Node {node.name} is not a placeholder")

    sig = ep.graph_signature
    name = node.name

    if name in sig.inputs_to_parameters:
        fqn = sig.inputs_to_parameters[name]
        ep.state_dict[fqn] = torch.nn.Parameter(value)
        return

    if name in sig.inputs_to_buffers:
        fqn = sig.inputs_to_buffers[name]
        if fqn in sig.non_persistent_buffers:
            ep.constants[fqn] = value
        else:
            ep.state_dict[fqn] = value
        return

    if name in sig.inputs_to_lifted_tensor_constants:
        fqn = sig.inputs_to_lifted_tensor_constants[name]
        ep.constants[fqn] = value
        return

    raise ValueError(
        f"Node {node.name} is not a parameter, buffer, or lifted tensor constant"
    )


def get_input_kind(ep: ExportedProgram, node: fx.Node) -> Optional[InputKind]:
    """Return the InputKind of a placeholder node, or None if not found."""
    if node.op != "placeholder":
        return None

    sig = ep.graph_signature
    name = node.name

    if name in sig.inputs_to_parameters:
        return InputKind.PARAMETER
    if name in sig.inputs_to_buffers:
        return InputKind.BUFFER
    if name in sig.inputs_to_lifted_tensor_constants:
        return InputKind.CONSTANT_TENSOR

    return None


def get_fqn(ep: ExportedProgram, node: fx.Node) -> Optional[str]:
    """Return the fully-qualified name backing a constant placeholder, or None.

    The fqn is the key into ``ep.state_dict`` / ``ep.constants`` and the
    ``InputSpec.target`` -- the un-prefixed logical name (node ``p_a_b`` -> fqn
    ``a.b``), i.e. the inverse of the ``c_``/``b_``/``p_`` node-name prefixing.
    """
    if node.op != "placeholder":
        return None

    sig = ep.graph_signature
    name = node.name

    if name in sig.inputs_to_parameters:
        return sig.inputs_to_parameters[name]
    if name in sig.inputs_to_buffers:
        return sig.inputs_to_buffers[name]
    if name in sig.inputs_to_lifted_tensor_constants:
        return sig.inputs_to_lifted_tensor_constants[name]

    return None


# The node-name prefix export gives each lifted input by kind. Mirroring it keeps
# add_constant's placeholders indistinguishable from export's own, and — because
# every lifted node then shares its kind's prefix — makes node-name uniqueness
# imply fqn uniqueness (fqn == node name minus prefix).
_INPUT_KIND_PREFIX: dict[InputKind, str] = {
    InputKind.PARAMETER: "p_",
    InputKind.BUFFER: "b_",
    InputKind.CONSTANT_TENSOR: "c_",
}


def _fake_mode_from_node(node: fx.Node) -> FakeTensorMode:
    """Pull the shared ``FakeTensorMode`` off a node's ``meta['val']``.

    A single-output node carries a ``FakeTensor``; a multi-output op (e.g.
    ``native_layer_norm``) carries a tuple/list of them. The fake mode is the
    same graph-wide, so grab it from the first fake tensor either way.
    """
    val = node.meta["val"]
    if isinstance(val, (tuple, list)):
        val = next(v for v in val if isinstance(v, FakeTensor))
    return val.fake_mode


def add_constant(
    ep: ExportedProgram,
    name: str,
    tensor: torch.Tensor,
    before_node: fx.Node,
    kind: InputKind,
) -> fx.Node:
    """Add a new constant to the ExportedProgram and create a placeholder node for it.

    ``name`` must be a logical (unprefixed) base -- add_constant owns the
    prefixing. The placeholder node name is the export-style ``<prefix><fqn>``
    (``c_`` constant, ``b_`` buffer, ``p_`` parameter) and the fqn -- the key into
    ep.constants / the ``InputSpec.target`` -- is that node name with the prefix
    stripped back off.

    This keeps node names and fqns in the same bijection export maintains
    (``node == prefix + fqn``), so ``graph.placeholder``'s node-name
    uniquification already implies fqn uniqueness for the kind: a free
    ``<prefix><fqn>`` node name means a free fqn. The assert guards the one case
    that escapes it -- a cross-kind fqn clash, where the same fqn string could
    back both ep.constants and ep.state_dict (separate dicts, different prefixes)
    -- and any future violation of the prefix invariant, failing loudly instead
    of silently sharing a backing entry (which constant_prop later turns into a
    KeyError in get_lifted_tensor_constant). See test_graph_utils.
    """
    graph = ep.graph_module.graph
    last_placeholder = graph.find_nodes(op="placeholder")[-1]

    # add_constant only lifts params/buffers/constants, which all carry a nonempty
    # export prefix; reject any other kind rather than silently prefixing with "".
    assert kind in _INPUT_KIND_PREFIX, (
        f"add_constant supports {set(_INPUT_KIND_PREFIX)}, got {kind}"
    )
    prefix = _INPUT_KIND_PREFIX[kind]
    # ``name`` must be logical: add_constant owns the prefixing, so a name already
    # carrying this kind's prefix is a caller bug (it would double-prefix).
    assert not name.startswith(prefix), (
        f"add_constant expects a logical (unprefixed) name for {kind}, got {name!r}"
    )

    with graph.inserting_after(last_placeholder):
        placeholder = graph.placeholder(prefix + name)
        placeholder.target = placeholder.name
        fake_mode = _fake_mode_from_node(before_node)
        placeholder.meta["val"] = fake_mode.from_tensor(tensor, static_shapes=True)

    fqn = placeholder.name[len(prefix) :]
    assert fqn not in ep.constants and fqn not in ep.state_dict, (
        f"add_constant: fqn {fqn!r} already backed; prefix invariant violated"
    )

    new_spec = InputSpec(
        kind=kind,
        arg=TensorArgument(name=placeholder.name),
        target=fqn,
        persistent=True,
    )
    ep._graph_signature = ExportGraphSignature(
        input_specs=list(ep.graph_signature.input_specs) + [new_spec],
        output_specs=list(ep.graph_signature.output_specs),
    )

    set_constant(ep, placeholder, tensor)

    return placeholder
