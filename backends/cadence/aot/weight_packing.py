# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Sub-byte weight packing for the Cadence backend.

PT2E has no sub-byte storage dtype, so a 6-bit weight is normally an ``int8``
tensor with its range clamped to ``[-32, 31]``. That measures the accuracy of
the reduced precision faithfully but saves no memory. This module physically
packs those values so the saving is real.

Layout
------
Packing is per output-channel **row**, so a ``[out_dim, in_dim]`` weight becomes
``[out_dim, ceil(in_dim * bits / 8)]``. Three things fall out of that choice:

* ``out_dim`` survives as ``weight.size(0)``, so shape inference and the
  existing fully-connected meta kernels keep working unchanged.
* The dtype stays ``int8`` (the bytes are a container, not a value), so compile
  time type dispatch keeps matching ``(int8, int8)``.
* Rows stay byte aligned, which is what makes the padding zero for a model whose
  row lengths are all multiples of the group size.

The 6-bit layout is a planar 4+2 split, chosen over torchao's UINT6 so the
interchangeable, including with its vectorized counterpart:
``pytorch/ao/torchao/csrc/cpu/torch_free_kernels/fallback/bitpacking/uint6.h``

    p0 low 6 bits = u0,  p0 high 2 bits = u3 bits 1:0
    p1 low 6 bits = u1,  p1 high 2 bits = u3 bits 3:2
    p2 low 6 bits = u2,  p2 high 2 bits = u3 bits 5:4

Signed values are stored as offset binary (``+2**(bits-1)``), also matching
torchao.

The remap
---------
Given only the packed blob, ``in_dim`` and the bit width, any logical ``(row,
col)`` is recoverable with shifts and masks and no division. For 6 bits:

    row_stride = in_dim * 6 // 8
    g, p       = k >> 2, k & 3
    base       = r * row_stride + 3 * g
    p < 3  ->  packed[base + p] & 0x3F
    p == 3 ->  bits gathered from the top of all three bytes

Because a matmul walks ``k`` for a fixed output channel, and that is the axis we
packed along, the kernel streams three bytes and emits four weights. There is no
indexed gather, which is the whole reason for packing along the innermost loop
axis.
"""

import logging

import torch
from executorch.backends.cadence.aot.quantizer.pattern_utils import (
    add_constant_placeholder,
    EXPORTED_PROGRAM_META_KEY,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.passes.constant_prop_pass import constant_prop_pass
from torch._guards import detect_fake_mode
from torch.export.exported_program import ExportedProgram

logger: logging.Logger = logging.getLogger(__name__)

# Number of logical values that pack into a whole number of bytes, per width.
# group_values * bits == group_bytes * 8
_GROUPS: dict[int, tuple[int, int]] = {
    4: (2, 1),  # two nibbles per byte
    6: (4, 3),  # planar 4+2: 4 values per 3 bytes, as two bit-planes
    8: (1, 1),  # identity
}

SUPPORTED_BITS: tuple[int, ...] = tuple(sorted(_GROUPS))


def group_shape(bits: int) -> tuple[int, int]:
    """(values per group, bytes per group) for ``bits``."""
    if bits not in _GROUPS:
        raise ValueError(
            f"unsupported weight bit width {bits}; supported: {SUPPORTED_BITS}"
        )
    return _GROUPS[bits]


def packed_row_bytes(in_dim: int, bits: int) -> int:
    """Bytes one packed row occupies. Raises if the row would need padding."""
    values, nbytes = group_shape(bits)
    if in_dim % values:
        raise ValueError(
            f"in_dim {in_dim} is not a multiple of the {bits}-bit group size "
            f"{values}; row-aligned packing would need padding"
        )
    return in_dim // values * nbytes


def offset(bits: int) -> int:
    """Offset-binary bias, matching torchao's signed convention."""
    return 1 << (bits - 1)


def pack_rows(weight: torch.Tensor, bits: int) -> torch.Tensor:
    """``[out_dim, in_dim]`` int8 -> ``[out_dim, packed_row_bytes]`` int8."""
    if weight.dim() != 2:
        raise ValueError(f"expected a 2D weight, got {weight.dim()}D")
    if weight.dtype != torch.int8:
        raise ValueError(f"expected int8 weight, got {weight.dtype}")
    if bits == 8:
        return weight.contiguous()

    out_dim, in_dim = weight.shape
    row_bytes = packed_row_bytes(in_dim, bits)
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    if weight.numel() and (int(weight.min()) < lo or int(weight.max()) > hi):
        raise ValueError(
            f"weight range [{int(weight.min())}, {int(weight.max())}] does not "
            f"fit {bits} bits [{lo}, {hi}]"
        )

    values, _ = group_shape(bits)
    u = (weight.to(torch.int16) + offset(bits)).to(torch.uint8)
    g = u.reshape(out_dim, in_dim // values, values)

    if bits == 4:
        # Split, not interleaved: byte k holds value k in the low nibble and
        # value k + in_dim/2 in the high nibble. Masking the low nibbles of a
        # word then yields consecutive values, so the decode is one AND and one
        # shift per word with no de-interleave. This is exactly plane A of the
        # 6-bit layout below.
        u_flat = u.reshape(out_dim, in_dim)
        half = in_dim // 2
        packed = (u_flat[:, :half] & 0x0F) | ((u_flat[:, half:] & 0x0F) << 4)
        packed = packed.reshape(out_dim, row_bytes)
    else:  # bits == 6, planar 4+2
        # Two bit-planes rather than 4-values-in-3-bytes. Same 0.75 B/value,
        # but nothing straddles a byte, so the decode is elementwise masks and
        # shifts on whole words instead of a stride-3 cross-byte gather.
        #
        # Values are paired k with k+H, NOT k with k+1: masking the low nibbles
        # of a word then yields four *consecutive* values, so the decode emits
        # natural order and no side needs to know about lane order.
        u_flat = u.reshape(out_dim, in_dim)
        half, quarter = in_dim // 2, in_dim // 4
        lo = u_flat & 0x0F
        hi = (u_flat >> 4) & 0x03
        plane_a = lo[:, :half] | (lo[:, half:] << 4)
        plane_b = (
            hi[:, 0:quarter]
            | (hi[:, quarter : 2 * quarter] << 2)
            | (hi[:, 2 * quarter : 3 * quarter] << 4)
            | (hi[:, 3 * quarter : 4 * quarter] << 6)
        )
        packed = torch.cat([plane_a, plane_b], dim=-1).reshape(out_dim, row_bytes)

    return packed.view(torch.int8).contiguous()


def unpack_rows(packed: torch.Tensor, in_dim: int, bits: int) -> torch.Tensor:
    """``[out_dim, packed_row_bytes]`` int8 -> ``[out_dim, in_dim]`` int8."""
    if bits == 8:
        return packed.contiguous()
    if packed.dim() != 2:
        raise ValueError(f"expected a 2D packed tensor, got {packed.dim()}D")
    out_dim, row_bytes = packed.shape
    expected = packed_row_bytes(in_dim, bits)
    if row_bytes != expected:
        raise ValueError(
            f"packed row is {row_bytes} B but in_dim {in_dim} at {bits} bits "
            f"needs {expected} B"
        )

    if bits == 4:
        raw = packed.view(torch.uint8)
        u = torch.cat([raw & 0x0F, (raw >> 4) & 0x0F], dim=-1)
    else:  # bits == 6, planar 4+2 (see pack_rows)
        raw = packed.view(torch.uint8)
        half, quarter = in_dim // 2, in_dim // 4
        plane_a = raw[:, :half]
        plane_b = raw[:, half : half + quarter]
        lo = torch.cat([plane_a & 0x0F, (plane_a >> 4) & 0x0F], dim=-1)
        hi = torch.cat(
            [(plane_b >> shift) & 0x03 for shift in (0, 2, 4, 6)], dim=-1
        )
        u = lo | (hi << 4)
    return (u.to(torch.int16) - offset(bits)).to(torch.int8)


def remap(packed: torch.Tensor, in_dim: int, bits: int, row: int, col: int) -> int:
    """Read one logical weight out of the packed blob.

    Scalar mirror of what the kernel does, kept as an executable statement of
    the index map. Only useful for tests; the kernel streams rows instead.
    """
    if bits == 8:
        return int(packed[row, col])
    flat = packed.view(torch.uint8).reshape(-1)
    row_stride = packed_row_bytes(in_dim, bits)
    row_base = row * row_stride

    if bits == 4:  # split, see pack_rows
        half = in_dim // 2
        byte = int(flat[row_base + (col if col < half else col - half)])
        v = byte & 0x0F if col < half else (byte >> 4) & 0x0F
    else:  # bits == 6, planar 4+2 (see pack_rows)
        half, quarter = in_dim // 2, in_dim // 4
        a = int(flat[row_base + (col if col < half else col - half)])
        nibble = a & 0x0F if col < half else (a >> 4) & 0x0F
        b = int(flat[row_base + half + (col % quarter)])
        v = nibble | (((b >> (2 * (col // quarter))) & 0x03) << 4)
    return v - offset(bits)


def _write_constant(
    ep: ExportedProgram, node: torch.fx.Node, value: torch.Tensor
) -> bool:
    """Replace the constant backing `node` with `value`. True if it was written.

    Kept local rather than shared with the QAT folding pass, which needs the
    same three placeholder kinds: importing that pass here would drag the whole
    quantizer into a module the reference implementations depend on.
    """
    sig = ep.graph_signature
    if node.name in sig.inputs_to_parameters:
        ep.state_dict[sig.inputs_to_parameters[node.name]] = torch.nn.Parameter(
            value, requires_grad=False
        )
    elif node.name in sig.inputs_to_buffers:
        ep.state_dict[sig.inputs_to_buffers[node.name]] = value
    elif node.name in sig.inputs_to_lifted_tensor_constants:
        ep.constants[sig.inputs_to_lifted_tensor_constants[node.name]] = value
    else:
        return False

    # The shape changed, so a stale FakeTensor here would make every downstream
    # consumer read the wrong in_dim.
    fake_mode = detect_fake_mode(
        tuple(n.meta["val"] for n in ep.graph.nodes if n.op == "placeholder")
    )
    if fake_mode is not None:
        node.meta["val"] = fake_mode.from_tensor(value, static_shapes=True)
        node.meta["val"].constant = value
    else:
        node.meta["val"] = value
    return True


def _fc_targets() -> tuple[EdgeOpOverload, EdgeOpOverload]:
    """The two fully-connected overloads the transform rewrites.

    `.default` carries tensor qparams and `.per_tensor` scalar ones; the packed
    op only has the tensor form, so scalars are lifted to length-1 constants on
    the way through. Resolved lazily: the edge namespace is only populated once
    ops_registrations has been imported, and this module must stay importable
    on its own.
    """
    packet = exir_ops.edge.cadence.quantized_fully_connected
    return packet.default, packet.per_tensor


def _resolve_weight(ep: ExportedProgram, node: object) -> torch.Tensor | None:
    """The constant tensor backing a weight placeholder, if it is one."""
    if not isinstance(node, torch.fx.Node) or node.op != "placeholder":
        return None
    sig = ep.graph_signature
    if node.name in sig.inputs_to_parameters:
        return ep.state_dict.get(sig.inputs_to_parameters[node.name])
    if node.name in sig.inputs_to_buffers:
        return ep.state_dict.get(sig.inputs_to_buffers[node.name])
    if node.name in sig.inputs_to_lifted_tensor_constants:
        return ep.constants.get(sig.inputs_to_lifted_tensor_constants[node.name])
    return None


def can_pack(weight: torch.Tensor, bits: int) -> bool:
    """Whether `weight` packs at `bits` without losing information."""
    if weight.dim() != 2 or weight.dtype != torch.int8:
        return False
    if weight.shape[1] % group_shape(bits)[0]:
        return False
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return not weight.numel() or (int(weight.min()) >= lo and int(weight.max()) <= hi)


def pack_fully_connected_weights(ep: ExportedProgram, bits: int) -> bool:
    """Pack constant fully-connected weights to `bits` and retarget their nodes.

    Runs after the pass pipeline, on a graph that is otherwise final, because
    packing changes the weight's shape and so has to write through the
    ExportedProgram rather than the GraphModule that passes are handed.

    A layer whose values do not fit `bits`, or whose in_dim is not a multiple of
    the group size, is left alone. Degrading one layer to 8 bits is better than
    failing the compile, and the caller can see which layers packed from the
    return value of `packed_layers`.
    """
    if bits == 8:
        return False
    group_shape(bits)  # reject unsupported widths before touching the graph

    targets = _fc_targets()
    gm = ep.graph_module
    # add_constant_placeholder reads the program out of the graph module.
    gm.meta[EXPORTED_PROGRAM_META_KEY] = ep
    changed = False
    try:
        for node in list(ep.graph.nodes):
            if node.op != "call_function" or node.target not in targets:
                continue
            if _pack_one(ep, gm, node, bits):
                changed = True
    finally:
        gm.meta.pop(EXPORTED_PROGRAM_META_KEY, None)

    if changed:
        ep.graph.eliminate_dead_code()
        gm.recompile()
    return changed


def fold_and_pack_fully_connected_weights(ep: ExportedProgram, bits: int) -> bool:
    """Constant-fold the weight subgraphs, then pack. The entry point to use.

    On a real model the weight is not a plain placeholder: the conv-to-linear
    replacement leaves a view/permute chain between the constant and the
    operator that consumes it, and packing rewrites the constant itself.
    Folding first is what makes conv-derived layers packable at all - without it
    they all decline silently, while a synthetic `nn.Linear` packs fine.

    Only call this once the pass list has finished: folding drops those nodes,
    so nothing downstream may still expect them.
    """
    if bits == 8:
        return False
    constant_prop_pass(ep)
    return pack_fully_connected_weights(ep, bits)


def _pack_one(
    ep: ExportedProgram,
    gm: torch.fx.GraphModule,
    node: torch.fx.Node,
    bits: int,
) -> bool:
    scalar_qparams = node.target is _fc_targets()[1]
    weight_node = node.args[1]
    weight = _resolve_weight(ep, weight_node)
    if weight is None:
        logger.info("%s: weight constant not resolvable", node.name)
        return False
    if not can_pack(weight, bits):
        logger.info(
            "%s: declines to pack at %d bits: shape %s, range [%d, %d]",
            node.name,
            bits,
            tuple(weight.shape),
            int(weight.min()) if weight.numel() else 0,
            int(weight.max()) if weight.numel() else 0,
        )
        return False
    assert isinstance(weight_node, torch.fx.Node)
    # A weight feeding more than this node would be packed once and read twice.
    if any(u is not node for u in weight_node.users):
        return False

    in_dim = weight.shape[1]
    if not _write_constant(ep, weight_node, pack_rows(weight, bits)):
        return False

    src, _, bias, in_zero_point = node.args[0], node.args[1], node.args[2], node.args[3]
    qparams = list(node.args[4:7])
    if scalar_qparams:
        qparams = [
            add_constant_placeholder(
                gm, torch.tensor([q], dtype=torch.int32), node, name
            )
            for q, name in zip(qparams, ("wzp", "out_multiplier", "out_shift"))
        ]
    node.target = exir_ops.edge.cadence.quantized_fully_connected_packed.default
    node.args = (
        src,
        weight_node,
        bias,
        in_dim,
        bits,
        in_zero_point,
        *qparams,
        *node.args[7:],
    )
    return True
