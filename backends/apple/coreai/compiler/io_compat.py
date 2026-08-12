# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Delegate-boundary compatibility checks for the Core AI backend.

ExecuTorch calls a delegate with plain tensors whose dtypes/shapes come from the
edge subgraph boundary.  The converted ``.aimodel`` must declare I/O that matches
what ExecuTorch will feed/read, or the runtime will mismatch.  :func:`
assert_io_compatible` compares the two *before* compilation and fails loudly.
"""

import logging
from typing import Any, List, Sequence, Tuple

import torch
from executorch.backends.apple.coreai.compiler.constants import MAIN_ENTRYPOINT
from torch.export.exported_program import ExportedProgram
from torch.export.graph_signature import InputKind

logger = logging.getLogger(__name__)

# coreai's element-type spelling for each torch dtype (from coreai_torch's
# _type_mapping: signed ints -> siN, unsigned -> uiN, bool -> i1, floats -> fN).
# These are the un-narrowed spellings, asserted exactly. coreai narrows
# int64->si32 and float64->f32 in get_tensor_type, so those two dtypes surface
# here as mismatches by design (that is the signal).
_TORCH_TO_COREAI_ELT = {
    torch.float16: "f16",
    torch.float32: "f32",
    torch.float64: "f64",
    torch.bfloat16: "bf16",
    torch.bool: "i1",
    torch.int8: "si8",
    torch.uint8: "ui8",
    torch.int16: "si16",
    torch.uint16: "ui16",
    torch.int32: "si32",
    torch.uint32: "ui32",
    torch.int64: "si64",
}


def _coreai_io(program, entrypoint: str = MAIN_ENTRYPOINT):
    """Core AI graph I/O as lists of (element_type_str, shape_or_None).

    Ranked tensors -> (element_type_str, shape_tuple).  Any non-tensor coreai
    type is represented as (str(type), None) so the comparison can flag it,
    rather than assuming a tensor and crashing on ``.element_type`` / ``.shape``.
    """
    from coreai._compiler.ir import RankedTensorType

    func_type = program.get_graph(entrypoint).function_type.value

    def _describe(t):
        if RankedTensorType.isinstance(t):
            return (str(t.element_type), tuple(t.shape))
        return (str(t), None)

    inputs = [_describe(t) for t in func_type.inputs]
    outputs = [_describe(t) for t in func_type.results]
    return inputs, outputs


def _edge_io(edge_program: ExportedProgram):
    """Edge subgraph I/O as lists of (dtype_or_typename, shape_or_None).

    Tensors -> (torch_dtype, shape_tuple).  Non-tensor I/O (symint / const
    int/float) -> (type_name_str, None), kept as entries so both sides stay
    positionally aligned with :func:`_coreai_io` (which also emits non-tensor
    entries).

    Inputs walk ``graph_signature.input_specs``, which is ordered like the
    placeholders. Mutated buffers are included alongside the user inputs,
    mirroring ``TorchConverter._register_io``: a mutation is passed in and
    handed back, so coreai gives it a graph argument.
    """
    placeholders = {
        n.name: n for n in edge_program.graph.nodes if n.op == "placeholder"
    }
    signature = edge_program.graph_signature
    mutated = set(signature.buffers_to_mutate.values())

    inputs = []
    for spec in signature.input_specs:
        is_mutated_buffer = spec.kind == InputKind.BUFFER and spec.target in mutated
        if spec.kind != InputKind.USER_INPUT and not is_mutated_buffer:
            continue
        node = placeholders.get(getattr(spec.arg, "name", None))
        # A non-tensor input carries its literal value on the spec instead.
        val = node.meta.get("val") if node is not None else spec.arg.value
        if hasattr(val, "dtype"):
            inputs.append((val.dtype, tuple(val.shape)))
        else:
            inputs.append((type(val).__name__, None))
    outputs = []
    for arg in edge_program.graph.output_node().args[0]:
        val = arg.meta.get("val") if hasattr(arg, "meta") else arg
        if hasattr(val, "dtype"):
            outputs.append((val.dtype, tuple(val.shape)))
        else:
            outputs.append((type(val).__name__, None))
    return inputs, outputs


def io_mismatches(
    coreai_io: Sequence[Tuple[str, Tuple[Any, ...]]],
    edge_io: Sequence[Tuple[Any, Tuple[Any, ...]]],
    kind: str,
) -> List[str]:
    """Return human-readable mismatch messages between coreai and edge I/O.

    Compares count, per-tensor dtype (all mapped types: floats and ints), rank,
    and static (concrete) dims. Symbolic (dynamic) dims are skipped: coreai is
    specialized to a concrete size while the edge dim stays symbolic.
    """
    if len(coreai_io) != len(edge_io):
        return [
            f"{kind} count mismatch: .aimodel has {len(coreai_io)}, "
            f"ExecuTorch feeds {len(edge_io)}"
        ]
    errors: List[str] = []
    for i, ((elt, cshape), (dtype, eshape)) in enumerate(zip(coreai_io, edge_io)):
        coreai_nontensor = cshape is None
        edge_nontensor = eshape is None
        if coreai_nontensor or edge_nontensor:
            # At least one side is non-tensor; only a class mismatch matters.
            if coreai_nontensor != edge_nontensor:
                errors.append(
                    f"{kind} {i}: tensor/non-tensor mismatch "
                    f"(.aimodel={elt}, ExecuTorch feeds {dtype})"
                )
            continue
        expected = _TORCH_TO_COREAI_ELT.get(dtype)
        if expected is not None and elt != expected:
            errors.append(
                f"{kind} {i}: dtype mismatch (.aimodel={elt}, ExecuTorch feeds "
                f"{dtype}, expected {expected})"
            )
        if len(cshape) != len(eshape):
            errors.append(
                f"{kind} {i}: rank mismatch (.aimodel={cshape}, "
                f"ExecuTorch={eshape})"
            )
            continue
        for d, (cdim, edim) in enumerate(zip(cshape, eshape)):
            if isinstance(edim, int) and cdim != edim:
                errors.append(
                    f"{kind} {i} dim {d}: static shape mismatch "
                    f"(.aimodel={cdim}, ExecuTorch={edim})"
                )
    return errors


def assert_io_compatible(program, edge_program: ExportedProgram) -> None:
    """Raise if the ``.aimodel`` boundary I/O is incompatible with ExecuTorch."""
    coreai_in, coreai_out = _coreai_io(program)
    edge_in, edge_out = _edge_io(edge_program)
    logger.info(
        "Core AI delegate boundary I/O:\n"
        "  inputs:  .aimodel=%s\n"
        "           ExecuTorch=%s\n"
        "  outputs: .aimodel=%s\n"
        "           ExecuTorch=%s",
        coreai_in,
        edge_in,
        coreai_out,
        edge_out,
    )
    errors = io_mismatches(coreai_in, edge_in, "input") + io_mismatches(
        coreai_out, edge_out, "output"
    )
    if errors:
        raise ValueError(
            "Core AI delegate boundary is incompatible with ExecuTorch:\n  "
            + "\n  ".join(errors)
        )
