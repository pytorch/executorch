# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Direct native lowering.

Where the ExecuTorch flow is export -> [quantize] -> to_edge -> to_backend -> to_executorch, this
is export -> [quantize] -> to_native, and yields a NativeProgramManager to save as a .ptn. See README.md.

Lowering runs the same to_edge_transform_and_lower pipeline as the ExecuTorch path
(the default native passes plus NativePartitioner), but opts the partitioner into
PTN serialization, so each delegate hands its constants back on NativeDelegateInfo
instead of copying them into a NamedDataStore. They are merged into one
multi-method Program and packed once, on save.

Nothing about lowering is configurable yet: the passes, the partitioner, and the
edge compile config are all owned here, because correctness of the package depends
on them. See to_native for what a future compile_config argument would have to
enforce.

Eligible immutable constants are deduped exactly once, on save, across every method
-- preprocess does no hashing and no packing. That is both cheaper than deduping per
method and more correct, since per-method alias maps do not compose.
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import torch

from executorch.backends.native.preprocess import NativeDelegateInfo
from executorch.backends.native.serialization.graph_serialize import (
    _same_tensor,
    collect_data_keys,
    deserialize_program,
    merge_programs,
    validate_program,
)
from executorch.backends.native.serialization.schema import Program
from executorch.exir._serialize._ptn import write_ptn
from torch.utils import _pytree as pytree

if TYPE_CHECKING:
    from torch.export import ExportedProgram
    from torch.fx import GraphModule


class NativeProgramManager:
    """One or more methods lowered for the native runtime, ready to save.

    The counterpart to ExecutorchProgramManager: ``methods`` and ``save`` mirror it
    deliberately, so the two pipelines read alike. The surface is intentionally
    small, we will expand this later.

    The graph blob is the only source of truth: method names and package-wide
    mutability are read back out of it on demand rather than carried alongside,
    so a manager cannot describe a package it does not hold.

    Constants are held as tensor references returned by lowering. ``save`` may
    normalize device or layout before writing them.
    """

    def __init__(
        self,
        ptg: bytes,
        constants: dict[str, torch.Tensor],
    ) -> None:
        # TODO add args help
        self._ptg = ptg
        self._constants = constants

    @property
    def methods(self) -> set[str]:
        """Names of the methods in the program."""
        return {method.name for method in deserialize_program(self._ptg).methods}

    def save(self, path: str) -> None:
        """Write the program and its constants to ``path`` as a .ptn package."""
        mutable_keys = frozenset(_mutable_data_keys(deserialize_program(self._ptg)))
        write_ptn(path, self._ptg, self._constants, mutable_keys=mutable_keys)


def _check_delegate_outputs(
    output_node: torch.fx.Node,
    delegate_call: torch.fx.Node,
    getitems: list[torch.fx.Node],
    method_name: str,
) -> None:
    """Require every output leaf to be an ordered projection of the delegate."""
    stray = [n.name for n in getitems if n.args[0] is not delegate_call]
    if stray:
        raise ValueError(
            f"to_native: method {method_name!r} indexes something other than the "
            f"native delegate: {sorted(stray)}. Every value must come from the "
            f"delegated subgraph."
        )

    # all_input_nodes is a Node-only ordered set: it drops literal leaves and
    # collapses repeated uses of the same Node. Walk the actual output pytree so
    # `return y, 7` and `return y, y` cannot masquerade as `return y`.
    leaves, _ = pytree.tree_flatten(output_node.args[0])
    for position, value in enumerate(leaves):
        if value is delegate_call and len(leaves) == 1:
            continue
        if not isinstance(value, torch.fx.Node):
            raise ValueError(
                f"to_native: method {method_name!r} returns literal {value!r} in "
                f"position {position}, which does not come from the native "
                f"delegate and would be dropped."
            )
        if not (
            value.op == "call_function"
            and value.target is operator.getitem
            and value.args[0] is delegate_call
        ):
            raise ValueError(
                f"to_native: method {method_name!r} returns {value.name!r}, which "
                f"does not come from the native delegate and would be dropped. "
                f"Every output must derive from the delegated subgraph."
            )
        if value.args[1] != position:
            raise ValueError(
                f"to_native: method {method_name!r} returns delegate output "
                f"{value.args[1]} in position {position}. The outer graph must "
                f"return the delegate's outputs in order, each exactly once; "
                f"reordering, duplicating, or omitting them would change the "
                f"method's contract."
            )


def _check_delegate_contract(
    graph_module: GraphModule,
    user_inputs: list[object],
    method_name: str,
) -> None:
    """Require the top-level graph to be an identity wrapper around the delegate.

    Only the delegate's graph is packaged, so anything the outer graph contributes
    is lost. An op allowlist is not enough to prove that: the outer graph can drop,
    reorder, duplicate, or inject values without running a single op. This checks
    that the wrapper adds nothing, in four ways.

    No residual ops. An op the partitioner rejected can sit beside a delegated
    region and would simply vanish.

    Inputs match. The delegate must receive exactly the method's user inputs, in
    order, so a dropped or extra public input cannot slip through.

    Outputs are an identity projection. Every output leaf must be
    ``getitem(delegate, i)`` with ``i`` equal to its position, covering the
    delegate's outputs exactly once each and in order. That rejects a pass-through
    (``return x, x + 1`` needs no op at all), a swap, a duplicate, and a gap.

    Every getitem is rooted in the delegate call, so none can smuggle in a value
    from elsewhere.

    Not covered: an output the delegate produces that the outer graph never
    references. The index rule catches a gap between returned outputs but not a
    dropped tail, and the delegate's own output_specs are not a usable reference --
    they include buffer-mutation writebacks the outer graph legitimately does not
    re-surface, and their order does not match the delegate call site. Establishing
    the delegate's call-site arity needs the serialized program, so it belongs after
    the merge rather than here.

    TODO: check for a dropped trailing output using the merged program's method
    output specs. TODO: compare mutation specs -- buffer- and input-mutation targets
    and ordering -- and dynamic-shape constraints on the matched inputs; neither is
    representable in the package yet, so a mismatch could not be honored even if
    detected.
    """
    from executorch.exir.lowered_backend_module import executorch_call_delegate

    delegate_calls = []
    residual = []
    getitems = []
    output_node = None
    for node in graph_module.graph.nodes:
        if node.op == "output":
            output_node = node
            continue
        if node.op in ("placeholder", "get_attr"):
            continue
        if node.op == "call_function" and node.target is executorch_call_delegate:
            delegate_calls.append(node)
            continue
        if node.op == "call_function" and node.target is operator.getitem:
            getitems.append(node)
            continue
        residual.append(node)

    if residual:
        raise ValueError(
            f"to_native: method {method_name!r} did not fully delegate to the "
            f"native backend; {len(residual)} op(s) remain outside the delegate and "
            f"would be silently dropped: {sorted(str(n.target) for n in residual)}"
        )
    if len(delegate_calls) != 1:
        raise ValueError(
            f"to_native: method {method_name!r} has {len(delegate_calls)} native "
            f"delegate calls; exactly one is required."
        )
    delegate_call = delegate_calls[0]
    if output_node is None:
        raise ValueError(f"to_native: method {method_name!r} has no output node.")

    delegate_inputs = [
        arg.name if isinstance(arg, torch.fx.Node) else arg
        for arg in delegate_call.args[1:]
    ]
    if delegate_inputs != user_inputs:
        raise ValueError(
            f"to_native: method {method_name!r} does not pass its inputs straight "
            f"to the native delegate. Method takes {user_inputs}, delegate receives "
            f"{delegate_inputs}; a dropped or reordered public input would change "
            f"the method's contract."
        )

    _check_delegate_outputs(output_node, delegate_call, getitems, method_name)


def _mutable_data_keys(program: Program) -> set[str]:
    """Data keys some method mutates.

    Mutability is package-wide: the runtime holds one buffer per key, so a second
    method treating it as read-only does not make the storage shareable.
    """
    keys: set[str] = set()
    for method in program.methods:
        for constant in method.constants or []:
            if constant.mutated:
                keys.add(constant.data_key)
    return keys


def _delegate_info(lowered: object, method_name: str) -> NativeDelegateInfo:
    """Read back what preprocess attached, refusing anything else."""
    meta = getattr(lowered, "meta", None) or {}
    info = meta.get("_delegate_info_meta")
    if not isinstance(info, NativeDelegateInfo):
        raise ValueError(
            f"to_native: method {method_name!r} lowered to a delegate carrying "
            f"{type(info).__name__} instead of NativeDelegateInfo."
        )
    return info


def _merge_constants(
    per_method: dict[str, dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Union per-method constants by fqn.

    serialize_program treats an fqn as one buffer across every method, so hold to
    that here rather than letting a later method silently clobber an earlier one.
    _same_tensor short-circuits on identity and shared storage, which is the usual
    case for a weight referenced by several methods.
    """
    merged: dict[str, torch.Tensor] = {}
    source: dict[str, str] = {}
    for method_name, constants in per_method.items():
        for fqn, tensor in constants.items():
            existing = merged.get(fqn)
            if existing is None:
                merged[fqn] = tensor
                source[fqn] = method_name
            elif not _same_tensor(existing, tensor):
                raise ValueError(
                    f"to_native: constant {fqn!r} differs between methods "
                    f"{source[fqn]!r} and {method_name!r}; a fully-qualified name "
                    f"must name one buffer across all methods."
                )
    return merged


def _check_shared_mutable_buffers(program: Program) -> None:
    """Hold the merged Program to serialize_program's cross-method guarantee.

    A mutable buffer fqn used by several methods is one runtime buffer, so its
    shape and dtype must agree. serialize_program validates this; merge_programs
    concatenates methods without looking, so check it here instead.
    """
    seen: dict[str, tuple[str, object]] = {}
    for method in program.methods:
        meta_by_name = {tv.name: tv.meta for tv in (method.graph.tensor_values or [])}
        for buffer in method.mutable_buffers or []:
            meta = meta_by_name.get(buffer.name)
            if meta is None:
                continue
            previous = seen.get(buffer.fqn)
            if previous is not None and previous[1] != meta:
                raise ValueError(
                    f"to_native: mutable buffer {buffer.fqn!r} has a different "
                    f"shape/dtype in methods {previous[0]!r} and {method.name!r}; a "
                    f"buffer shared across methods must match."
                )
            seen[buffer.fqn] = (method.name, meta)


def _validate_merged_program(
    program: Program, constants: dict[str, torch.Tensor]
) -> None:
    """Reject an incomplete or structurally invalid merged native program."""
    # Validate graph SSA, nested subgraphs, mutable-buffer metadata, and constant
    # bindings before applying the cross-method/package-wide checks below.
    validate_program(program, set(constants))
    _check_shared_mutable_buffers(program)

    # Catches any data key the validator does not cover, including quantization
    # scale/zero-point keys carried by graph tensor metadata.
    missing = sorted(collect_data_keys(program) - constants.keys())
    if missing:
        raise ValueError(
            f"to_native: constant(s) referenced by the graph have no backing "
            f"tensor: {missing}"
        )


def _normalize_programs(
    programs: "ExportedProgram | dict[str, ExportedProgram]",
) -> dict[str, ExportedProgram]:
    """Validate and normalize the public input before starting expensive lowering."""
    from torch.export import ExportedProgram as TorchExportedProgram

    if isinstance(programs, TorchExportedProgram):
        return {"forward": programs}
    if not isinstance(programs, dict):
        raise TypeError(
            "to_native: programs must be an ExportedProgram or a dict mapping "
            f"method names to ExportedProgram, got {type(programs).__name__}."
        )
    if not programs:
        raise ValueError("to_native: programs must contain at least one method.")

    method_programs: dict[str, ExportedProgram] = {}
    for method_name, program in programs.items():
        if not isinstance(method_name, str):
            raise TypeError(
                "to_native: every method name must be str, got "
                f"{type(method_name).__name__}."
            )
        if not method_name:
            raise ValueError("to_native: method names must be non-empty strings.")
        if not isinstance(program, TorchExportedProgram):
            raise TypeError(
                f"to_native: method {method_name!r} must map to an "
                f"ExportedProgram, got {type(program).__name__}."
            )
        method_programs[method_name] = program
    return method_programs


def to_native(
    programs: "ExportedProgram | dict[str, ExportedProgram]",
) -> NativeProgramManager:
    """Lower ExportedProgram(s) through the native backend.

    ``programs`` is a single ``ExportedProgram`` (lowered as the sole ``forward``
    method) or a dict mapping method name to ``ExportedProgram``.

    Nothing about lowering is configurable yet. The passes, the partitioner, and
    the edge compile config are all owned here, because correctness of the package
    depends on them. Exposing the compile config is worth reconsidering.
    ``constant_methods`` and ETRecord are likewise deferred.

    Args:
        programs: The exported program(s) to lower.

    Returns:
        A ``NativeProgramManager``; call ``save`` to write a .ptn package.

    Raises:
        TypeError: If ``programs`` is not an ``ExportedProgram`` or a method-name
            dictionary containing only ``ExportedProgram`` values.
        ValueError: If the method dictionary is empty, a method name is empty, a
            method does not fully delegate to the native backend, a constant
            differs between methods, or a data key the graph references has no
            backing tensor.
    """
    method_programs = _normalize_programs(programs)

    from executorch.backends.native import get_default_compile_config
    from executorch.backends.native.partitioner import NativePartitioner
    from executorch.backends.native.passes import get_default_passes
    from executorch.exir import to_edge_transform_and_lower
    from executorch.exir.lowered_backend_module import get_lowered_submodules

    edge = to_edge_transform_and_lower(
        method_programs,
        transform_passes=get_default_passes(),
        partitioner=[
            NativePartitioner(external_constants_tag=None, _serialize_as_ptn=True)
        ],
        compile_config=get_default_compile_config(),
    )

    # preprocess already serialized each method's delegated subgraph during
    # lowering and handed back its constants; reuse both instead of running
    # to_executorch or re-exporting.
    method_blobs: dict[str, bytes] = {}
    method_constants: dict[str, dict[str, torch.Tensor]] = {}
    for method_name in method_programs:
        method_program = edge.exported_program(method_name)
        lowered = get_lowered_submodules(method_program.graph_module)
        if len(lowered) != 1:
            raise ValueError(
                f"to_native: method {method_name!r} lowered to {len(lowered)} "
                f"native delegates; exactly one is required. A method must be "
                f"wholly supported by the native backend: an identity method "
                f"produces none, and disconnected supported regions produce "
                f"several. See the README for this limitation of the current "
                f"implementation."
            )
        module = lowered[0][1]
        _check_delegate_contract(
            method_program.graph_module,
            list(method_program.graph_signature.user_inputs),
            method_name,
        )
        method_blobs[method_name] = module.processed_bytes
        method_constants[method_name] = _delegate_info(module, method_name).constants

    ptg_blob = merge_programs(method_blobs)
    constants = _merge_constants(method_constants)

    program = deserialize_program(ptg_blob)
    _validate_merged_program(program, constants)

    return NativeProgramManager(ptg=ptg_blob, constants=constants)


__all__ = ["NativeProgramManager", "to_native"]
