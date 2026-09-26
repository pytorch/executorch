# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rewrite functional ``*_copy`` view ops into aliasing view ops."""

import torch

from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule, Node

_COPY_SUFFIX = "_copy"


def _alias_op_for_copy(
    copy_op: torch._ops.OpOverload,
) -> torch._ops.OpOverload | None:
    """Return the aliasing view op for a functional ``*_copy`` view op, or None.

    ExecuTorch functionalizes aliasing view ops into functional ``_copy`` forms
    (``view -> view_copy``, ``permute -> permute_copy``, ...). This inverts that
    for the native delegate: ``aten::<base>_copy.<overload>`` maps to
    ``aten::<base>.<overload>`` (same overload name), but only when the candidate
    is a genuine view op — i.e. its first arg is a read-only alias
    (``Tensor(a) self``). That check rejects non-view ``_copy`` ops such as
    ``_to_copy`` (a dtype/device cast, not a view).
    """
    name = copy_op._schema.name  # e.g. "aten::slice_copy"
    if "::" not in name:
        return None
    namespace, base = name.split("::", 1)
    if namespace != "aten" or not base.endswith(_COPY_SUFFIX):
        return None

    alias_base = base[: -len(_COPY_SUFFIX)]
    packet = getattr(torch.ops.aten, alias_base, None)
    if packet is None:
        return None
    overload_name = copy_op._schema.overload_name or "default"
    alias_op = getattr(packet, overload_name, None)
    if alias_op is None:
        return None

    args = alias_op._schema.arguments
    if not args:
        return None
    first = args[0]
    # A view op reads-aliases its first arg (Tensor(a)); a mutating op would set
    # is_write, and a copy/cast has no alias_info at all.
    if first.alias_info is None or first.alias_info.is_write:
        return None
    return alias_op


# View-family ops whose size argument may contain an inferred -1 and/or symbolic
# dims. Their base names (no namespace); used to substitute the known output shape
# when re-running under fake mode.
_VIEW_SIZE_OPS: frozenset = frozenset({"view", "reshape", "_unsafe_view", "expand"})


def _recompute_view_val(
    alias_op: torch._ops.OpOverload, node: Node
) -> torch.Tensor | None:
    """Re-run ``alias_op`` on fake inputs to get the view's true shape/strides.

    The ``_copy`` op's ``meta["val"]`` has contiguous strides (it materialized a
    copy); the aliasing op generally does not (e.g. ``permute``/``slice`` produce
    non-contiguous views). Recomputing keeps the serialized stride metadata
    honest (the native runtime treats it as authoritative). Returns None if the
    value cannot be produced (caller then leaves the node as a copy).

    Two things make a naive re-run fail on dynamic shapes: symbolic dims arrive as
    references to in-graph sym_size nodes (resolved here to their SymInt values),
    and a size arg may hold an inferred ``-1`` that the meta kernel cannot resolve
    against symbolic dims. For size-taking view ops the output shape is already
    known, so it is used directly as the size (dropping the ``-1``).
    """
    from torch._guards import detect_fake_mode

    def resolve(a: object) -> object:
        if isinstance(a, Node):
            return a.meta.get("val")
        if isinstance(a, (list, tuple)):
            return type(a)(resolve(x) for x in a)
        return a

    out_val = node.meta.get("val")
    fake_args = [resolve(a) for a in node.args]
    fake_kwargs = {k: resolve(v) for k, v in node.kwargs.items()}

    base = alias_op._schema.name.split("::", 1)[-1]
    if base in _VIEW_SIZE_OPS and isinstance(out_val, torch.Tensor):
        fake_args = [
            list(out_val.shape) if isinstance(a, (list, tuple)) else a
            for a in fake_args
        ]

    fake_mode = detect_fake_mode(fake_args)
    # A meta kernel can legitimately refuse this input: RuntimeError covers both
    # plain shape/stride errors and the fake-tensor family (UnsupportedOperator,
    # DataDependentOutput, DynamicOutputShape, UnsupportedFakeTensor, and
    # NotImplementedError, which all subclass it); the rest catch a malformed
    # arg list. Anything else is a bug in this pass and should surface.
    try:
        if fake_mode is not None:
            with fake_mode:
                result = alias_op(*fake_args, **fake_kwargs)
        else:
            result = alias_op(*fake_args, **fake_kwargs)
    except (RuntimeError, IndexError, TypeError, ValueError):
        return None
    return result if isinstance(result, torch.Tensor) else None


class ReplaceCopyWithAliasPass(ExportPass):
    """Rewrite functional ``*_copy`` view ops into aliasing view ops.

    ExecuTorch functionalizes aliasing view ops (``view``, ``permute``,
    ``slice``, ``transpose``, ``squeeze``, ...) into their ``_copy`` forms so the
    core runtime never aliases. The native runtime can execute true zero-copy
    views, so inside the delegate we invert that: each ``<op>_copy`` that is a
    genuine view (see ``_alias_op_for_copy``) is rewritten to the aliasing
    ``<op>``, and its fake-value metadata is recomputed so the serialized strides
    describe the alias rather than a contiguous copy.

    Runs in ``NativeBackend.preprocess`` (never as a pre-partition transform),
    since aliasing aten ops are not core-tagged and must stay inside the delegate.
    At runtime each view's base buffer must outlive the view. Views whose result
    is a delegate output are skipped: the output buffer may be reassigned by the
    caller, so aliasing into it is unsafe.

    Multi-output views (``split``/``unbind``) are left as copies for now — their
    getitem consumers need handling this pass does not yet do.

    A view is also left as a copy when its aliased layout is not dim-order
    expressible (e.g. a gapped last-dim slice): the serialized TensorMeta records
    only dim_order, not strides, so such a layout cannot be represented and must
    stay a materialized (contiguous) copy.
    """

    def call(self, graph_module: GraphModule) -> PassResult:
        from executorch.backends.native.serialization.graph_serialize import _dim_order

        graph = graph_module.graph
        modified = False

        for node in graph.nodes:
            if node.op != "call_function":
                continue
            op = getattr(node.target, "_op", node.target)
            if not isinstance(op, torch._ops.OpOverload):
                continue
            alias_op = _alias_op_for_copy(op)
            if alias_op is None:
                continue
            if any(user.op == "output" for user in node.users):
                continue
            if not isinstance(node.meta.get("val"), torch.Tensor):
                continue

            new_val = _recompute_view_val(alias_op, node)
            if new_val is None:
                continue
            # TensorMeta serializes only dim_order, not strides, so an aliased view
            # whose layout is not dim-order expressible (e.g. a gapped last-dim
            # slice) cannot be represented; leave it as a materialized copy.
            try:
                _dim_order(new_val)
            except ValueError:
                continue

            node.target = alias_op
            node.meta["val"] = new_val
            modified = True

        if modified:
            graph.lint()

        return PassResult(graph_module, modified)
