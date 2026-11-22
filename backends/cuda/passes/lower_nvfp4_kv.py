# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Lower ``triton::nvfp4_kv_write`` so AOTInductor compiles its Triton kernel.

Functionalization turns the in-place cache write into
``auto_functionalized[_v2](triton.nvfp4_kv_write, ...)`` whose outputs feed the
buffer-mutation outputs of the program.  AOTInductor would keep that mutable
custom op as an (unsupported) fallback kernel, because ``triton_op`` only
decomposes into its Triton kernel under an active ``FunctionalTensorMode``.

This pass splices the op's launch body, traced and functionalized, in place of
the ``auto_functionalized[_v2]`` node: a ``triton_kernel_wrapper_functional`` of
the fused write kernel over the four cache buffers.  The graph keeps its
functional form (new cache values -> buffer-mutation outputs), which Inductor
re-inplaces into a direct write of the cache buffers, exactly as it does for
``index_copy`` on a BF16 cache.
"""

from __future__ import annotations

import dataclasses
import operator

import torch

from torch.export import ExportedProgram
from torch.export.graph_signature import TensorArgument
from torch.fx.experimental.proxy_tensor import make_fx

_MUTATED = ("k_q", "k_s", "v_q", "v_s")


def _write_fn(ring: bool, eps: float, has_rope: bool):
    # Imported here so that importing the CUDA backend does not load the
    # NVFP4 kernel modules.
    from executorch.backends.cuda.triton.kernels.nvfp4_kv import _nvfp4_kv_write_impl

    if has_rope:

        def fn(k, v, pos, kq, ks, vq, vs, inv):
            _nvfp4_kv_write_impl(k, v, pos, kq, ks, vq, vs, inv, ring, eps)
            return kq, ks, vq, vs

    else:

        def fn(k, v, pos, kq, ks, vq, vs):
            _nvfp4_kv_write_impl(k, v, pos, kq, ks, vq, vs, None, ring, eps)
            return kq, ks, vq, vs

    return fn


class LowerNVFP4KVWritePass:
    requires_exported_program = True

    def __call__(self, ep: ExportedProgram) -> ExportedProgram:
        # Registered by importing the NVFP4 KV kernels: without them the
        # program cannot contain the op.
        if not hasattr(torch.ops.triton, "nvfp4_kv_write"):
            return ep
        gm = ep.graph_module
        graph = gm.graph
        hops = (
            torch.ops.higher_order.auto_functionalized,
            torch.ops.higher_order.auto_functionalized_v2,
        )
        target_op = torch.ops.triton.nvfp4_kv_write.default
        renames: dict[str, str] = {}
        for node in list(graph.nodes):
            if node.op != "call_function" or node.target not in hops:
                continue
            if node.args[0] is not target_op:
                continue
            kw = node.kwargs
            # v1 passes the mutated tensors themselves and returns
            # (op_out, *mutated args in schema order); v2 passes them as
            # `_all_bases` and returns (op_out, *bases).
            if "_all_bases" in kw:
                bases = list(kw["_all_bases"])
                out_index = [kw[f"_{name}_base_index"] for name in _MUTATED]
                base_of = [bases[i] for i in out_index]
            else:
                out_index = list(range(len(_MUTATED)))
                base_of = [kw[name] for name in _MUTATED]
            inv = kw.get("inv_freq")
            has_rope = inv is not None
            call_args = [kw["k"], kw["v"], kw["input_pos"], *base_of] + (
                [inv] if has_rope else []
            )
            example = [a.meta["val"] for a in call_args]
            fn = _write_fn(
                bool(kw.get("ring", False)), float(kw.get("k_norm_eps", -1.0)), has_rope
            )
            traced = make_fx(
                torch.func.functionalize(fn, remove="mutations_and_views")
            )(*example)
            env = dict(
                zip([n for n in traced.graph.nodes if n.op == "placeholder"], call_args)
            )
            placeholders = {n for n in traced.graph.nodes if n.op == "placeholder"}
            outs = None
            with graph.inserting_before(node):
                for tn in traced.graph.nodes:
                    if tn.op == "placeholder":
                        continue
                    if tn.op == "output":
                        outs = torch.fx.map_arg(tn.args[0], env.get)
                        continue
                    # functionalize re-applies input mutations with copy_ at the
                    # end; the program carries them as buffer-mutation outputs.
                    if (
                        tn.target is torch.ops.aten.copy_.default
                        and tn.args[0] in placeholders
                    ):
                        continue
                    env[tn] = graph.node_copy(tn, env.__getitem__)
            new_for_base = dict(zip(out_index, outs))
            for user in list(node.users):
                assert user.target is operator.getitem, user
                idx = user.args[1] - 1
                if idx < 0:
                    assert not user.users
                    graph.erase_node(user)
                    continue
                new = new_for_base[idx]
                renames[user.name] = new.name
                user.replace_all_uses_with(new)
                graph.erase_node(user)
            graph.erase_node(node)
        if renames:
            sig = ep.graph_signature
            specs = []
            for spec in sig.output_specs:
                if isinstance(spec.arg, TensorArgument) and spec.arg.name in renames:
                    spec = dataclasses.replace(
                        spec, arg=TensorArgument(name=renames[spec.arg.name])
                    )
                specs.append(spec)
            sig.output_specs[:] = specs
            graph.lint()
            gm.recompile()
        return ep
