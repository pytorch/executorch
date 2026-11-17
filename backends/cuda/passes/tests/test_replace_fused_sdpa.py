# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.aoti.passes.replace_view_copy_with_view import (
    ReplaceViewCopyWithViewPass,
)
from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.backends.cuda.passes.replace_fused_sdpa import (
    ReplaceFusedSdpaWithSdpaPass,
)
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import export


def _program_for_cuda_passes(model, inputs, dynamic_shapes=None):
    """The Edge program of ``model`` as the CUDA backend's passes get it: no
    op decomposed, as the CUDA partitioner asks, and views for copies."""
    program = export(model, inputs, dynamic_shapes=dynamic_shapes, strict=True)
    # to_edge_transform_and_lower functionalizes the program before it asks
    # the partitioner which ops to keep.
    program = program.run_decompositions({})
    ops, _ = CudaPartitioner([]).ops_to_not_decompose(program)
    program = to_edge(
        program,
        compile_config=EdgeCompileConfig(_check_ir_validity=False, preserve_ops=ops),
    ).exported_program()
    ReplaceViewCopyWithViewPass()(program.graph_module)
    return program


class _Fn(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args):
        return self.fn(*args)


class TestReplaceFusedSdpaWithSdpaPass(unittest.TestCase):
    def test_runs_before_triton_replacement(self):
        """The calls it rewrites must still reach the Triton SDPA kernels."""
        names = [type(p).__name__ for p in CudaBackend.get_custom_passes([])]
        self.assertLess(
            names.index("ReplaceFusedSdpaWithSdpaPass"),
            names.index("ReplaceEdgeOpWithTritonOpPass"),
        )

    def test_rewrite(self):
        """Fused attention becomes scaled_dot_product_attention only when that
        computes the same result."""
        aten = torch.ops.aten

        def cudnn(q, k):
            return aten._scaled_dot_product_cudnn_attention(q, k, k, None, False)[0]

        def causal(q, k, bias=None):
            return aten._scaled_dot_product_efficient_attention(
                q, k, k, bias, False, is_causal=True
            )[0]

        def grouped(q, k):
            return aten._scaled_dot_product_flash_attention(q, k, k, scale=0.2)[0]

        def with_log_sum_exp(q, k):
            return aten._scaled_dot_product_efficient_attention(q, k, k, None, True)[:2]

        def flash(q, k):
            return aten._scaled_dot_product_flash_attention(q, k, k)[0]

        def in_branch(q, k):
            return torch.cond(q.sum() > 0, flash, lambda q, k: q * 2, (q, k))

        q = torch.randn(1, 2, 16, 64)
        one_head = torch.randn(1, 1, 16, 64)
        decode = q[:, :, :1].clone()
        length = torch.export.Dim("length", min=2, max=64)
        keys = torch.export.Dim("keys", min=2, max=64)
        # The last item is the keyword arguments of the new call, or None when
        # the fused call has to stay.
        for fn, inputs, lengths, kwargs in (
            (cudnn, (q, q), None, {}),
            (causal, (q, q), None, {}),
            (causal, (q, q.clone()), (length, length), {}),
            (grouped, (q, one_head), None, {"scale": 0.2, "enable_gqa": True}),
            (in_branch, (q, q), None, {}),
            # With different query and key lengths, the variants and the Triton
            # decode kernel place the causal mask differently.
            (causal, (decode, q), None, None),
            (causal, (q, q.clone()), (length, keys), None),
            (causal, (q, q, torch.randn(1, 2, 16, 16)), None, None),
            (with_log_sum_exp, (q, q), None, None),
        ):
            shapes = [t.shape for t in inputs]
            with self.subTest(fn=fn.__name__, shapes=shapes, lengths=lengths):
                dynamic_shapes = None
                if lengths:
                    dynamic_shapes = {"args": tuple({2: n} for n in lengths)}
                program = _program_for_cuda_passes(_Fn(fn), inputs, dynamic_shapes)
                ReplaceFusedSdpaWithSdpaPass()(program)
                program.validate()
                nodes = [
                    node
                    for module in program.graph_module.modules()
                    if isinstance(module, torch.fx.GraphModule)
                    for node in module.graph.nodes
                ]
                sdpa = exir_ops.edge.aten.scaled_dot_product_attention.default
                calls = [node.kwargs for node in nodes if node.target == sdpa]
                self.assertEqual(calls, [] if kwargs is None else [kwargs])
                fused = [
                    node
                    for node in nodes
                    if "aten._scaled_dot_product_" in str(node.target)
                ]
                self.assertEqual(not fused, kwargs is not None)
