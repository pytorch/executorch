# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.utils._pytree as pytree
from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.passes.tests.test_replace_fused_sdpa import (
    _Fn,
    _program_for_cuda_passes,
)
from executorch.exir.backend.canonical_partitioners.duplicate_constant_node_pass import (
    duplicate_constant_node,
)
from executorch.exir.passes.replace_complex_with_real_pass import (
    _VIEWS,
    ReplaceComplexWithRealPass,
)


class TestReplaceComplexWithRealPass(unittest.TestCase):
    def test_registered(self):
        """The rewrite must stay in the CUDA backend's passes."""
        self.assertTrue(
            any(
                isinstance(p, ReplaceComplexWithRealPass)
                for p in CudaBackend.get_custom_passes([])
            )
        )

    def test_rewrite(self):
        """The rewrite of complex numbers gives eager's results."""
        angles = torch.outer(torch.arange(8.0), torch.rand(3))

        class Rotate(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # A plain attribute becomes a constant of the program.
                self.freqs = torch.polar(torch.ones_like(angles), angles)
                self.phase = torch.nn.Parameter(
                    torch.polar(torch.ones(2, 6), torch.rand(2, 6)), requires_grad=False
                )

            def forward(self, x, positions, scale):
                f = torch.stack(self.freqs[positions].unbind(-1), -1)
                f = f.unsqueeze(-2).expand(-1, 2, -1).flatten(-2)
                # Torch-TensorRT turns views into _reshape_copy.
                f = torch.ops.aten._reshape_copy(f, [-1, 1, 6])
                pairs = torch.view_as_complex(x.unflatten(-1, (-1, 2)))
                # The product keeps the permuted layout, so export flattens
                # the transpose back with a view.
                scaled = pairs.unflatten(-1, (2, 3)).permute(0, 1, -1, -2) * scale
                scaled = scaled.transpose(-1, -2).flatten(-2)
                phase = self.phase.repeat(2, 1).select(-2, 1).detach()
                # The cast sets the precision of everything after it.
                rotated = scaled * f.to(torch.complex128) * phase
                first, second = rotated.chunk(2, dim=-1)
                second = torch.cat(second.split([1, 2], dim=-1), -1)
                joined = torch.cat([first, second], -1)
                return (
                    torch.view_as_real(joined.squeeze(-2).clone()),
                    torch.view_as_real(joined.unsqueeze(0).squeeze((0, -2))),
                )

        model = Rotate()
        scale = torch.randn(3, 2)
        x = torch.randn(5, 1, 12)
        length = torch.export.Dim("length", min=2, max=64)
        program = _program_for_cuda_passes(
            model,
            (x, torch.tensor([3, 0, 7, 7, 1]), scale),
            dynamic_shapes=({0: length}, {0: length}, None),
        )
        ReplaceComplexWithRealPass()(program)
        for node in program.graph.nodes:
            val = node.meta.get("val")
            self.assertFalse(isinstance(val, torch.Tensor) and val.is_complex())
            if node.target in _VIEWS:
                # A view copy returns new tensors laid out in order.
                self.assertTrue(all(t.is_contiguous() for t in pytree.tree_leaves(val)))
        # The backend retraces the program when its decomposition table applies.
        program = program.run_decompositions({})
        for positions in (torch.tensor([3, 0, 7, 7, 1]), torch.tensor([6, 2, 4])):
            inputs = (x[: len(positions)], positions, scale)
            torch.testing.assert_close(program.module()(*inputs), model(*inputs))

    def test_tensor_with_two_placeholders(self):
        """Partitioning gives a tensor a placeholder per delegate that reads it,
        and the pass stores the tensor once."""

        class Rotate(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("freqs", torch.polar(torch.ones(3), torch.rand(3)))

            def forward(self, x, y):
                return tuple(
                    torch.view_as_real(torch.view_as_complex(t) * self.freqs)
                    for t in (x, y)
                )

        model = Rotate()
        inputs = (torch.randn(3, 2), torch.randn(3, 2))
        program = _program_for_cuda_passes(model, inputs)
        freqs = next(n for n in program.graph.nodes if n.name == "b_freqs")
        for i, user in enumerate(freqs.users):
            user.meta["delegation_tag"] = f"tag{i}"
        self.assertEqual(
            duplicate_constant_node(program, "b_freqs"), {"b_freqs_copy_0"}
        )
        ReplaceComplexWithRealPass()(program)
        for node in program.graph.nodes:
            val = node.meta.get("val")
            self.assertFalse(isinstance(val, torch.Tensor) and val.is_complex())
        torch.testing.assert_close(program.module()(*inputs), model(*inputs))

    def test_leaves_unsupported_programs(self):
        """A program with a complex op that the rewrite does not handle keeps
        all its complex numbers, so no program is left half rewritten."""

        class MutatedBuffer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("freqs", torch.randn(3, dtype=torch.cfloat))

            def forward(self, x):
                self.freqs.mul_(torch.view_as_complex(x))
                return torch.view_as_real(self.freqs.clone())

        def square(x):
            z = torch.view_as_complex(x)
            return torch.view_as_real(z * z)

        def channels_last(x):
            z = torch.view_as_complex(x).clone(memory_format=torch.channels_last)
            return torch.view_as_real(z)

        def in_branch(x):
            return torch.cond(x.sum() > 0, square, torch.neg, (square(x),))

        pairs = torch.randn(1, 2, 3, 4, 2)
        for model, inputs in (
            (_Fn(lambda z: torch.view_as_real(z * z)), (torch.randn(3).cfloat(),)),
            (_Fn(lambda x: torch.view_as_real(torch.view_as_complex(x) + 1)), (pairs,)),
            (_Fn(square), (pairs.half(),)),
            (_Fn(lambda x: torch.view_as_complex(x).float()), (pairs,)),
            (_Fn(channels_last), (pairs,)),
            (_Fn(in_branch), (pairs,)),
            (MutatedBuffer(), (torch.randn(3, 2),)),
        ):
            program = _program_for_cuda_passes(model, inputs)
            code = program.graph_module.code
            tensors = {**program.state_dict, **program.constants}
            ReplaceComplexWithRealPass()(program)
            self.assertEqual(program.graph_module.code, code)
            for name, tensor in {**program.state_dict, **program.constants}.items():
                self.assertIs(tensor, tensors[name])
