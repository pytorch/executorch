# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Antialiased bilinear and bicubic resizes on the CUDA backend.

Inductor has no lowering for these resizes, so the backend rewrites them into
plain tensor math. These tests lower real programs, run them through the Python
runtime, and compare with eager PyTorch.
"""

import functools
import os
import tempfile
import unittest
from typing import Optional, Sequence, Tuple

import torch
from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from torch.export import Dim, export


class Resize(torch.nn.Module):
    """Resizes x by scale_factor, or else to like's height and a width of 56."""

    def __init__(
        self,
        mode: str,
        align_corners: bool = False,
        scale_factor: Optional[Tuple[float, float]] = None,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        size = None if self.scale_factor else (like.shape[2], 56)
        return torch.nn.functional.interpolate(
            x,
            size=size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
            antialias=True,
        )


# Dynamic heights let one program shrink, grow, and resize to 1.
DYNAMIC_HEIGHTS = (
    {2: Dim("in_height", min=4, max=64)},
    {2: Dim("out_height", min=1, max=32)},
)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
@unittest.skipIf(
    torch.version.hip is not None,
    "loads through the Python runtime, which the ROCm build does not include",
)
class TestAntialiasedResize(unittest.TestCase):
    def _run(
        self,
        model: Resize,
        cases: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        dynamic_shapes=None,
        **tolerance: float,
    ) -> None:
        """Lower model for cases[0], then run every case and compare with eager."""
        exported = export(model, cases[0], dynamic_shapes=dynamic_shapes)
        spec = CudaBackend.generate_method_name_compile_spec("forward")
        lowered = to_edge_transform_and_lower(
            exported,
            partitioner=[CudaPartitioner([spec])],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )
        from executorch.runtime import Runtime

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.pte")
            with open(path, "wb") as f:
                lowered.to_executorch().write_to_file(f)
            method = Runtime.get().load_program(path).load_method("forward")
            for x, like in cases:
                (output,) = method.execute([x, like])
                # The rewrite stands in for ATen's CUDA kernel.
                expected = model(x.cuda().float(), like.cuda()).to(x.dtype).cpu()
                torch.testing.assert_close(
                    output.cpu(), expected, equal_nan=True, **tolerance
                )

    def _check_dynamic_heights(self, mode: str, align_corners: bool) -> None:
        cases = []
        # 21 to 7 and 22 to 8 give whole ratios, with and without align_corners.
        heights = ((48, 20), (64, 20), (10, 20), (10, 1), (21, 7), (22, 8))
        for in_height, out_height in heights:
            x = torch.randn(1, 3, in_height, 32)
            # As in ATen, a NaN reaches only the outputs whose window has it.
            x[0, 0, -1, -1] = float("nan")
            cases.append((x, torch.empty(1, 1, out_height)))
        self._run(
            Resize(mode, align_corners),
            cases,
            DYNAMIC_HEIGHTS,
            atol=1e-5,
            rtol=1e-4,
        )

    def test_bilinear_antialias_resize(self):
        self._check_dynamic_heights("bilinear", align_corners=False)

    def test_bicubic_antialias_resize(self):
        self._check_dynamic_heights("bicubic", align_corners=True)

    def test_antialias_resize_with_uneven_scale_factor(self):
        # 37 * 1.75 and 51 * 0.45 are not whole, so the scale the resize uses
        # differs from output size / input size.
        x = torch.randn(1, 3, 37, 51)
        self._run(
            Resize("bicubic", scale_factor=(1.75, 0.45)),
            [(x, torch.empty(0))],
            atol=1e-5,
            rtol=1e-4,
        )

    def test_half_precision_antialias_resize(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                x = torch.randn(1, 3, 48, 32).to(
                    dtype=dtype, memory_format=torch.channels_last
                )
                self._run(
                    Resize("bilinear"),
                    [(x, torch.empty(1, 1, 20))],
                    atol=1e-2,
                    rtol=1e-2,
                )


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestAntialiasedResizeDecomposition(unittest.TestCase):
    def test_keeps_memory_format(self):
        # The program's output layout is fixed at export, so only a direct call
        # shows the rewrite's own layout.
        resize = CudaBackend.get_decomposition_table()[
            torch.ops.aten._upsample_bilinear2d_aa.default
        ]
        x = torch.randn(1, 3, 48, 32, device="cuda")
        output = resize(x.to(memory_format=torch.channels_last), [20, 56], False)
        self.assertTrue(output.is_contiguous(memory_format=torch.channels_last))

    def test_align_corners_compiles_with_dynamic_sizes(self):
        from torch._inductor.compile_fx import compile_fx

        model = Resize("bicubic", align_corners=True)
        compiled = torch.compile(
            model,
            backend=functools.partial(
                compile_fx, decompositions=CudaBackend.get_decomposition_table()
            ),
            dynamic=True,
            fullgraph=True,
        )
        for in_height, out_height in ((48, 20), (10, 1)):
            x = torch.randn(1, 3, in_height, 32, device="cuda")
            like = torch.empty(1, 1, out_height, device="cuda")
            torch.testing.assert_close(
                compiled(x, like), model(x, like), atol=1e-5, rtol=1e-4
            )
