# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import torch
from executorch.backends.mlx.passes import get_default_passes
from executorch.backends.mlx.test.test_utils import (
    export_model_to_pte,
    load_tensors_from_bin,
    OpTestCase,
    register_test,
    run_cpp_test_runner,
)
from executorch.examples.models.muse_glimmer.model.dflash_model import rotate_half
from executorch.examples.models.muse_glimmer.model.model import rotate_interleaved
from executorch.exir import EdgeCompileConfig


@register_test
class RotaryTest(OpTestCase):
    name = "rotary_fusion"

    def __init__(
        self, M=1, H=32, D=128, interleaved=True, expose=False, dtype=torch.float16
    ):
        self.M, self.H, self.D = M, H, D
        self.interleaved, self.expose, self.dtype = interleaved, expose, dtype
        self.name = (
            f"rotary_fusion_m{M}_h{H}_d{D}_i{interleaved}_expose{expose}_{dtype}"
        )

    def create_model(self):
        rotate = rotate_interleaved if self.interleaved else rotate_half
        expose, dtype, dim = self.expose, self.dtype, self.D

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "inv_freq",
                    1.0 / (500000.0 ** (torch.arange(0, dim, 2).float() / dim)),
                )

            def forward(self, x, positions):
                x = x[..., ::2].transpose(1, 2)
                phase = positions.float()[:, None] * self.inv_freq[None, :]
                cos, sin = phase.cos()[None, None], phase.sin()[None, None]
                if expose:
                    rotated = rotate(x.float(), cos, sin)
                    return rotated.to(dtype), rotated
                return rotate(x, cos, sin)

        return Model()

    def create_inputs(self):
        return torch.randn(1, 8, self.H, 2 * self.D).to(self.dtype), torch.arange(8)

    def create_test_inputs(self):
        x = torch.randn(
            1, self.M, self.H, 2 * self.D, generator=torch.Generator().manual_seed(841)
        ).to(self.dtype)
        if self.M:
            x[..., :8] = torch.tensor(
                [65504, 0, -65504, 0, 1e-7, 0, -1e-7, 0], dtype=self.dtype
            )
        positions = torch.tensor([131071, 0, 1, 1048577, 262143, 4096, 0, 17])
        positions = positions.repeat((self.M + 7) // 8)[: self.M]
        return x, positions

    def get_dynamic_shapes(self):
        tokens = torch.export.Dim("tokens", min=0, max=1024)
        return {"x": {1: tokens}, "positions": {0: tokens}}

    def get_transform_passes(self):
        return get_default_passes()

    def get_edge_compile_config(self):
        return EdgeCompileConfig(_check_ir_validity=False)

    def generate_test_files(self, verbose=False):
        with patch.dict("os.environ", {"ET_MLX_FUSE_ROPE": "1"}):
            paths = super().generate_test_files(verbose)
        assert (b"mlx_fp16_rotary" in paths[0].read_bytes()) == (
            self.dtype == torch.float16 and not self.expose
        )
        with patch.dict("os.environ", {"ET_MLX_FUSE_ROPE": "0"}):
            export_model_to_pte(
                self.create_model().eval(),
                self.create_inputs(),
                self.get_test_dir() / "unfused.pte",
                dynamic_shapes=self.get_dynamic_shapes(),
                transform_passes=self.get_transform_passes(),
                edge_compile_config=self.get_edge_compile_config(),
            )
        return paths

    def compare_with_actual(self, actual_output_path, use_dtype_tolerances=False):
        directory = self.get_test_dir()
        baseline = directory / "unfused_output.bin"
        if not run_cpp_test_runner(
            directory / "unfused.pte", directory / "input.bin", baseline
        ):
            return False, "Unfused execution failed"
        expected = load_tensors_from_bin(baseline)
        actual = load_tensors_from_bin(actual_output_path)
        if len(expected) != len(actual) or any(
            a.shape != b.shape
            or a.dtype != b.dtype
            or (a.numel() and not torch.equal(a.view(torch.uint8), b.view(torch.uint8)))
            for a, b in zip(expected, actual)
        ):
            return False, "Fusion changed the native MLX output bits"
        return True, "Bit-identical to native MLX"

    @classmethod
    def get_test_configs(cls):
        return [
            cls(M=m, H=h, interleaved=interleaved)
            for interleaved in (True, False)
            for h in (2, 32)
            for m in (0, 1, 8, 16, 513)
        ] + [cls(expose=True), cls(dtype=torch.bfloat16), cls(D=66)]
