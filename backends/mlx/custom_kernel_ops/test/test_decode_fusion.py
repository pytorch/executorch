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
from executorch.exir import EdgeCompileConfig


@register_test
class DecodeFusionTest(OpTestCase):
    name = "decode_fusion"

    def __init__(
        self,
        kind="norm",
        M=1,
        D=6656,
        expose=False,
        swapped=False,
        dtype=torch.float16,
        eps=1e-8,
        exhaustive=False,
    ):
        self.kind, self.M, self.D = kind, M, D
        self.expose, self.swapped, self.dtype = expose, swapped, dtype
        self.eps, self.exhaustive = eps, exhaustive
        self.flag = "ET_MLX_FUSE_NORM_ADD" if kind == "norm" else "ET_MLX_FUSE_SOFT_CAP"
        self.kernel = (
            b"mlx_fp16_rms_norm_add" if kind == "norm" else b"mlx_fp32_soft_cap"
        )
        self.name = f"decode_fusion_{kind}_m{M}_d{D}_expose{expose}_swap{swapped}_{dtype}_eps{eps}_all{exhaustive}"

    def create_model(self):
        kind, dim, dtype = self.kind, self.D, self.dtype
        expose, swapped, eps = self.expose, self.swapped, self.eps

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                if kind == "norm":
                    self.register_buffer(
                        "weight",
                        torch.randn(
                            dim, generator=torch.Generator().manual_seed(43)
                        ).to(dtype),
                    )
                else:
                    self.register_buffer("multiplier", torch.tensor(0.19611613513))
                    self.register_buffer("cap", torch.tensor(20.0))

            def forward(self, x):
                if kind == "norm":
                    value, residual = x[..., ::2], x[..., 1::2]
                    normalized = torch.nn.functional.rms_norm(
                        value, (dim,), self.weight, eps
                    )
                    result = normalized + residual if swapped else residual + normalized
                    return (result, normalized) if expose else result
                scaled = x.float() * self.multiplier
                result = torch.tanh(scaled / self.cap) * self.cap
                return (result, scaled) if expose else result

        return Model()

    def create_inputs(self):
        return (
            torch.randn(1, 8, self.D * (2 if self.kind == "norm" else 1)).to(
                self.dtype
            ),
        )

    def create_test_inputs(self):
        if self.exhaustive:
            x = (
                torch.arange(65536, dtype=torch.int32)
                .to(torch.uint16)
                .view(torch.float16)
            )
            x = torch.where(torch.isfinite(x), x, 0).reshape(1, self.M, self.D)
        else:
            x = torch.randn(
                1,
                self.M,
                self.D * (2 if self.kind == "norm" else 1),
                generator=torch.Generator().manual_seed(841),
            ).to(self.dtype)
            if self.kind == "norm" and self.M > 1:
                x[:, 0, ::2] = 0
                x[:, 1, ::2] *= 1e-4
                if self.M > 2:
                    x[:, 2, ::2] = 65504
        return (x,)

    def get_dynamic_shapes(self):
        return {"x": {1: torch.export.Dim("tokens", min=0, max=1024)}}

    def get_transform_passes(self):
        return get_default_passes()

    def get_edge_compile_config(self):
        return EdgeCompileConfig(_check_ir_validity=False)

    def generate_test_files(self, verbose=False):
        with patch.dict("os.environ", {self.flag: "1"}):
            paths = super().generate_test_files(verbose)
        assert (self.kernel in paths[0].read_bytes()) == (
            self.dtype == torch.float16 and not self.expose
        )
        with patch.dict("os.environ", {self.flag: "0"}):
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
        return (
            [cls(M=m) for m in (0, 1, 8, 16, 513)]
            + [cls(M=8, D=d) for d in (127, 128, 129, 4096, 4097)]
            + [
                cls(swapped=True),
                cls(eps=1e-5),
                cls(expose=True),
                cls(dtype=torch.bfloat16),
            ]
            + [cls(kind="soft_cap", M=m, D=202048) for m in (0, 1, 8, 16)]
            + [
                cls(kind="soft_cap", M=256, D=256, exhaustive=True),
                cls(kind="soft_cap", D=257, expose=True),
                cls(kind="soft_cap", D=257, dtype=torch.bfloat16),
            ]
        )
