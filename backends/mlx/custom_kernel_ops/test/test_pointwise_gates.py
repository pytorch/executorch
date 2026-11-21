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
class PointwiseGateTest(OpTestCase):
    name = "pointwise_gate"

    def __init__(
        self,
        M=33,
        K=129,
        swiglu=True,
        swapped=False,
        expose=False,
        strided=False,
        broadcast=False,
        dtype=torch.float16,
        exhaustive=False,
        attention_layout=False,
    ):
        self.M = M
        self.K = K
        self.swiglu = swiglu
        self.swapped = swapped
        self.expose = expose
        self.strided = strided
        self.broadcast = broadcast
        self.dtype = dtype
        self.exhaustive = exhaustive
        self.attention_layout = attention_layout
        self.name = (
            f"pointwise_gate_m{M}_k{K}_swiglu{swiglu}_swapped{swapped}"
            f"_expose{expose}_strided{strided}_broadcast{broadcast}"
            f"_{dtype}_exhaustive{exhaustive}_attention{attention_layout}"
        )

    def create_model(self):
        k = self.K
        swiglu, swapped, expose = self.swiglu, self.swapped, self.expose
        strided, broadcast = self.strided, self.broadcast
        attention_layout = self.attention_layout

        class Model(torch.nn.Module):
            def forward(self, x):
                gate, up = x[..., :k], x[..., k:]
                if attention_layout:
                    gate = gate.transpose(1, 2).contiguous()
                    up = up.transpose(1, 2)
                if strided:
                    gate, up = gate.transpose(0, 1), up.transpose(0, 1)
                if broadcast:
                    up = up[..., :1]
                activated = (
                    torch.nn.functional.silu(gate) if swiglu else torch.sigmoid(gate)
                )
                out = up * activated if swapped else activated * up
                return (out, activated) if expose else out

        return Model()

    def create_inputs(self):
        shape = (1, 32, 8, 2 * self.K) if self.attention_layout else (8, 2 * self.K)
        return (
            torch.randn(shape, generator=torch.Generator().manual_seed(11)).to(
                self.dtype
            ),
        )

    def create_test_inputs(self):
        if self.exhaustive:
            gate = (
                torch.arange(65536, dtype=torch.int32)
                .to(torch.uint16)
                .view(torch.float16)
            )
            gate = torch.where(torch.isfinite(gate), gate, 0).reshape(self.M, self.K)
            return (torch.cat((gate, torch.ones_like(gate)), dim=-1),)
        shape = (
            (1, 32, self.M, 2 * self.K)
            if self.attention_layout
            else (self.M, 2 * self.K)
        )
        x = torch.randn(shape, generator=torch.Generator().manual_seed(17)).to(
            self.dtype
        )
        return (x,)

    def get_dynamic_shapes(self):
        return {
            "x": {
                2 if self.attention_layout else 0: torch.export.Dim(
                    "tokens", min=0, max=1024
                )
            }
        }

    def get_transform_passes(self):
        return get_default_passes()

    def get_edge_compile_config(self):
        return EdgeCompileConfig(_check_ir_validity=False)

    def generate_test_files(self, verbose=False):
        with patch.dict("os.environ", {"ET_MLX_FUSE_GATES": "1"}):
            paths = super().generate_test_files(verbose)
        fused = b"mlx_fp16_gate" in paths[0].read_bytes()
        assert fused == (
            self.dtype == torch.float16 and not self.expose and not self.broadcast
        )
        with patch.dict("os.environ", {"ET_MLX_FUSE_GATES": "0"}):
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
            [cls(M=m) for m in (0, 1, 7, 8, 16, 31, 32, 33)]
            + [cls(M=m, K=19968) for m in (511, 512, 513)]
            + [
                cls(M=m, K=128, swiglu=False, attention_layout=True)
                for m in (0, 8, 16, 512)
            ]
            + [
                cls(swiglu=False),
                cls(swapped=True),
                cls(swiglu=False, swapped=True),
                cls(strided=True),
                cls(swiglu=False, strided=True),
                cls(expose=True),
                cls(broadcast=True),
                cls(dtype=torch.bfloat16),
                cls(M=256, K=256, exhaustive=True),
                cls(M=256, K=256, swiglu=False, exhaustive=True),
            ]
        )
