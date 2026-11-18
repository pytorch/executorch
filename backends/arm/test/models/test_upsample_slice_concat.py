# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineFP,
    TosaPipelineINT,
)


class UpsampleSliceConcat(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.upsample = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(8, 24, 4, stride=2, padding=1),
            torch.nn.PixelShuffle(2),
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.upsample(x)
        outputs = []
        for start in (0, 2):
            for offset in range(9):
                branch = x[:, start : start + 2] + float(offset)
                outputs.append(
                    torch.cat([branch[:, :1] / 2.0, branch[:, 1:] / 4.0], dim=1)
                )
        return tuple(outputs)


model = UpsampleSliceConcat().eval()
model_inputs = (torch.randn(1, 8, 3, 5),)
input_t = tuple[torch.Tensor]


def test_upsample_slice_concat_tosa_FP() -> None:
    pipeline = TosaPipelineFP[input_t](
        model,
        model_inputs,
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.count_tosa_ops({"TRANSPOSE": 3, "RESHAPE": 2})
    pipeline.run()


def test_upsample_slice_concat_tosa_INT() -> None:
    pipeline = TosaPipelineINT[input_t](
        model,
        model_inputs,
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
        run_on_tosa_ref_model=False,
    )
    pipeline.pop_stage("to_executorch")
    pipeline.count_tosa_ops({"TRANSPOSE": 3, "RESHAPE": 2})
    pipeline.run()
