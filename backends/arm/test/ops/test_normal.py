# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm.test.tester.test_pipeline import TosaPipelineFP


input_t = tuple[torch.Tensor]
MEAN = 2.0
STD = 3.0
NUM_SAMPLES = 100_000


class Normal(torch.nn.Module):
    aten_op = "torch.ops.aten.normal.float_float"
    exir_op = "executorch_exir_dialects_edge__ops_aten_normal_float_float"
    randn_exir_op = "executorch_exir_dialects_edge__ops_aten_randn_default"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.normal(MEAN, STD, size=x.shape)


def _compare_normal_statistics(
    reference: torch.Tensor,
    output: torch.Tensor,
    _qparams,
) -> None:
    assert output.shape == reference.shape
    assert output.dtype == reference.dtype
    assert torch.isfinite(output).all()

    torch.testing.assert_close(
        output.mean(),
        torch.tensor(MEAN),
        atol=0.06,
        rtol=0.0,
    )
    torch.testing.assert_close(
        output.std(correction=0),
        torch.tensor(STD),
        atol=0.05,
        rtol=0.0,
    )


def test_normal_tosa_FP() -> None:
    inputs = (torch.zeros(NUM_SAMPLES),)
    pipeline = TosaPipelineFP[input_t](
        Normal(),
        inputs,
        Normal.aten_op,
        Normal.exir_op,
        run_on_tosa_ref_model=False,
    )
    pipeline.add_stage_after(
        "to_edge_transform_and_lower",
        pipeline.tester.check,
        [Normal.randn_exir_op],
        suffix="portable_randn",
    )
    pipeline.add_stage(
        pipeline.tester.run_method_and_compare_outputs,
        inputs=inputs,
        compare_callback=_compare_normal_statistics,
    )
    pipeline.run()
