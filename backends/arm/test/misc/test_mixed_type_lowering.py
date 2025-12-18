# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import Counter, defaultdict

import torch
from executorch.backends.arm.test.tester.test_pipeline import TosaPipelineINT


def combine_op_dicts(*dicts):
    merged = defaultdict(Counter)
    for d in dicts:
        for op, dtypes in d.items():
            merged[op].update(dtypes)
    return {op: dict(counts) for op, counts in merged.items()}


# TODO Figure out how to handle multiple dq/q nodes properly
# See backends/arm/_passes/decompose_quant_nodes.py for details
dq_tosa_ops = {
    "CAST": {"FP32": 1, "INT32": 1},
    "SUB": {"INT32": 1},  # zero-point subtraction
    "MUL": {"FP32": 1},  # scale multiplication
}
q_tosa_ops = {
    "CAST": {"INT8": 1},
    "MUL": {"FP32": 1},  # scale multiplication
    "ADD": {"FP32": 2},  # zero-point addition, rounding
    "SUB": {"FP32": 1},  # for rounding
    "CLAMP": {"FP32": 1},  # clamp
    "GREATER_EQUAL": {"BOOL": 1},  # for rounding
    "SELECT": {"FP32": 1},  # for rounding
    "CEIL": {"FP32": 1},  # for rounding
    "FLOOR": {"FP32": 1},  # for rounding
}
q_dq_tosa_ops = combine_op_dicts(dq_tosa_ops, q_tosa_ops)


class AddSigmoidMul(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, y):
        return self.sigmoid(x + y) * x


def test_mixed_type_lowering():
    model = AddSigmoidMul()
    input_data = (torch.randn(1, 16, 16, 16), torch.randn(1, 16, 16, 16))

    pipeline = TosaPipelineINT[type(input_data)](
        model, input_data, [], [], qtol=1, tosa_extensions=["FP"]
    )
    pipeline.quantizer.set_module_type(torch.nn.Sigmoid, None)
    expected_tosa_dtype_counts = combine_op_dicts(
        {
            "SIGMOID": {"FP32": 1},  # SIGMOID should be executed in FP32
            "ADD": {"INT32": 1},  # ADD should be executed in INT32
            "MUL": {"INT32": 1},  # MUL should be executed in INT32
        },
        q_dq_tosa_ops,
    )

    pipeline.add_stage_after(
        "to_edge_transform_and_lower",
        pipeline.tester.check_dtype_count,
        expected_tosa_dtype_counts,
    )
    pipeline.run()
