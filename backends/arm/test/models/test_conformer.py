# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.arm_tester import ArmTester
from executorch.exir import EdgeCompileConfig

from torchaudio.models import Conformer


dim = 16
lengths = torch.randint(1, 100, (10,), dtype=torch.int32)
input_data = torch.rand(10, int(lengths.max()), dim)

conformer = Conformer(
    input_dim=dim,
    num_heads=4,
    ffn_dim=64,
    num_layers=2,
    depthwise_conv_kernel_size=31,
)
conformer = conformer.eval()

_edge_compile_config: EdgeCompileConfig = EdgeCompileConfig(
    _skip_dim_order=True,  # TODO(T182928844): Delegate dim order op to backend.
)


def test_conformer_tosa_MI():
    (
        ArmTester(
            conformer,
            example_inputs=(input_data, lengths),
            compile_spec=common.get_tosa_compile_spec(tosa_version="TOSA-0.80+MI"),
        )
        .export()
        .dump_operator_distribution("ta_conformer_ops_export.txt")
        .to_edge()  # todo: .to_edge_transform_and_lower(edge_compile_config=_edge_compile_config)
        .dump_operator_distribution("ta_conformer_ops_toedge.txt")
        # .to_executorch()
        # .run_method_and_compare_outputs(inputs=(input_data, lengths))
        # .serialize(PATH)
    )
