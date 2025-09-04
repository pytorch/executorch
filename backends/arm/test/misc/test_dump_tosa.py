# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile

from typing import Tuple

import torch
from executorch.backends.arm.arm_backend import ArmCompileSpecBuilder

from executorch.backends.arm.test import common
from executorch.backends.arm.tosa_partitioner import TOSAPartitioner
from executorch.backends.arm.tosa_specification import TosaSpecification
from executorch.exir import to_edge
from executorch.exir.capture._config import ExecutorchBackendConfig

input_t1 = Tuple[torch.Tensor]


class Linear(torch.nn.Module):
    inputs = {
        "randn": (torch.randn(2, 8),),
    }

    def __init__(self):
        super().__init__()
        in_features = 8
        out_features = 16
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        y = torch.matmul(x, self.weight.t())
        return torch.add(y, self.bias)


def _file_non_empty(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


@common.parametrize("test_data", Linear.inputs)
def test_MI_dump_delegate_data(test_data: input_t1):

    m = Linear().eval()
    ep = torch.export.export(m, test_data, strict=True)

    tosa_spec = TosaSpecification.create_from_string("TOSA-1.0+FP")
    partitioner = TOSAPartitioner(
        ArmCompileSpecBuilder().tosa_compile_spec(tosa_spec=tosa_spec).build()
    )
    edge = to_edge(ep).to_backend(partitioner)

    config = ExecutorchBackendConfig(extract_delegate_segments=True)
    exec_pm = edge.to_executorch(config)

    tmp_dir = tempfile.mkdtemp()
    out_file = os.path.join(tmp_dir, "delegate_MI.tosa")
    prefix, ext = os.path.splitext(out_file)
    # Ensure extension is provided, if not then use default
    if not ext:
        ext = ".tosa"
    exec_pm.dump_delegate_data(path=prefix, extension=ext)
    print(f"delegate file ⇒ {prefix + ext}")

    assert os.path.exists(prefix + ext), f"File {prefix + ext} not created"
    assert os.path.getsize(prefix + ext) > 0, f"File {prefix + ext} is empty"
