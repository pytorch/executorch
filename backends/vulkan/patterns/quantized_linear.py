# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache
from typing import Callable, List, Optional

import torch
from executorch.exir import EdgeCompileConfig, to_edge
from torch.export import export

# Import torchao modules conditionally to avoid import errors during pattern matching
from torchao.quantization.granularity import PerGroup
from torchao.quantization.quant_api import IntxWeightOnlyConfig, quantize_
from torchao.utils import unwrap_tensor_subclass


class TorchAOWeightOnlyQuantizedLinearPattern(torch.nn.Module):
    """
    Quantized linear pattern produced when quantizing linear layers using
    `torchao.quantization.quant_api.quantize_()` with IntxWeightOnlyConfig.
    """

    def __init__(
        self,
        in_features: int = 512,
        out_features: int = 256,
        bias: bool = False,
        group_size: int = 64,
        weight_bits: int = 4,
        granularity_class: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        self.group_size = group_size
        self.weight_bits = weight_bits

        if self.weight_bits == 4:
            # pyre-ignore[16]
            self.weight_dtype = torch.int4
        else:
            self.weight_dtype = torch.int8

        if granularity_class is not None:
            self.quant_granularity = granularity_class(self.group_size)
        else:
            self.quant_granularity = PerGroup(self.group_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def apply_quantization(self):
        q_config = IntxWeightOnlyConfig(
            weight_dtype=self.weight_dtype,
            granularity=self.quant_granularity,
        )
        quantize_(self, q_config)
        unwrap_tensor_subclass(self)
        return self


@lru_cache(maxsize=None)
def get_torchao_wo_quantized_linear_graphs() -> List[torch.fx.GraphModule]:
    graphs = []

    # Different configurations to test
    configs = [
        # gemv pattern
        (1, 1, 128, 128, False, 64, 4, PerGroup),
        # gemm pattern
        (1, 8, 128, 128, False, 64, 4, PerGroup),
    ]

    for (
        batch_size,
        seq_len,
        in_features,
        out_features,
        bias,
        group_size,
        weight_bits,
        granularity_class,
    ) in configs:
        for dtype in [torch.float32]:
            xs = []
            xs.append(torch.randn(batch_size, seq_len, in_features, dtype=dtype))
            if batch_size == 1:
                xs.append(torch.randn(seq_len, in_features, dtype=dtype))

            for x in xs:
                # Create and quantize the pattern
                pattern = TorchAOWeightOnlyQuantizedLinearPattern(
                    in_features=in_features,
                    out_features=out_features,
                    bias=bias,
                    group_size=group_size,
                    weight_bits=weight_bits,
                    granularity_class=granularity_class,
                )

                # Apply quantization
                pattern = pattern.apply_quantization()

                # Export the quantized pattern
                edge = to_edge(
                    export(
                        pattern,
                        (x,),
                    ),
                    compile_config=EdgeCompileConfig(_check_ir_validity=False),
                )
                gm = edge.exported_program().graph_module
                graphs.append(gm)

    return graphs
