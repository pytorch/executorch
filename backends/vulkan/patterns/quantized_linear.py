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


class QuantizedLinearPattern(torch.nn.Module):
    """
    Quantized linear pattern specifically for int4 weight-only quantization
    using IntxWeightOnlyConfig with int4 weights and PerGroup granularity.
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
def get_wo_quantized_linear_graphs() -> List[torch.fx.GraphModule]:
    """
    Returns a list of quantized linear graphs that match the patterns
    produced by IntxWeightOnlyConfig quantization for both fp32 and fp16 inputs.
    """

    graphs = []

    # Different configurations to test
    configs = [
        (128, 128, False, 64, 4, PerGroup),
    ]

    for (
        in_features,
        out_features,
        bias,
        group_size,
        weight_bits,
        granularity_class,
    ) in configs:
        for dtype in [torch.float32, torch.float16]:
            # Create sample input
            batch_size = 2
            seq_len = 8
            x = torch.randn(batch_size, seq_len, in_features, dtype=dtype)

            # Create and quantize the pattern
            pattern = QuantizedLinearPattern(
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
                    strict=True,
                ),
                compile_config=EdgeCompileConfig(_check_ir_validity=False),
            )
            gm = edge.exported_program().graph_module
            graphs.append(gm)

    return graphs
