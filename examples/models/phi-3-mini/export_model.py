# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.backends.transforms.duplicate_dynamic_quant_chain import (
    DuplicateDynamicQuantChainPass,
)
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.utils.configs import get_xnnpack_edge_compile_config
from executorch.exir import to_edge
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e

from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
from torch.export import export

from transformers import Phi3ForCausalLM


def main() -> None:
    torch.random.manual_seed(0)

    model = Phi3ForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

    example_inputs = (torch.randint(0, 100, (1, 100), dtype=torch.long),)
    dynamic_shape = {"input_ids": {1: torch.export.Dim("sequence_length", max=128)}}

    xnnpack_quant_config = get_symmetric_quantization_config(
        is_per_channel=True, is_dynamic=True
    )
    xnnpack_quantizer = XNNPACKQuantizer()
    xnnpack_quantizer.set_global(xnnpack_quant_config)

    with torch.nn.attention.sdpa_kernel(
        [torch.nn.attention.SDPBackend.MATH]
    ), torch.no_grad():
        model = capture_pre_autograd_graph(
            model, example_inputs, dynamic_shapes=dynamic_shape
        )
        model = prepare_pt2e(model, xnnpack_quantizer)
        model(*example_inputs)
        model = convert_pt2e(model, fold_quantize=False)
        DuplicateDynamicQuantChainPass()(model)
        model = export(
            model, example_inputs, dynamic_shapes=dynamic_shape, strict=False
        )

    edge_config = get_xnnpack_edge_compile_config()
    edge_manager = to_edge(model, compile_config=edge_config)
    edge_manager = edge_manager.to_backend(XnnpackPartitioner(has_dynamic_shapes=True))
    et_program = edge_manager.to_executorch()

    with open("phi-3-mini.pte", "wb") as file:
        file.write(et_program.buffer)


if __name__ == "__main__":
    main()
