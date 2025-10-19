# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import argparse

import torch

from executorch.backends.transforms.duplicate_dynamic_quant_chain import (
    DuplicateDynamicQuantChainPass,
)
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
from executorch.backends.xnnpack.utils.configs import get_xnnpack_edge_compile_config
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.passes.sym_shape_eval_pass import ConstraintBasedSymShapeEvalPass
from torch.export import export as torch_export
from torch.nn.attention import SDPBackend
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

from transformers import Phi3ForCausalLM
from transformers.cache_utils import StaticCacheConfig

from transformers.integrations.executorch import TorchExportableModuleForDecoderOnlyLM


def _prepare_export_inputs(max_seq_len: int, sliding_window: int):
    """
    Prepare example inputs and configurations for export.

    Returns:
        example_input_ids (torch.Tensor): Example input IDs tensor.
        example_cache_position (torch.Tensor): Example cache position tensor.
        dynamic_shapes (dict or None): Dynamic shape specifications for export.
        strict (bool): Whether to use strict export mode.
    """
    # Prepare inputs with dynamic shapes
    seq_length = 3  # Sequence length > 1 to avoid specialization issues
    example_input_ids = torch.zeros((1, seq_length), dtype=torch.long)
    example_cache_position = torch.arange(seq_length, dtype=torch.long)
    max_dim = min(max_seq_len, sliding_window) - 1
    seq_len_dim = torch.export.Dim("seq_length_dim", max=max_dim)
    dynamic_shapes = {
        "input_ids": {1: seq_len_dim},
        "cache_position": {0: seq_len_dim},
    }

    return example_input_ids, example_cache_position, dynamic_shapes


def export(args) -> None:
    torch.manual_seed(0)

    if args.context_length == "4k":
        model_name = "microsoft/Phi-3-mini-4k-instruct"
    elif args.context_length == "128k":
        model_name = "microsoft/Phi-3-mini-128k-instruct"
    else:
        raise Exception(
            f"Invalid context length {args.context_length}. Should be either 4k or 128k"
        )

    with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
        model = Phi3ForCausalLM.from_pretrained(model_name)
        model.generation_config.cache_implementation = "static"
        model.generation_config.cache_config = StaticCacheConfig(
            batch_size=1, max_cache_len=model.config.max_position_embeddings
        )

        exportable_module = TorchExportableModuleForDecoderOnlyLM(
            model,
            max_batch_size=1,
            max_cache_len=model.config.max_position_embeddings,
        )
        input_ids, cache_position, dynamic_shapes = _prepare_export_inputs(
            model.config.max_position_embeddings, model.config.sliding_window
        )
        example_inputs = (input_ids, cache_position)
        exported_program = exportable_module.export(
            input_ids, cache_position, dynamic_shapes, strict=False
        )
        # Apply RemoveTransposes pass to remove
        # any back-to-back transpose ops that are not needed
        # e.g. output of update_cache is transposed and
        # input to custom_sdpa is transposed.
        from executorch.extension.llm.export.export_passes import (
            RemoveRedundantTransposes,
        )

        mutated_gm = RemoveRedundantTransposes()(exported_program.module())[0]

        xnnpack_quant_config = get_symmetric_quantization_config(
            is_per_channel=True, is_dynamic=True
        )
        xnnpack_quantizer = XNNPACKQuantizer()
        xnnpack_quantizer.set_global(xnnpack_quant_config)

        gm = prepare_pt2e(mutated_gm, xnnpack_quantizer)  # pyre-fixme[6]
        gm(*example_inputs)
        gm = convert_pt2e(gm)
        DuplicateDynamicQuantChainPass()(gm)
        exported_program = torch_export(
            gm, example_inputs, dynamic_shapes=dynamic_shapes, strict=False
        )

    edge_config = get_xnnpack_edge_compile_config()
    edge_manager = to_edge_transform_and_lower(
        exported_program,
        partitioner=[XnnpackPartitioner()],
        compile_config=edge_config,
        constant_methods={
            "get_eos_ids": [32000],
            "use_kv_cache": True,
            "enable_dynamic_shape": True,
            "get_max_seq_len": model.config.max_position_embeddings - 1,
        },
    )
    edge_manager = edge_manager.to_backend(XnnpackPartitioner())
    et_program = edge_manager.to_executorch(
        ExecutorchBackendConfig(
            extract_delegate_segments=True,
            do_quant_fusion_and_const_prop=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            sym_shape_eval_pass=ConstraintBasedSymShapeEvalPass(),
        )
    )

    with open(args.output_name, "wb") as file:
        file.write(et_program.buffer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--context_length",
        type=str,
        default="4k",
        choices=["4k", "128k"],
        help="Phi-3-mini provides two context length variants: 4k and 128k",
    )
    parser.add_argument(
        "-s",
        "--seq_len",
        type=int,
        default=128,
        help="Maximum number of tokens including prompt to generate",
    )
    parser.add_argument(
        "-o",
        "--output_name",
        default="phi-3-mini.pte",
        help="Override the output filename of the saved pte model file.",
    )
    export(parser.parse_args())


if __name__ == "__main__":
    main()
