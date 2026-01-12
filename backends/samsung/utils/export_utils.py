# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import List, Optional, Tuple

import executorch.exir as exir
import torch
from executorch.backends.samsung._passes.fuse_conv_act import FuseConvActPass
from executorch.backends.samsung._passes.remove_useless_ops import RemoveUselessOpPass
from executorch.backends.samsung.partition.enn_partitioner import EnnPartitioner
from executorch.backends.samsung.quantizer.quantizer import EnnQuantizer, Precision
from executorch.backends.transforms.decompose_sdpa import (
    DecomposeScaledDotProductAttention,
)
from executorch.backends.transforms.remove_clone_ops import RemoveCloneOpsTransform
from executorch.exir import EdgeCompileConfig
from executorch.exir.backend.backend_details import CompileSpec
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_manager import PassType
from executorch.exir.program._program import to_edge_transform_and_lower
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


def get_edge_compile_config():
    # Maybe most ops in non-decomposition list should be added here
    # TODO: to confirm whether all op in none-decomposed table should be added here
    return EdgeCompileConfig(
        _skip_dim_order=True,
        _core_aten_ops_exception_list=[
            exir_ops.edge.aten.max_pool2d.default,
            exir_ops.edge.aten.linear.default,
            exir_ops.edge.aten.hardswish.default,
            exir_ops.edge.aten.prelu.default,
            exir_ops.edge.aten.pixel_shuffle.default,
            exir_ops.edge.aten._safe_softmax.default,
            exir_ops.edge.aten.layer_norm.default,
            exir_ops.edge.aten.matmul.default,
            exir_ops.edge.aten.hardsigmoid.default,
        ],
    )


def get_enn_pass_list() -> List[PassType]:
    return [
        RemoveUselessOpPass(),
        RemoveCloneOpsTransform(),
        FuseConvActPass(),
    ]


def quantize_module(
    module: torch.nn.Module,
    inputs,
    calibration_dataset,
    precision: Precision,
    is_per_channel: bool = True,
    is_qat: bool = False,
) -> torch.nn.Module:
    quantizer = EnnQuantizer()
    quantizer.setup_quant_params(precision, is_per_channel, is_qat)
    logging.info("Export nn module for quantization...")
    exported_module = torch.export.export(module, inputs).module()
    DecomposeScaledDotProductAttention()(exported_module)
    logging.info("Quantizing the module...")
    annotated_module = prepare_pt2e(exported_module, quantizer)
    for data in calibration_dataset:
        annotated_module(*data)
    quantized_module = convert_pt2e(annotated_module, fold_quantize=False)
    logging.info("Quantizing finished.")
    return quantized_module


def to_edge_transform_and_lower_to_enn(
    module: torch.nn.Module,
    inputs: Tuple[torch.Tensor],
    custom_pass_config: List[PassType] = None,
    compile_specs: Optional[CompileSpec] = None,
) -> exir.ExecutorchProgramManager:
    assert compile_specs is not None, "For now, we must deliver complile specs"
    prog = torch.export.export(module, inputs)
    pass_list = get_enn_pass_list()
    if custom_pass_config:
        pass_list.extend(custom_pass_config)
    return to_edge_transform_and_lower(
        prog,
        pass_list,
        {"forward": [EnnPartitioner(compile_specs)]},
        compile_config=get_edge_compile_config(),
    )
