# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import List

import torch
from executorch.backends.example.example_operators.ops import module_to_annotator
from torch import fx
from torch.ao.quantization.observer import HistogramObserver, MinMaxObserver
from torch.ao.quantization.pt2e.graph_utils import find_sequential_partitions
from torch.ao.quantization.quantizer import QuantizationSpec, Quantizer
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import OperatorConfig


def get_uint8_tensor_spec(observer_or_fake_quant_ctr):
    return QuantizationSpec(
        dtype=torch.uint8,
        quant_min=0,
        quant_max=255,
        qscheme=torch.per_tensor_affine,
        is_dynamic=False,
        observer_or_fake_quant_ctr=observer_or_fake_quant_ctr,
    )


@dataclass
class ExampleQuantConfig:
    input_quant_spec: QuantizationSpec
    output_quant_spec: QuantizationSpec
    weight_quant_spec: QuantizationSpec
    bias_quant_spec: QuantizationSpec


default_static_config = ExampleQuantConfig(
    get_uint8_tensor_spec(HistogramObserver),
    get_uint8_tensor_spec(HistogramObserver),
    get_uint8_tensor_spec(MinMaxObserver),
    # pyre-fixme[6]: Incompatible parameter type [6]: In call `ExampleQuantConfig.__init__`, for 4th positional argument, expected `QuantizationSpec` but got `None`.
    None,  # #bias quantization can be configured here or done in a pass later on.
)


def check_for_outside_users(partitions) -> bool:
    """
    Make sure that all the users of this partiton are within the delegatable subgraph,
    except the last partition. If we quantize partitions that have users outside this
    subgraph then delegation of this partition to the backend will not be possible.
    """
    for source_partition in partitions[:-1]:
        if len(source_partition.output_nodes) != 1:
            return True
        if len(source_partition.output_nodes[0].users) != 1:
            return True
    return False


class ExampleQuantizer(Quantizer):
    def __init__(self, quantizer_supported_modules=None, quant_config=None):
        super().__init__()
        if quantizer_supported_modules is not None:
            self.quantizer_supported_modules = quantizer_supported_modules
            for module in self.quantizer_supported_modules:
                if module not in module_to_annotator.keys():
                    assert 0, f"{module} is not supported by this quantizer"
        else:
            self.quantizer_supported_modules = module_to_annotator.keys()
        if quant_config is not None:
            self.quant_config = quant_config
        else:
            self.quant_config = default_static_config

    def annotate(self, model):
        for supported_modules in self.quantizer_supported_modules:
            # print("supported modules: ", supported_modules)
            fused_partitions = find_sequential_partitions(
                model,
                list(supported_modules),
            )

            for partitions in fused_partitions:
                if check_for_outside_users(partitions):
                    continue

                source_module_list = ()
                for partition in partitions:
                    source_module_list += (partition,)

                annotator = module_to_annotator[supported_modules].annotate_handle
                annotator(partitions, self.quant_config)

        return model

    def validate(self, model: fx.GraphModule) -> None:
        pass

    @classmethod
    def get_supported_operators(cls) -> List[OperatorConfig]:
        return []
