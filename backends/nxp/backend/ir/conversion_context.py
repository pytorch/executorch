# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch import Node
from torch.nn import Parameter

from executorch.backends.nxp.backend.ir.conversion_config import ConversionConfig
from executorch.backends.nxp.backend.ir.converter.builder.aten_model_builder_director import AtenModelBuilderDirector
from executorch.backends.nxp.backend.node_format_inference import NodeFormat


class ConversionContext:
    tflite_builder: AtenModelBuilderDirector
    conversion_config: ConversionConfig
    parameters_mapping: dict[str, Parameter]
    node_formats: dict[Node, NodeFormat]

    def __init__(self, tflite_builder: AtenModelBuilderDirector, conversion_config: ConversionConfig,
                 parameters_mapping: dict,
                 node_formats: dict[Node, NodeFormat], ):
        """
        Context with data related to current conversion.

        :param tflite_builder: TFLite model builder.
        :param conversion_config: Conversion configuration flags and metadata.
        """
        self.tflite_builder = tflite_builder
        self.conversion_config = conversion_config
        self.parameters_mapping = parameters_mapping
        self.node_formats = node_formats
