# Copyright 2024-2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from executorch.backends.nxp.backend.custom_delegation_options import (
    CustomDelegationOptions,
)
from executorch.backends.nxp.backend.ir.conversion_config import ConversionConfig
from executorch.backends.nxp.backend.ir.converter.builder.aten_model_builder_director import (
    AtenModelBuilderDirector,
)
from torch.export import ExportGraphSignature
from torch.nn import Parameter


class ConversionContext:
    tflite_builder: AtenModelBuilderDirector
    conversion_config: ConversionConfig
    parameters_mapping: dict[str, Parameter]
    custom_delegation_options: CustomDelegationOptions

    def __init__(
        self,
        tflite_builder: AtenModelBuilderDirector,
        conversion_config: ConversionConfig,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
        edge_program_signature: ExportGraphSignature,
    ):
        """
        Context with data related to current conversion.

        :param tflite_builder: TFLite model builder.
        :param conversion_config: Conversion configuration flags and metadata.
        :param parameters_mapping: Dictionary mapping node names to their data.
        :param custom_delegation_options: Options that affect which nodes will be delegated.
        :param edge_program_signature: Description of the inputs of the edge graph.
        """
        self.tflite_builder = tflite_builder
        self.conversion_config = conversion_config
        self.parameters_mapping = parameters_mapping
        self.custom_delegation_options = custom_delegation_options
        self.edge_program_signature = edge_program_signature
