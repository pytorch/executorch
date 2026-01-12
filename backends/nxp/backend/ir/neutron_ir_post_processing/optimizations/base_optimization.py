# Copyright 2024-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod

from executorch.backends.nxp.backend.ir.conversion_config import ConversionConfig
from executorch.backends.nxp.backend.ir.converter.builder import model_builder
from executorch.backends.nxp.backend.ir.neutron_ir_post_processing.graph_utils import (
    create_tensor_to_operator_dictionaries,
    InputTensorToOpsMap,
    OutputTensorToOpMap,
)
from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec


class BaseOptimization(ABC):
    _builder: "model_builder.ModelBuilder"

    def __init__(
        self,
        builder: "model_builder.ModelBuilder",
        conversion_config: ConversionConfig,
        neutron_target_spec: NeutronTargetSpec,
    ):
        self._builder = builder
        self._conversion_config = conversion_config
        self.neutron_target_spec = neutron_target_spec

    def _create_tensor_to_operator_dictionaries(
        self,
    ) -> tuple[InputTensorToOpsMap, OutputTensorToOpMap]:
        return create_tensor_to_operator_dictionaries(self._builder)

    @abstractmethod
    def __call__(self) -> bool:
        """Execute the optimization and return `True` if the optimization had an effect and the model was modified.
        `False` otherwise.
        """
        pass
