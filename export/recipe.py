# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Export recipe definitions for ExecuTorch.

This module provides the data structures needed to configure the export process
for ExecuTorch models, including export configurations and quantization recipes.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Sequence

from executorch.exir._warnings import experimental

from executorch.exir.backend.partitioner import Partitioner
from executorch.exir.capture import EdgeCompileConfig, ExecutorchBackendConfig
from executorch.exir.pass_manager import PassType
from torch.export import ExportedProgram
from torchao.core.config import AOBaseConfig
from torchao.quantization.pt2e.quantizer import Quantizer


class Mode(str, Enum):
    """
    Export mode enumeration.

    Attributes:
        DEBUG: Debug mode with additional checks and information
        RELEASE: Release mode optimized for performance
    """

    DEBUG = "debug"
    RELEASE = "release"


@dataclass
class QuantizationRecipe:
    """
    Configuration recipe for quantization.

    This class holds the configuration parameters for quantizing a model.

    Attributes:
        quantizer: Optional quantizer for model quantization
    """

    quantizers: Optional[List[Quantizer]] = None
    ao_base_config: Optional[List[AOBaseConfig]] = None

    def get_quantizers(self) -> Optional[Quantizer]:
        """
        Get the quantizer associated with this recipe.

        Returns:
            The quantizer if one is set, otherwise None
        """
        return self.quantizers


@experimental(
    "This API and all of its related functionality such as ExportSession and ExportRecipe are experimental."
)
@dataclass
class ExportRecipe:
    """
    Configuration recipe for the export process.

    This class holds the configuration parameters for exporting a model,
    including compilation and transformation options.

    Attributes:
        name: Optional name for the recipe
        quantization_recipe: Optional quantization recipe for model quantization
        edge_compile_config: Optional edge compilation configuration
        pre_edge_transform_passes: Optional function to apply transformation passes
                                  before edge lowering
        edge_transform_passes: Optional sequence of transformation passes to apply
                              during edge lowering
        transform_check_ir_validity: Whether to check IR validity during transformation
        partitioners: Optional list of partitioners for model partitioning
        executorch_backend_config: Optional backend configuration for ExecuTorch
        mode: Export mode (debug or release)
    """

    name: Optional[str] = None
    quantization_recipe: Optional[QuantizationRecipe] = None
    edge_compile_config: Optional[EdgeCompileConfig] = (
        None  # pyre-ignore[11]: Type not defined
    )
    pre_edge_transform_passes: Optional[
        Callable[[ExportedProgram], ExportedProgram]
        | List[Callable[[ExportedProgram], ExportedProgram]]
    ] = None
    edge_transform_passes: Optional[Sequence[PassType]] = None
    transform_check_ir_validity: bool = True
    partitioners: Optional[List[Partitioner]] = None
    executorch_backend_config: Optional[ExecutorchBackendConfig] = (
        None  # pyre-ignore[11]: Type not defined
    )
    mode: Mode = Mode.RELEASE
