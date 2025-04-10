# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Export recipe definitions for ExecuTorch.

This module provides the data structures needed to configure the export process
for ExecuTorch models, including export configurations.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Sequence

from executorch.exir.backend.partitioner import Partitioner
from executorch.exir.capture import EdgeCompileConfig, ExecutorchBackendConfig
from executorch.exir.pass_manager import PassType
from torch.ao.quantization.quantizer import Quantizer
from torch.export import ExportedProgram


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
class ExportRecipe:
    """
    Configuration recipe for the export process.

    This class holds the configuration parameters for exporting a model,
    including quantization, compilation, and transformation options.

    Attributes:
        name: Optional name for the recipe
        quantizer: Optional quantizer for model quantization
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
    quantizer: Optional[Quantizer] = None
    edge_compile_config: Optional[EdgeCompileConfig] = ( # pyre-ignore[11]: Type not defined
        None
    )
    pre_edge_transform_passes: Optional[
        Callable[[ExportedProgram], ExportedProgram]
    ] = None
    edge_transform_passes: Optional[Sequence[PassType]] = None
    transform_check_ir_validity: bool = True
    partitioners: Optional[list[Partitioner]] = None
    executorch_backend_config: Optional[ExecutorchBackendConfig] = ( # pyre-ignore[11]: Type not defined
        None
    )
    mode: Mode = Mode.RELEASE

    def get_quantizer(self) -> Optional[Quantizer]:
        """
        Get the quantizer associated with this recipe.

        Returns:
            The quantizer if one is set, otherwise None
        """
        return self.quantizer
