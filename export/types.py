# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum


class StageType(str, Enum):
    """
    Enum representing the different stages in the ExecuTorch export pipeline.
    """

    SOURCE_TRANSFORM = "source_transform"
    QUANTIZE = "quantize"
    TORCH_EXPORT = "torch_export"
    ATEN_TRANSFORM = "aten_transform"
    TO_EDGE_TRANSFORM_AND_LOWER = "to_edge_transform_and_lower"
    TO_EDGE = "to_edge"
    EDGE_PROGRAM_MANAGER_TRANSFORM = "edge_program_manager_transform"
    TO_BACKEND = "to_backend"
    TO_EXECUTORCH = "to_executorch"
