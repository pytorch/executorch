# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Any


class PipelineStage(ABC):
    """Abstract base class for all pipeline stages.

    Each stage delegates its work to an engine-specific strategy
    resolved by the EngineProxy.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def invoke(self, context: Any, input_config: Any) -> Any:
        """Execute this pipeline stage.

        Args:
            context: The PipelineContext providing global pipeline settings.
            input_config: The stage-specific input configuration dataclass.

        Returns:
            The stage-specific output configuration dataclass.
        """
        ...
