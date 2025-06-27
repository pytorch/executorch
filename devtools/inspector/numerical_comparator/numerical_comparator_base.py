# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from abc import ABC, abstractmethod
from typing import Any


class NumericalComparatorBase(ABC):
    @abstractmethod
    def compare(self, a: Any, b: Any) -> float:
        """Compare two intermediate output and return a result.

        This method should be overridden by subclasses to provide custom comparison logic.

        Args:
            a: The first intermediate output to compare.
            b: The second intermediate output to compare.

        Returns:
            A numerical result indicating the comparison outcome.
        """
        pass
