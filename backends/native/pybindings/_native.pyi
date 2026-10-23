# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from os import PathLike
from typing import Any

import numpy as np
from numpy.typing import NDArray

class TensorInfo:
    @property
    def shape(self) -> list[int]: ...
    @property
    def dtype(self) -> np.dtype[Any]: ...

class MethodInfo:
    @property
    def name(self) -> str: ...
    @property
    def inputs(self) -> list[TensorInfo]: ...
    @property
    def outputs(self) -> list[TensorInfo]: ...

class Model:
    @staticmethod
    def load(path: str | PathLike[str]) -> Model: ...
    @property
    def method_names(self) -> list[str]: ...
    def method_info(self, name: str = "forward") -> MethodInfo: ...
    def create_session(self, engine: str = "ggml") -> Session: ...

class Session:
    def run(
        self, inputs: Sequence[NDArray[Any]], method: str = "forward"
    ) -> list[NDArray[Any]]: ...
