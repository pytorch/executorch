# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from typing import Any, Dict, List, Sequence, Tuple

class ExecutorchModule:
    def run_method(self, method_name: str, inputs: Sequence[Any]) -> List[Any]: ...
    def forward(self, inputs: Sequence[Any]) -> List[Any]: ...

def _load_for_executorch(path: str) -> ExecutorchModule: ...
def _load_for_executorch_from_buffer(buffer: bytes) -> ExecutorchModule: ...
