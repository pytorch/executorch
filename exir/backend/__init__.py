# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from executorch.exir.backend.backend_api import to_backend, to_backend_multiple
from executorch.exir.backend.backend_details import BackendDetails
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import Partitioner
from executorch.exir.backend.utils import (
    is_identical_graph,
    remove_first_quant_and_last_dequant,
    replace_quantized_partition_with_op,
)

__all__ = [
    "to_backend",
    "to_backend_multiple",
    "BackendDetails",
    "is_identical_graph",
    "Partitioner",
    "CompileSpec",
    "remove_first_quant_and_last_dequant",
    "replace_quantized_partition_with_op",
]
