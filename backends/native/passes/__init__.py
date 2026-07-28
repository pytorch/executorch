# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Graph transformation passes for the native backend.

Passed to ``to_edge_transform_and_lower`` via its ``transform_passes`` argument so
they run on the edge-dialect graph before partitioning/delegation. Each pass lives
in its own module; this package aggregates them and exposes the default pass list.
"""

from typing import List, Union

from executorch.exir.pass_base import ExportedProgramPassBase, ExportPass
from executorch.exir.passes.cse_pass import CSEPass

__all__ = [
    "get_default_passes",
]


def get_default_passes() -> List[Union[ExportPass, ExportedProgramPassBase]]:
    """Passes enabled by default for the native backend."""
    return [
        CSEPass(),
    ]
