# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Graph transformation passes for the MLX backend.
"""

from typing import List

from executorch.exir.pass_base import ExportPass


def get_default_passes() -> List[ExportPass]:
    """
    Returns a list of passes that are enabled by default for the MLX backend.
    """
    return []
