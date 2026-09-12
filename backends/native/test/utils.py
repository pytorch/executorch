# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared helpers for the native backend tests."""

import torch

from executorch.backends.native import get_default_compile_config
from executorch.exir import to_edge


def _transformed(model, example_inputs, passes):
    edge = to_edge(
        torch.export.export(model, example_inputs),
        compile_config=get_default_compile_config(),
    )
    return edge.transform(passes).exported_program()
