# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Module transforms migrating linear layers to conv2d.

The two transforms here are the halves of one migration and are ordered:
:func:`prepare_conv_submodules` establishes the weight shapes that
:func:`convert_linear_to_conv2d` then reads.
"""

from __future__ import annotations

from typing import Any


def prepare_conv_submodules(module: Any) -> Any:
    """Let attention and feed-forward blocks pre-shape their weights.

    Must run before :func:`convert_linear_to_conv2d`, which reads the shapes
    these hooks establish.
    """
    for layer in module.layers:
        if getattr(layer.attention, "prepare_attention_conv", None):
            layer.attention.prepare_attention_conv()
        if getattr(layer.feed_forward, "prepare_feedforward_conv", None):
            layer.feed_forward.prepare_feedforward_conv()
    return module


def convert_linear_to_conv2d(module: Any) -> Any:
    """Rewrite every ``nn.Linear`` as a 1x1 conv2d, which HTP runs faster.

    The backend util walks named attributes, so linears reachable only through
    an ``nn.Sequential`` are left alone.
    """
    from executorch.backends.qualcomm.utils.utils import (
        convert_linear_to_conv2d as _convert,
    )

    return _convert(module)
