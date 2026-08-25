# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

from executorch.backends.arm._passes.convert_squeezes_to_view import (
    ConvertSqueezesToViewPass,
)
from executorch.backends.arm._passes.rewrite_conv_pass import RewriteConvPass
from executorch.backends.arm._passes.size_adjust_input_pass import SizeAdjustInputPass
from executorch.backends.transforms.convert_conv1d_to_conv2d_pass import (
    ConvertConv1dToConv2dPass,
)
from executorch.exir.pass_base import ExportPass


class Conv1dUnsqueezePass(ConvertConv1dToConv2dPass):
    """Arm wrapper for the shared Conv1d-to-Conv2d transform."""

    _passes_required_after: Set[Type[ExportPass]] = {
        ConvertSqueezesToViewPass,
        RewriteConvPass,
        SizeAdjustInputPass,
    }
