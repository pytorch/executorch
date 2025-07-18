# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file registers torch ops that are not yet in coremltools, or are in a more recent version of
# coremltools than is used by ExecuTorch.  Each op registered here should have a link to a PR in coremltools that adds
# the op to the coremltools library.

from coremltools.converters.mil.frontend.torch.ops import transpose, unbind
from coremltools.converters.mil.frontend.torch.torch_op_registry import (
    register_torch_op,
)


# https://github.com/apple/coremltools/pull/2556
@register_torch_op(override=False)
def transpose_copy(context, node):
    transpose(context, node)

# https://github.com/apple/coremltools/pull/2557
@register_torch_op(override=False)
def unbind_copy(context, node):
    unbind(context, node)
