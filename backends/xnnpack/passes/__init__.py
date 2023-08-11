# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.xnnpack.passes.channels_last_tagged_reshape_pass import (
    ChannelsLastTaggedReshapePass,
)
from executorch.backends.xnnpack.passes.conv1d_unsqueeze_pass import Conv1dUnsqueezePass
from executorch.backends.xnnpack.passes.convert_to_linear import ConvertToLinearPass
from executorch.backends.xnnpack.passes.fuse_batch_norm_with_conv import (
    FuseBatchNormWithConvPass,
)
from executorch.backends.xnnpack.passes.prelu_reshape_pass import PReLUReshapePass
from executorch.backends.xnnpack.passes.remove_getitem_op import RemoveGetItemPass
from executorch.backends.xnnpack.passes.tag_implicit_q_dq_pass import TagImplicitQDqPass

from executorch.exir.passes import PassManager
from executorch.exir.passes.const_prop_pass import ConstPropPass

xnnpack_delegation_passes = PassManager(
    passes=[
        ConvertToLinearPass(),
        ConstPropPass(),
        FuseBatchNormWithConvPass(),
        RemoveGetItemPass(),
        Conv1dUnsqueezePass(),
        PReLUReshapePass(),
        ChannelsLastTaggedReshapePass(),
        TagImplicitQDqPass(),
    ]
)
