# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from typing import List

import torch
from executorch.backends.xnnpack.partition.config.xnnpack_config import (
    ConfigPrecisionType,
    XNNPartitionerConfig,
)
from executorch.backends.xnnpack.passes.fuse_batch_norm_with_conv import (
    FuseBatchNormWithConvPass,
)
from torch.export import ExportedProgram


class BatchNormConfig(XNNPartitionerConfig):
    target_name = "_native_batch_norm_legit_no_training.default"

    def check_constraints(self, node: torch.fx.Node, ep: ExportedProgram) -> bool:
        if not self.check_common_constraints(node, ep):
            return False

        bn = node
        conv = node.all_input_nodes[0]

        return FuseBatchNormWithConvPass.can_fuse(conv, bn, ep)

    def get_node_and_deps(
        self, node: torch.fx.Node, ep: ExportedProgram
    ) -> List[torch.fx.Node]:
        deps = [node]

        # weight, bias, running_mean, running_var
        deps.extend(node.all_input_nodes[1:5])

        # All the users of batchnorm node must be getitem ops. batchnorm
        # returns a 3-element tuple. Each user must only access the first
        # element of the tuple.
        if [
            (user.target == operator.getitem and user.args[1] == 0)
            for user in node.users
        ].count(False):
            return []

        deps.extend(list(node.users.keys()))
        return deps

    def supported_precision_types(self) -> List[ConfigPrecisionType]:
        return [ConfigPrecisionType.FP32]
