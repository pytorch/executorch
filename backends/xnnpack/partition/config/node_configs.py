# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import operator
from typing import List, Optional

import torch
from executorch.backends.xnnpack._passes.fuse_batch_norm_with_conv import (
    FuseBatchNormWithConvPass,
)
from executorch.backends.xnnpack.partition.config.xnnpack_config import (
    ConfigPrecisionType,
    XNNPartitionerConfig,
)
from executorch.backends.xnnpack.utils.utils import is_param_node
from executorch.exir.backend.canonical_partitioners.config_partitioner import (
    format_target_name,
)
from executorch.exir.backend.utils import WhyNoPartition
from torch.export import ExportedProgram

logger = logging.getLogger(__name__)
why = WhyNoPartition(logger=logger)


class BatchNormConfig(XNNPartitionerConfig):
    target_name = "_native_batch_norm_legit_no_training.default"

    def check_constraints(self, node: torch.fx.Node, ep: ExportedProgram) -> bool:
        if not self.check_common_constraints(node, ep):
            return False

        bn = node
        conv = node.all_input_nodes[0]

        if conv.op != "call_function":
            return False

        conv_name = format_target_name(conv.target.__name__)  # pyre-ignore

        if conv_name not in ["convolution.default"]:
            why(node, f"Invalid conv target {conv_name}")
            return False

        can_fuse = FuseBatchNormWithConvPass.can_fuse(conv, bn, ep)
        if not can_fuse:
            why(node, "BatchNorm cannot be fused with Convolution")
            return False

        return True

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


class MaxDimConfig(XNNPartitionerConfig):
    target_name = "max.dim"

    def check_constraints(self, node: torch.fx.Node, ep: ExportedProgram) -> bool:
        # We support max_dim as long as we don't return indices
        supported_dtypes = {torch.float32, torch.float16, torch.int8, torch.qint8}
        node_val = node.meta.get("val")
        output_0 = node_val[0]

        input_node = node.all_input_nodes[0]
        if len(input_node.meta.get("val").shape) != 4:
            why(node, f"Unsupported input rank {input_node.meta.get('val').shape}")
            return False
        # Don't check indicies dtype
        if output_0.dtype not in supported_dtypes:
            why(node, f"Unsupported output dtype {output_0.dtype}")
            return False

        max_input = node.all_input_nodes[0]
        if max_input.meta.get("val").dtype not in supported_dtypes:
            why(node, f"Unsupported input dtype {max_input.meta.get('val').dtype}")
            return False

        # Make sure that all users are getitems of the first output
        for user in node.users:
            if not (user.target == operator.getitem and user.args[1] == 0):
                why(node, "Unsupported user of max.dim")
                return False

        return True

    def get_node_and_deps(
        self, node: torch.fx.Node, ep: ExportedProgram
    ) -> List[torch.fx.Node]:
        getitems = list(node.users)

        return [node] + getitems

    def get_original_aten(self) -> Optional[torch._ops.OpOverload]:
        return None

    def supported_precision_types(self) -> List[ConfigPrecisionType]:
        return [ConfigPrecisionType.FP32]


class PreluConfig(XNNPartitionerConfig):
    target_name = "prelu.default"

    def check_constraints(self, node: torch.fx.Node, ep: ExportedProgram) -> bool:
        if not self.check_common_constraints(node, ep):
            return False

        weight = node.all_input_nodes[1]
        is_param = is_param_node(ep, weight)
        if not is_param:
            why(node, "Prelu weight must be a parameter")
            return False
        return True

    def get_original_aten(self) -> Optional[torch._ops.OpOverload]:
        return torch.ops.aten.prelu.default

    def get_node_and_deps(
        self, node: torch.fx.Node, ep: ExportedProgram
    ) -> List[torch.fx.Node]:
        weight = node.all_input_nodes[1]

        return [node, weight]

    def supported_precision_types(self) -> List[ConfigPrecisionType]:
        return [ConfigPrecisionType.FP32]
