# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from abc import abstractmethod
from enum import Enum
from typing import List, Optional

import torch
from executorch.backends.xnnpack.utils.quant_utils import (
    is_dequant,
    is_qparam,
    is_quant,
)
from executorch.exir.backend.canonical_partitioners.config_partitioner import (
    format_target_name,
    PartitionerConfig,
)
from executorch.exir.backend.utils import WhyNoPartition
from torch.export import ExportedProgram
from torch.fx.experimental.symbolic_shapes import has_free_unbacked_symbols

logger = logging.getLogger(__name__)
why = WhyNoPartition(logger=logger)


class ConfigPrecisionType(Enum):
    FP32 = 1
    STATIC_QUANT = 2
    DYNAMIC_QUANT = 3


class XNNPartitionerConfig(PartitionerConfig):
    """
    Base partitioner config for XNNPACK Partitioner Configs. Base wrapper class
    for all XNNPACK Partitioner Configs allows us to apply control over
    all PartitionerConfigs. XNNPACK Partitioner config also sets a property
    for supported precision types. This allows partitioner configs to set
    the precision types they support, and let users toggle which precision
    types they want to enable
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.enabled_precision_types = self.supported_precision_types()
        # Flag used in GEMMConfig()
        self.force_non_static_weights_for_f32_linear = kwargs.get(
            "force_non_static_weights_for_f32_linear", False
        )

    def get_partition(
        self, node: torch.fx.Node, ep: ExportedProgram
    ) -> List[torch.fx.Node]:
        """
        Overriding abstract method get_partition.

        Returns the partitioned nodes from get_node_and_deps, but also labels them
        with the name of the XNNPartitionerConfig class which return this set of nodes.
        This enforces that all partitions returned by XNNPartitioner configs are labeled
        with the partitioner config which returned them
        """
        partitioned_nodes = self.get_node_and_deps(node, ep)
        # label partitioned nodes with the name of the partitioner config
        for node in partitioned_nodes:
            if "xnn_partitioner_config" in node.meta:
                node.meta["xnn_partitioner_config"].append(self.__class__.__name__)
            else:
                node.meta["xnn_partitioner_config"] = [self.__class__.__name__]

        return partitioned_nodes

    def get_original_aten(self) -> Optional[torch._ops.OpOverload]:
        # By default if not specified, we do not halt decomposition for those configs
        return None

    @abstractmethod
    def supported_precision_types(self) -> List[ConfigPrecisionType]:
        """
        Returns the supported PrecisionType of this partitioner config
        """
        pass

    @abstractmethod
    def get_node_and_deps(
        self, node: torch.fx.Node, ep: ExportedProgram
    ) -> List[torch.fx.Node]:
        """
        Takes in a node and its exported program and returns a list of nodes
        and its dependencies that need to be partitioned together

        Args:
            node: Node to be partitioned
            ep: Exported program of the graph module
        Returns:
            List of nodes that can be partitioned
        """
        pass

    def set_enabled_precision_types(
        self, precision_types: Optional[List[ConfigPrecisionType]]
    ):
        """
        Set the enabled precisions.

        We take the intersection of the precision_types we wish to enable with
        the precision types that this config supports. If enabled_precisions is empty, i.e.
        the config does not support any of the precision types we want to enable,
        then we will not partition nothing and return false at the common constraints
        """

        if precision_types:
            enabled_precisions = []
            for precision in precision_types:
                if precision in self.supported_precision_types():
                    enabled_precisions.append(precision)

            self.enabled_precision_types = enabled_precisions

    def check_common_constraints(
        self, node: torch.fx.Node, ep: ExportedProgram
    ) -> bool:
        """
        Checks common xnnpack constraints

        Args:
            node (torch.fx.Node): Node to check common constraints against
            ep (ExportedProgram): Exported Program to check constraints against

        Returns:
            True or False whether this node is partitionable
        """
        assert (
            node.op == "call_function"
            and format_target_name(node.target.__name__)  # pyre-ignore
            == self.target_name
        )

        if len(self.enabled_precision_types) == 0:
            why(node, reason="not enabled precision types")
            return False

        has_valid_dtypes = self._check_node_has_valid_dtype(node)
        if not has_valid_dtypes:
            why(node, reason="invalid dtype")
            return False

        return True

    def _check_inputs_are_valid_dtypes(self, node, valid_dtypes):
        # Check inputs are valid and have the same dtypes
        # Gather all args which are nodes
        args_to_check = []
        reference_dtype = None
        for arg in node.args:
            if isinstance(arg, list) or isinstance(arg, tuple):
                for item in arg:
                    if isinstance(item, torch.fx.Node):
                        args_to_check.append(item)

            if isinstance(arg, torch.fx.Node):
                args_to_check.append(arg)

        for arg in args_to_check:
            arg_val = arg.meta.get("val", None)

            if arg_val is None or isinstance(arg_val, tuple):
                continue

            # Being conservative for now, UX >> Perf
            # TODO: We need a pass to scrub these out.
            if not isinstance(arg_val, torch.Tensor):
                return False

            # XNNPACK does not support empty tensors. But we can't get numel()
            # for unbacked symints, so we conservatively bail out here if any
            # dimension of the tensor is unbacked symint.
            if has_free_unbacked_symbols(arg_val) or arg_val.numel() == 0:
                return False

            if arg_val.dtype not in valid_dtypes:
                return False

            # Use the first dtype as reference
            reference_dtype = reference_dtype or arg_val.dtype

            # Check for mixed dtypes
            if arg_val.dtype != reference_dtype:
                # Get op name if the attribute exists, otherwise use the full node target for logging
                op_name = (
                    node.target.__name__
                    if hasattr(node.target, "__name__")
                    else str(node.target)
                )
                why(
                    node,
                    reason=(
                        f"{op_name} does not support mixed input dtypes, "
                        f"got: [{reference_dtype}, {arg_val.dtype}]"
                    ),
                )
                return False

        return True

    def _check_outputs_are_valid_dtypes(self, node, valid_dtypes):
        # Check outputs are valid
        node_val = node.meta.get("val", None)

        if node_val is None:
            return True

        if not isinstance(node_val, tuple):
            node_val = (node_val,)

        for val in node_val:
            if not isinstance(val, torch.Tensor):
                return False

            if val.dtype not in valid_dtypes:
                return False

        return True

    def _check_node_has_valid_dtype(self, node):
        valid_dtypes = {
            torch.float32,
            torch.float16,
        }
        # Only allow int8 and quant dtypes for quant operations
        if is_quant(node) or is_dequant(node) or is_qparam(node):
            valid_dtypes.update(
                {
                    torch.qint32,
                    torch.qint8,
                    torch.quint8,
                    torch.int8,
                }
            )

        if (
            node.op != "placeholder"
            and node.op != "call_function"
            and node.op != "get_attr"
        ):
            return False

        return self._check_inputs_are_valid_dtypes(
            node, valid_dtypes
        ) and self._check_outputs_are_valid_dtypes(node, valid_dtypes)
