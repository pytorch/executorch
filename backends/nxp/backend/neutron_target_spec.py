# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Target Spec for the NXP Neutron NPU

from enum import Enum

import torch

from executorch.backends.nxp.backend.neutron_converter_manager import (
    NeutronConverterManager,
)
from executorch.exir.dialects._ops import ops as exir_ops

from torch.fx import Node


class NeutronHWVersion(Enum):
    N1 = 1
    N3 = 2


class NeutronTargetNeutronC:
    @staticmethod
    def is_supported_fused_activation__aten(node_: Node) -> bool:
        """Node operator is supported fused activation on Neutron for Linear and Conv2D."""
        return node_.op == "call_function" and (
            node_.target
            in (
                torch.ops.aten.relu.default,  # TODO Add torch.ops.aten.leaky_relu.default once it is supported
                torch.ops.aten.relu_.default,
                torch.ops.aten.sigmoid.default,
                torch.ops.aten.sigmoid_.default,
                torch.ops.aten.tanh.default,
                torch.ops.aten.tanh_.default,
            )
            or (
                (
                    node_.target == torch.ops.aten.hardtanh.default
                    or node_.target == torch.ops.aten.hardtanh_.default
                )
                and (
                    node_.args[1:3] == (0.0, 6.0)  # is converted to Relu6
                    or node_.args[1:3] == (0.0, float("inf"))  # is converted to Relu
                )
            )
        )

    @staticmethod
    def is_supported_fused_activation__edge(node_: Node) -> bool:
        """Node operator is supported fused activation on Neutron for Linear and Conv2D."""
        return node_.op == "call_function" and (
            node_.target
            in (
                exir_ops.edge.aten.relu.default,  # TODO Add torch.ops.aten.leaky_relu.default once it is supported
                exir_ops.edge.aten.sigmoid.default,
                exir_ops.edge.aten.tanh.default,
            )
            or (
                (node_.target == exir_ops.edge.aten.hardtanh.default)
                and (
                    node_.args[1:3] == (0.0, 6.0)  # is converted to Relu6
                    or node_.args[1:3] == (0.0, float("inf"))  # is converted to Relu
                )
            )
        )

    @staticmethod
    def is_fusable_conv_or_linear__aten(node_: Node) -> bool:
        """Node operator is supported fusable Linear or Conv2D on Neutron."""
        return node_.op == "call_function" and (
            node_.target == torch.ops.aten.conv2d.default
            or node_.target == torch.ops.aten.addmm.default
            or node_.target == torch.ops.aten.mm.default
            or (
                node_.target == torch.ops.aten.linear.default
                and len(node_.meta["val"].shape) == 2
            )
        )

    @staticmethod
    def is_fusable_conv_or_linear__edge(node_: Node) -> bool:
        """Node operator in edge dialect is supported fusable Linear or Conv2D on Neutron."""
        return node_.op == "call_function" and (
            node_.target == exir_ops.edge.aten.addmm.default
            or node_.target == exir_ops.edge.aten.mm.default
            or (
                node_.target == exir_ops.edge.aten.convolution.default
                and len(node_.meta["val"].shape) == 4
            )
        )


class NeutronTargetSpec:
    """
    The functionality for probing the properties of Neutron Target.
    """

    def __init__(self, target: str, neutron_converter_flavor: str):

        converter_manager = NeutronConverterManager(neutron_converter_flavor)
        converter_manager.verify_target(target)
        neutron_converter = converter_manager.get_converter()
        self.neutron_target = neutron_converter.getNeutronTarget(target)

        if self.is_subsystem():
            raise ValueError(
                f"Target `{target}` is not a neutron-C target. Only MCU targets are supported at the moment."
            )

        if self.get_hw_version() != NeutronHWVersion.N3:
            raise ValueError(
                f"Target `{target}` contains unsupported HW version. Only N3/N3+ targets are supported at the moment."
            )

        # Now only Neutron-C is supported
        self.neutron_target_info = NeutronTargetNeutronC()

    # Target name.
    def get_name(self) -> str:
        return self.neutron_target.name

    # Whether the target has subsystem (Neutron-S) or not (Neutron-C).
    def is_subsystem(self) -> bool:
        return self.neutron_target.subsystem

    # Number of compute units.
    def get_num_units(self) -> int:
        return self.neutron_target.numUnits

    # Number of compute pipelines.
    def get_num_pipes(self) -> int:
        return self.neutron_target.numPipes

    # Number of compute MACs.
    def get_num_macs(self) -> int:
        return self.neutron_target.numMacs

    # Neutron compute block hardware version.
    def get_hw_version(self) -> NeutronHWVersion:
        return NeutronHWVersion(self.neutron_target.hwVersion)
