# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Target Spec for the NXP Neutron NPU

from enum import Enum

from executorch.backends.nxp.backend.neutron_converter_manager import (
    NeutronConverterManager,
)


class NeutronHWVersion(Enum):
    N1 = 1
    N3 = 2


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
