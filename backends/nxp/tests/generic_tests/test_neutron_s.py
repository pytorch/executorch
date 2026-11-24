# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from executorch.backends.nxp.backend.neutron_target_spec import (
    NeutronCTargetInfo,
    NeutronHWVersion,
    NeutronSTargetInfo,
    NeutronTargetSpec,
)


def _patch_neutron_target(mocker, *, subsystem: bool):
    """Make `NeutronTargetSpec` see a fake N3 target with the given `subsystem` flag."""
    fake_target = SimpleNamespace(
        name="fake_target",
        subsystem=subsystem,
        hwVersion=NeutronHWVersion.N3.value,
        numUnits=1,
        numPipes=1,
        numMacs=1,
    )
    manager_mock = mocker.patch(
        "executorch.backends.nxp.backend.neutron_target_spec.NeutronCompilerManager"
    )
    compiler = manager_mock.return_value.get_compiler.return_value
    compiler.getNeutronTarget.return_value = fake_target


def test_imxrt700_uses_neutron_c_target_info():
    target_spec = NeutronTargetSpec("imxrt700")

    assert not target_spec.is_subsystem()
    assert isinstance(target_spec.neutron_target_info, NeutronCTargetInfo)


def test_subsystem_target_uses_neutron_s_target_info(mocker):
    _patch_neutron_target(mocker, subsystem=True)

    # Must not raise. Neutron-S targets used to be rejected here.
    target_spec = NeutronTargetSpec("fake_target")

    assert target_spec.is_subsystem()
    assert isinstance(target_spec.neutron_target_info, NeutronSTargetInfo)


def test_non_subsystem_target_uses_neutron_c_target_info(mocker):
    _patch_neutron_target(mocker, subsystem=False)

    target_spec = NeutronTargetSpec("fake_target")

    assert not target_spec.is_subsystem()
    assert isinstance(target_spec.neutron_target_info, NeutronCTargetInfo)


@pytest.mark.parametrize(
    "method_name",
    [
        "is_supported_fused_activation__aten",
        "is_supported_fused_activation__edge",
        "is_fusable_conv_or_linear__aten",
        "is_fusable_conv_or_linear__edge",
    ],
)
def test_neutron_s_target_info_is_defensive(method_name):
    # This is a temporary test until Neutron-S target support implementation reaches the verification of the fusing
    #  support.
    node = MagicMock()
    node.op = "call_function"

    assert getattr(NeutronSTargetInfo(), method_name)(node) is False
