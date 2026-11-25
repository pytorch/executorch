# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from executorch.backends.nxp.backend.ir.converter.node_converter import NodeConverter
from executorch.backends.nxp.backend.neutron_target_spec import (
    NeutronCTargetInfo,
    NeutronHWVersion,
    NeutronSTargetInfo,
    NeutronTargetSpec,
)
from executorch.backends.nxp.neutron_partitioner import NeutronPartitioner
from executorch.backends.nxp.tests.executorch_pipeline import (
    to_quantized_edge_program,
    to_quantized_executorch_program,
)
from executorch.backends.nxp.tests.simple_models import ConvFCSoftmaxModule


@dataclass(frozen=True)
class _NeutronSTarget:
    name: str
    num_units: int


# i.MX95 has a multi-unit Neutron-S, i.MX943 has a single-unit Neutron-S.
_NEUTRON_S_TARGETS = [
    _NeutronSTarget("imx95", 4),
    _NeutronSTarget("imx943", 1),
]

_parametrize_neutron_s_target = pytest.mark.parametrize(
    "neutron_s_target", _NEUTRON_S_TARGETS, ids=lambda t: t.name
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


@_parametrize_neutron_s_target
def test_neutron_s_target_spec_properties(neutron_s_target: _NeutronSTarget):
    target_spec = NeutronTargetSpec(neutron_s_target.name)

    assert target_spec.get_name() == neutron_s_target.name
    assert target_spec.is_subsystem()
    assert isinstance(target_spec.neutron_target_info, NeutronSTargetInfo)
    assert target_spec.get_num_units() == neutron_s_target.num_units


def _get_partitioner_target_specs(partitioner_init_spy) -> list[NeutronTargetSpec]:
    # `NeutronPartitioner.__init__(self, compile_spec, neutron_target_spec, ...)`
    return [call.args[2] for call in partitioner_init_spy.call_args_list]


def _get_compile_spec_target(partitioner_init_spy) -> str:
    # `NeutronPartitioner.__init__(self, compile_spec, neutron_target_spec, ...)`
    compile_spec = partitioner_init_spy.call_args.args[1]
    return next(spec.value.decode() for spec in compile_spec if spec.key == "target")


def _get_converter_target_specs(is_supported_spy) -> list[NeutronTargetSpec]:
    # Collect every `NeutronTargetSpec` passed to `NodeConverter.is_supported()`, regardless of whether it was
    #  passed positionally or as a keyword argument.
    return [
        arg
        for call in is_supported_spy.call_args_list
        for arg in list(call.args) + list(call.kwargs.values())
        if isinstance(arg, NeutronTargetSpec)
    ]


def _assert_target_propagated(
    partitioner_init_spy, is_supported_spy, neutron_s_target: _NeutronSTarget
):
    # The target spec used by the partitioner must describe the requested target.
    partitioner_specs = _get_partitioner_target_specs(partitioner_init_spy)
    assert len(partitioner_specs) == 1
    spec = partitioner_specs[0]
    assert spec.get_name() == neutron_s_target.name
    assert spec.is_subsystem()
    assert spec.get_num_units() == neutron_s_target.num_units
    assert _get_compile_spec_target(partitioner_init_spy) == neutron_s_target.name

    # The same target must make its way to the node converters.
    converter_specs = _get_converter_target_specs(is_supported_spy)
    assert len(converter_specs) > 0
    assert all(s.get_name() == neutron_s_target.name for s in converter_specs)


@_parametrize_neutron_s_target
def test_to_quantized_edge_program__neutron_s_target(
    mocker, neutron_s_target: _NeutronSTarget
):
    partitioner_init_spy = mocker.spy(NeutronPartitioner, "__init__")
    is_supported_spy = mocker.spy(NodeConverter, "is_supported")

    epm = to_quantized_edge_program(
        ConvFCSoftmaxModule(),
        (1, 4, 5, 5),
        target=neutron_s_target.name,
        use_recipe_export=False,
    )
    assert epm.exported_program() is not None

    _assert_target_propagated(partitioner_init_spy, is_supported_spy, neutron_s_target)


@_parametrize_neutron_s_target
def test_to_quantized_executorch_program__neutron_s_target(
    mocker, neutron_s_target: _NeutronSTarget
):
    partitioner_init_spy = mocker.spy(NeutronPartitioner, "__init__")
    is_supported_spy = mocker.spy(NodeConverter, "is_supported")

    exec_prog = to_quantized_executorch_program(
        ConvFCSoftmaxModule(),
        (1, 4, 5, 5),
        target=neutron_s_target.name,
        use_recipe_export=False,
    )
    assert exec_prog.buffer is not None

    _assert_target_propagated(partitioner_init_spy, is_supported_spy, neutron_s_target)


def test_to_quantized_executorch_program__default_target(mocker):
    partitioner_init_spy = mocker.spy(NeutronPartitioner, "__init__")

    to_quantized_executorch_program(
        ConvFCSoftmaxModule(), (1, 4, 5, 5), use_recipe_export=False
    )

    partitioner_specs = _get_partitioner_target_specs(partitioner_init_spy)
    assert len(partitioner_specs) == 1
    assert partitioner_specs[0].get_name() == "imxrt700"
    assert not partitioner_specs[0].is_subsystem()
