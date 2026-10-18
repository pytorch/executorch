# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test flow registration for the NXP Neutron backend.

This module registers the Neutron INT8 PTQ lowering flow so that all shared
operator tests under backends/test/suite/operators/ are automatically expanded
to generate a variant for the Neutron backend (e.g. test_add_f32[nxp_neutron]).

Running all Neutron operator suite tests:

    pytest -c /dev/null backends/test/suite/operators/ -m backend_nxp -n auto

Generating a JSON report:

    pytest -c /dev/null backends/test/suite/operators/ -m backend_nxp \
        --json-report --json-report-file=neutron_test_report.json
"""

# Register portable and quantized op kernels so that
# quantized_decomposed::dequantize_per_tensor / quantize_per_tensor are available.
import executorch.extension.pybindings.portable_lib  # noqa: F401
import executorch.kernels.quantized  # noqa: F401
from executorch.backends.nxp.tests.tester import NeutronTester
from executorch.backends.test.suite.flow import TestFlow


# Tests known to fail on Neutron due to known bugs. Marked as xfail
# (strict=True) so that an unexpected pass is also reported.
_NEUTRON_XFAILS = [
    # Operator failures:
    "test_lstm_return_hidden_states",  # Accuracy error.
    "test_lstm_seq_lengths",  # Timeout.
    "test_lstm_num_layers",  # Timeout.
    # Model failures:
    "dynamic_shapes",  # EIEX-1136
    "test_convnext_small",  # AIR-15298
    "test_densenet161",  # EIEX-1102
    "test_inception_v3",  # EIEX-1103
    "test_maxvit_t",  # EIEX-1104
    "test_resnet50",  # EIEX-1105
    "test_resnext50_32x4d",  # EIEX-1106
    "test_shufflenet_v2_x1_0",  # EIEX-1107
    "test_swin_v2_t",  # AIR-15305
    "test_vit_b_16",  # AIR-15306
    "test_wide_resnet50_2",  # EIEX-1110
    "test_wav2letter",  # AIR-15297
]


def _create_neutron_int8_ptq_flow(target: str = "imxrt700") -> TestFlow:
    """Create the standard INT8 PTQ flow for the Neutron backend.

    The tester_factory receives (model, example_inputs) from the suite
    framework (see runner.py).  All other Neutron-specific parameters use
    their defaults (random calibration, full delegation, etc.).
    """

    def tester_factory(model, example_inputs):
        return NeutronTester(model, example_inputs, target=target)

    def quantize_stage_factory():
        # Return None so that the tester uses its own NeutronQuantize default.
        # The suite runner calls tester.quantize(flow.quantize_stage_factory())
        # which accepts None and falls back to the tester's default stage.
        return None

    return TestFlow(
        name=f"nxp_neutron_{target}_int8_ptq",
        backend="nxp",
        tester_factory=tester_factory,
        quantize=True,
        quantize_stage_factory=quantize_stage_factory,
        # The suite framework will call serialize() if supports_serialize=True.
        # Neutron requires nsys + nxp_executor_runner to run serialized inference.
        # We mark it as supported so tests attempt serialization; if the
        # simulator tools are missing, the suite marks the test as
        # PTE_RUN_FAIL (which is expected and informative in that environment).
        supports_serialize=True,
        xfail_patterns=_NEUTRON_XFAILS,
    )


NEUTRON_IMXRT700_INT8_PTQ_FLOW = _create_neutron_int8_ptq_flow(target="imxrt700")
