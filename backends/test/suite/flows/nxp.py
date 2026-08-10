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

from executorch.backends.nxp.tests.tester import NeutronTester
from executorch.backends.test.suite.flow import TestFlow

# Register portable and quantized op kernels so that
# quantized_decomposed::dequantize_per_tensor / quantize_per_tensor are
# available when the suite is run without the NXP integration-repo
# conftest.py being loaded.
try:
    import executorch.extension.pybindings.portable_lib  # noqa: F401
    import executorch.kernels.quantized  # noqa: F401
except ImportError:
    pass


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
    )


NEUTRON_IMXRT700_INT8_PTQ_FLOW = _create_neutron_int8_ptq_flow(target="imxrt700")
