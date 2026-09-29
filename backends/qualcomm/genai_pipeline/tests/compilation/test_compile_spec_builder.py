# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import MagicMock, patch

from executorch.backends.qualcomm.genai_pipeline.compilation.compile_spec_builder import (
    QnnCompileSpecBuilder,
    resolve_backend_type,
    resolve_soc_model,
)
from executorch.backends.qualcomm.genai_pipeline.control_args import ControlArgs
from executorch.backends.qualcomm.serialization.qc_schema import (
    QcomChipset,
    QnnExecuTorchBackendType,
)

_TEST_SOC = "SM8750"

# The builder imports these lazily from their defining module, so patching
# there intercepts the call.
_HTP_SPEC = "executorch.backends.qualcomm.utils.utils.generate_htp_compiler_spec"
_GPU_SPEC = "executorch.backends.qualcomm.utils.utils.generate_gpu_compiler_spec"
_QNN_SPEC = (
    "executorch.backends.qualcomm.utils.utils." "generate_qnn_executorch_compiler_spec"
)


def _build_with_mocks(builder, **build_kwargs):
    """Invoke builder.build() with the spec generators mocked.

    Returns:
        A (htp_mock, gpu_mock, qnn_mock) tuple of the patched generators.
    """
    with patch(_HTP_SPEC) as htp, patch(_GPU_SPEC) as gpu, patch(_QNN_SPEC) as qnn:
        builder.build(**build_kwargs)
    return htp, gpu, qnn


class TestResolveSocModel(unittest.TestCase):

    def test_resolves_known_name(self):
        """A CLI-style SoC name maps to its QcomChipset member."""
        self.assertEqual(resolve_soc_model("SM8750"), QcomChipset.SM8750)

    def test_passes_through_enum(self):
        """An already-resolved QcomChipset is returned unchanged."""
        self.assertEqual(resolve_soc_model(QcomChipset.SM8650), QcomChipset.SM8650)

    def test_unknown_name_raises_with_supported_list(self):
        """An unknown name raises ValueError naming the valid SoCs."""
        with self.assertRaises(ValueError) as cm:
            resolve_soc_model("NOT_A_SOC")

        message = str(cm.exception)
        self.assertIn("NOT_A_SOC", message)
        self.assertIn("SM8750", message)


class TestResolveBackendType(unittest.TestCase):

    def test_resolves_supported_backends(self):
        """Backend names map to their QnnExecuTorchBackendType members."""
        expected = {
            "htp": QnnExecuTorchBackendType.kHtpBackend,
            "gpu": QnnExecuTorchBackendType.kGpuBackend,
            "lpai": QnnExecuTorchBackendType.kLpaiBackend,
            "dsp": QnnExecuTorchBackendType.kDspBackend,
        }

        for name, member in expected.items():
            with self.subTest(backend=name):
                self.assertEqual(resolve_backend_type(name), member)

    def test_unknown_backend_raises(self):
        """An unknown backend name raises ValueError rather than AttributeError."""
        with self.assertRaises(ValueError) as cm:
            resolve_backend_type("not_a_backend")

        message = str(cm.exception)
        self.assertIn("not_a_backend", message)
        self.assertIn("htp", message)

    def test_the_undefined_backend_is_not_selectable(self):
        """kUndefinedBackend is the unset value, so its name resolves to nothing."""
        with self.assertRaises(ValueError):
            resolve_backend_type(str(QnnExecuTorchBackendType.kUndefinedBackend))


class TestQnnCompileSpecBuilderConstruction(unittest.TestCase):

    def test_resolves_soc_model_eagerly(self):
        """The SoC is resolved at construction, not at build time."""
        builder = QnnCompileSpecBuilder(soc_model=_TEST_SOC)

        self.assertEqual(builder.soc_model, QcomChipset.SM8750)

    def test_rejects_unknown_soc_model(self):
        """A bad SoC fails at construction, before any graph work."""
        with self.assertRaises(ValueError):
            QnnCompileSpecBuilder(soc_model="NOT_A_SOC")

    def test_rejects_backend_without_spec_support(self):
        """A backend with no spec-generation path is rejected."""
        with self.assertRaises(ValueError) as cm:
            QnnCompileSpecBuilder(soc_model=_TEST_SOC, backend="lpai")

        self.assertIn("lpai", str(cm.exception))

    def test_backend_type_exposes_the_enum(self):
        """backend_type converts the backend name to the QNN enum."""
        builder = QnnCompileSpecBuilder(soc_model=_TEST_SOC, backend="gpu")

        self.assertEqual(builder.backend_type, QnnExecuTorchBackendType.kGpuBackend)


class TestQnnCompileSpecBuilderDeviceDefaults(unittest.TestCase):

    def test_device_target_enables_weight_sharing_and_shared_buffer(self):
        """On device, both weight sharing and shared buffer default to on."""
        builder = QnnCompileSpecBuilder(soc_model=_TEST_SOC)

        htp, _, qnn = _build_with_mocks(builder)

        self.assertTrue(htp.call_args.kwargs["use_weight_sharing"])
        self.assertTrue(qnn.call_args.kwargs["shared_buffer"])

    def test_x86_target_disables_weight_sharing_and_shared_buffer(self):
        """The emulator supports neither, so both default to off."""
        builder = QnnCompileSpecBuilder(soc_model=_TEST_SOC, enable_x86_64=True)

        htp, _, qnn = _build_with_mocks(builder)

        self.assertFalse(htp.call_args.kwargs["use_weight_sharing"])
        self.assertFalse(qnn.call_args.kwargs["shared_buffer"])

    def test_explicit_values_override_x86_defaults(self):
        """Passing the flags explicitly overrides the target-derived default."""
        builder = QnnCompileSpecBuilder(soc_model=_TEST_SOC, enable_x86_64=True)

        htp, _, qnn = _build_with_mocks(
            builder, use_weight_sharing=True, shared_buffer=True
        )

        self.assertTrue(htp.call_args.kwargs["use_weight_sharing"])
        self.assertTrue(qnn.call_args.kwargs["shared_buffer"])


class TestQnnCompileSpecBuilderForwarding(unittest.TestCase):

    def test_forwards_htp_precision_and_contexts(self):
        """use_fp16 and use_multi_contexts reach the HTP options helper."""
        builder = QnnCompileSpecBuilder(soc_model=_TEST_SOC)

        htp, _, _ = _build_with_mocks(builder, use_fp16=True, use_multi_contexts=True)

        self.assertTrue(htp.call_args.kwargs["use_fp16"])
        self.assertTrue(htp.call_args.kwargs["use_multi_contexts"])

    def test_forwards_soc_and_graph_options(self):
        """The SoC enum and per-graph options reach the spec generator."""
        builder = QnnCompileSpecBuilder(soc_model=_TEST_SOC)

        _, _, qnn = _build_with_mocks(builder, online_prepare=True, use_mha2sha=True)

        self.assertEqual(qnn.call_args.kwargs["soc_model"], QcomChipset.SM8750)
        self.assertTrue(qnn.call_args.kwargs["online_prepare"])
        self.assertTrue(qnn.call_args.kwargs["use_mha2sha"])

    def test_passes_backend_options_through(self):
        """The backend options object is handed to the spec generator as-is."""
        builder = QnnCompileSpecBuilder(soc_model=_TEST_SOC)
        options = MagicMock(name="backend_options")

        with patch(_HTP_SPEC, return_value=options), patch(_QNN_SPEC) as qnn:
            builder.build()

        self.assertIs(qnn.call_args.kwargs["backend_options"], options)


class TestQnnCompileSpecBuilderBackendSelection(unittest.TestCase):

    def test_htp_backend_uses_htp_options(self):
        """The HTP target builds HTP options and not GPU options."""
        builder = QnnCompileSpecBuilder(soc_model=_TEST_SOC, backend="htp")

        htp, gpu, _ = _build_with_mocks(builder)

        htp.assert_called_once()
        gpu.assert_not_called()

    def test_gpu_backend_uses_gpu_options(self):
        """The GPU target builds GPU options and not HTP options."""
        builder = QnnCompileSpecBuilder(soc_model=_TEST_SOC, backend="gpu")

        htp, gpu, _ = _build_with_mocks(builder)

        gpu.assert_called_once()
        htp.assert_not_called()

    def test_gpu_backend_warns_that_fp16_is_ignored(self):
        """GPU has no fp16 switch, so requesting it warns rather than silently dropping."""
        builder = QnnCompileSpecBuilder(soc_model=_TEST_SOC, backend="gpu")

        with patch(_GPU_SPEC), patch(_QNN_SPEC):
            with self.assertLogs(
                "executorch.backends.qualcomm.genai_pipeline.compilation."
                "compile_spec_builder",
                level="WARNING",
            ) as logs:
                builder.build(use_fp16=True)

        self.assertIn("use_fp16", logs.output[0])


class TestQnnCompileSpecBuilderAgainstRealApis(unittest.TestCase):
    """Exercises the real QNN spec generation, which needs no device."""

    def test_builds_real_compile_specs(self):
        """A default HTP build produces usable CompileSpec objects."""
        from executorch.exir.backend.compile_spec_schema import CompileSpec

        specs = QnnCompileSpecBuilder(soc_model=_TEST_SOC).build()

        self.assertTrue(specs)
        for spec in specs:
            with self.subTest(spec=spec.key):
                self.assertIsInstance(spec, CompileSpec)

    def test_multi_contexts_conflicts_with_online_prepare(self):
        """The underlying generator rejects this combination; it is not masked."""
        builder = QnnCompileSpecBuilder(soc_model=_TEST_SOC)

        with self.assertRaises(ValueError):
            builder.build(use_multi_contexts=True, online_prepare=True)


class TestQnnCompileSpecBuilderFromControlArgs(unittest.TestCase):
    """Covers the seam where a ``ControlArgs`` drives the builder.

    The two disagree on ``shared_buffer``: ``ControlArgs`` mirrors
    ``llama.py``'s parser, which defaults it off, while the builder defaults it
    on for a device target. A caller must therefore thread the value rather
    than rely on either default.
    """

    def test_control_args_shared_buffer_default_differs_from_the_builders(self):
        """The two defaults differ, which is why callers must pass it explicitly."""
        control_args = ControlArgs()

        _, _, qnn = _build_with_mocks(QnnCompileSpecBuilder(soc_model=_TEST_SOC))

        self.assertFalse(control_args.shared_buffer)
        self.assertTrue(qnn.call_args.kwargs["shared_buffer"])

    def test_threading_control_args_shared_buffer_is_honoured(self):
        """Passing the ControlArgs value explicitly overrides the builder default."""
        control_args = ControlArgs()

        _, _, qnn = _build_with_mocks(
            QnnCompileSpecBuilder(soc_model=_TEST_SOC),
            shared_buffer=control_args.shared_buffer,
        )

        self.assertFalse(qnn.call_args.kwargs["shared_buffer"])


if __name__ == "__main__":
    unittest.main()
