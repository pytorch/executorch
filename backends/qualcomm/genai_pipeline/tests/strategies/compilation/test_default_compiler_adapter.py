# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
    ARTIFACT_TEXT_DECODER,
    ARTIFACT_VISION_ENCODER,
)
from executorch.backends.qualcomm.genai_pipeline.compilation import (
    QnnCompileSpecBuilder,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.compilation.compiler_adapter import (
    CompilerAdapter,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.compilation.default_compiler_adapter import (
    DefaultCompilerAdapter,
)
from executorch.backends.qualcomm.genai_pipeline.tests.test_utils import (
    TEST_BACKEND_TYPE,
    TEST_SOC_CHIPSET,
    TEST_SOC_MODEL,
)
from executorch.backends.qualcomm.serialization.qc_schema import (
    QcomChipset,
    QnnExecuTorchBackendType,
)
from executorch.backends.qualcomm.utils.constants import QCOM_QNN_COMPILE_SPEC

# The adapter imports both lazily from their defining modules, so patching there
# intercepts the calls.
_LOWER = "executorch.backends.qualcomm.utils.utils.to_edge_transform_and_lower_to_qnn"
_DECODE = (
    "executorch.backends.qualcomm.serialization.qc_schema_serialize."
    "flatbuffer_to_option"
)

_FILE_NAME = "test_model"


def _make_edge_program_manager():
    """Create a mock EdgeProgramManager whose program writes bytes to file.

    ``write_to_file`` receives a real file object, so the mock writes to it to
    keep the on-disk artifact non-empty.
    """
    exec_prog_mgr = MagicMock(name="exec_prog_mgr")
    exec_prog_mgr.write_to_file.side_effect = lambda f: f.write(b"pte-bytes")

    edge_prog_mgr = MagicMock(name="edge_prog_mgr")
    edge_prog_mgr.to_executorch.return_value = exec_prog_mgr
    return edge_prog_mgr


def _compile(adapter, artifact_dir, **overrides):
    """Invoke compile_model with valid defaults and lowering mocked.

    Returns:
        A (result, lower_mock, edge_prog_mgr) tuple.
    """
    edge_prog_mgr = _make_edge_program_manager()
    kwargs = {
        "model": MagicMock(name="model"),
        "example_inputs": (MagicMock(name="example_input"),),
        "compile_specs": [MagicMock(name="compile_spec")],
        "artifact_dir": artifact_dir,
        "file_name": _FILE_NAME,
        "soc_model": TEST_SOC_CHIPSET,
        "backend_type": TEST_BACKEND_TYPE,
    }
    kwargs.update(overrides)

    with patch(_LOWER, return_value=edge_prog_mgr) as lower:
        result = adapter.compile_model(**kwargs)

    return result, lower, edge_prog_mgr


class TestDefaultCompilerAdapterProtocol(unittest.TestCase):

    def test_satisfies_the_compiler_adapter_protocol(self):
        """The adapter is usable where a CompilerAdapter is expected."""
        self.assertIsInstance(DefaultCompilerAdapter(), CompilerAdapter)


class TestDefaultCompilerAdapterArtifacts(unittest.TestCase):

    def test_writes_pte_named_after_file_name(self):
        """The artifact is written as <file_name>.pte in artifact_dir."""
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)

            result, _, _ = _compile(DefaultCompilerAdapter(), artifact_dir)

            expected = artifact_dir / f"{_FILE_NAME}.pte"
            self.assertEqual(result.artifact_paths, {ARTIFACT_TEXT_DECODER: expected})
            self.assertTrue(expected.exists())

    def test_creates_missing_artifact_directories(self):
        """A nested artifact_dir that does not exist is created."""
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp) / "nested" / "dir"

            _compile(DefaultCompilerAdapter(), artifact_dir)

            self.assertTrue(artifact_dir.is_dir())

    def test_written_artifact_is_not_empty(self):
        """The ExecuTorch program's bytes reach the file."""
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)

            result, _, _ = _compile(DefaultCompilerAdapter(), artifact_dir)

            path = result.artifact_paths[ARTIFACT_TEXT_DECODER]
            self.assertGreater(path.stat().st_size, 0)

    def test_keys_the_artifact_as_the_text_decoder_by_default(self):
        """Omitting artifact_key yields the text decoder, the text-only case."""
        with tempfile.TemporaryDirectory() as tmp:
            result, _, _ = _compile(DefaultCompilerAdapter(), Path(tmp))

            self.assertEqual(list(result.artifact_paths), [ARTIFACT_TEXT_DECODER])

    def test_keys_the_artifact_under_the_requested_name(self):
        """A non-decoder component is keyed as itself, not as the decoder."""
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)

            result, _, _ = _compile(
                DefaultCompilerAdapter(),
                artifact_dir,
                artifact_key=ARTIFACT_VISION_ENCODER,
            )

            expected = artifact_dir / f"{_FILE_NAME}.pte"
            self.assertEqual(result.artifact_paths, {ARTIFACT_VISION_ENCODER: expected})


class TestDefaultCompilerAdapterLoweringArguments(unittest.TestCase):

    def test_forwards_required_lowering_inputs(self):
        """model, inputs and compile specs reach lowering under its own names."""
        model = MagicMock(name="model")
        example_inputs = (MagicMock(name="example_input"),)
        compile_specs = [MagicMock(name="compile_spec")]

        with tempfile.TemporaryDirectory() as tmp:
            _, lower, _ = _compile(
                DefaultCompilerAdapter(),
                Path(tmp),
                model=model,
                example_inputs=example_inputs,
                compile_specs=compile_specs,
            )

        self.assertIs(lower.call_args.kwargs["module"], model)
        self.assertIs(lower.call_args.kwargs["inputs"], example_inputs)
        self.assertIs(lower.call_args.kwargs["compiler_specs"], compile_specs)

    def test_forwards_per_graph_lowering_inputs(self):
        """constant_methods, dep_table and passes_job reach lowering."""
        constant_methods = {"get_max_seq_len": 512}
        dep_table = {"pass": ["dependency"]}
        passes_job = MagicMock(name="passes_job")

        with tempfile.TemporaryDirectory() as tmp:
            _, lower, _ = _compile(
                DefaultCompilerAdapter(),
                Path(tmp),
                constant_methods=constant_methods,
                dep_table=dep_table,
                passes_job=passes_job,
            )

        self.assertIs(lower.call_args.kwargs["constant_methods"], constant_methods)
        self.assertIs(lower.call_args.kwargs["dep_table"], dep_table)
        self.assertIs(lower.call_args.kwargs["passes_job"], passes_job)

    def test_forwards_known_extra_options(self):
        """Recognised tuning knobs are forwarded to lowering."""
        skip_ops = {"llama.fallback.default"}

        with tempfile.TemporaryDirectory() as tmp:
            _, lower, _ = _compile(
                DefaultCompilerAdapter(),
                Path(tmp),
                extra_options={
                    "skip_node_op_set": skip_ops,
                    "convert_linear_to_conv2d": True,
                },
            )

        self.assertIs(lower.call_args.kwargs["skip_node_op_set"], skip_ops)
        self.assertTrue(lower.call_args.kwargs["convert_linear_to_conv2d"])

    def test_ignores_unknown_extra_options(self):
        """Options lowering does not accept are not forwarded to it."""
        with tempfile.TemporaryDirectory() as tmp:
            _, lower, _ = _compile(
                DefaultCompilerAdapter(),
                Path(tmp),
                extra_options={"not_a_lowering_option": True},
            )

        self.assertNotIn("not_a_lowering_option", lower.call_args.kwargs)

    def test_does_not_forward_dynamic_shapes(self):
        """HTP has no dynamic shapes, so none is passed even though the API has it."""
        with tempfile.TemporaryDirectory() as tmp:
            _, lower, _ = _compile(DefaultCompilerAdapter(), Path(tmp))

        self.assertNotIn("dynamic_shapes", lower.call_args.kwargs)


class TestDefaultCompilerAdapterExecutorchConfig(unittest.TestCase):

    def test_leaves_graph_io_unallocated_by_default(self):
        """Graph I/O is not allocated, since a shared buffer supplies it."""
        with tempfile.TemporaryDirectory() as tmp:
            _, _, edge_prog_mgr = _compile(DefaultCompilerAdapter(), Path(tmp))

        config = edge_prog_mgr.to_executorch.call_args.args[0]
        self.assertFalse(config.memory_planning_pass.alloc_graph_input)
        self.assertFalse(config.memory_planning_pass.alloc_graph_output)

    def test_builds_quant_io_by_default(self):
        """BuildQuantIo runs, so quantized I/O tensors get their types."""
        from executorch.backends.qualcomm._passes.build_quant_io import BuildQuantIo

        with tempfile.TemporaryDirectory() as tmp:
            _, _, edge_prog_mgr = _compile(DefaultCompilerAdapter(), Path(tmp))

        config = edge_prog_mgr.to_executorch.call_args.args[0]
        self.assertTrue(any(isinstance(p, BuildQuantIo) for p in config.passes))

    def test_extra_options_can_override_the_config(self):
        """A caller-supplied backend config replaces the default."""
        override = MagicMock(name="executorch_backend_config")

        with tempfile.TemporaryDirectory() as tmp:
            _, _, edge_prog_mgr = _compile(
                DefaultCompilerAdapter(),
                Path(tmp),
                extra_options={"executorch_backend_config": override},
            )

        self.assertIs(edge_prog_mgr.to_executorch.call_args.args[0], override)


class TestDefaultCompilerAdapterEtrecord(unittest.TestCase):

    def test_no_etrecord_unless_requested(self):
        """ETRecord is not retrieved when it was not requested."""
        with tempfile.TemporaryDirectory() as tmp:
            result, _, edge_prog_mgr = _compile(DefaultCompilerAdapter(), Path(tmp))

        self.assertIsNone(result.etrecord)
        edge_prog_mgr.to_executorch.return_value.get_etrecord.assert_not_called()

    def test_returns_etrecord_when_requested(self):
        """Requesting an ETRecord both enables it in lowering and returns it."""
        with tempfile.TemporaryDirectory() as tmp:
            result, lower, edge_prog_mgr = _compile(
                DefaultCompilerAdapter(),
                Path(tmp),
                extra_options={"generate_etrecord": True},
            )

        self.assertTrue(lower.call_args.kwargs["generate_etrecord"])
        self.assertIs(
            result.etrecord,
            edge_prog_mgr.to_executorch.return_value.get_etrecord.return_value,
        )


class TestDefaultCompilerAdapterValidation(unittest.TestCase):

    def test_missing_example_inputs_raises_before_lowering(self):
        """Without example_inputs the adapter raises rather than calling lowering."""
        adapter = DefaultCompilerAdapter()

        with patch(_LOWER) as lower:
            with self.assertRaises(ValueError) as cm:
                adapter.compile_model(
                    model=MagicMock(name="model"),
                    example_inputs=None,
                    compile_specs=[MagicMock(name="compile_spec")],
                    artifact_dir=Path("/tmp/never_written"),
                    file_name=_FILE_NAME,
                    soc_model=TEST_SOC_CHIPSET,
                    backend_type=TEST_BACKEND_TYPE,
                )

        self.assertIn("example_inputs", str(cm.exception))
        lower.assert_not_called()


class TestDefaultCompilerAdapterTargetConsistency(unittest.TestCase):
    """Uses real compile specs, whose target is readable back out of them.

    Lowering ignores the ``soc_model`` / ``backend_type`` arguments in favour of
    the specs, so a disagreement between the two would otherwise only surface on
    device.
    """

    def test_accepts_a_target_matching_the_specs(self):
        """The SoC and backend the specs carry are accepted."""
        specs = QnnCompileSpecBuilder(soc_model=TEST_SOC_CHIPSET).build()

        with tempfile.TemporaryDirectory() as tmp:
            result, lower, _ = _compile(
                DefaultCompilerAdapter(),
                Path(tmp),
                compile_specs=specs,
            )

            lower.assert_called_once()
            self.assertEqual(list(result.artifact_paths), [ARTIFACT_TEXT_DECODER])

    def test_accepts_a_soc_name_matching_the_specs(self):
        """A string SoC is resolved before comparing, as the pipeline passes one.

        ``PipelineContext`` carries ``soc_model`` as a name, so the check must
        normalise rather than compare a ``str`` against a ``QcomChipset``.
        """
        specs = QnnCompileSpecBuilder(soc_model=TEST_SOC_CHIPSET).build()

        with tempfile.TemporaryDirectory() as tmp:
            _, lower, _ = _compile(
                DefaultCompilerAdapter(),
                Path(tmp),
                compile_specs=specs,
                soc_model=TEST_SOC_MODEL,
                backend_type=str(TEST_BACKEND_TYPE),
            )

            lower.assert_called_once()

    def test_soc_name_contradicting_the_specs_raises_before_lowering(self):
        """A string naming another SoC still fails the check."""
        specs = QnnCompileSpecBuilder(soc_model=QcomChipset.SM8650).build()
        adapter = DefaultCompilerAdapter()

        with patch(_LOWER) as lower:
            with self.assertRaises(ValueError) as cm:
                adapter.compile_model(
                    model=MagicMock(name="model"),
                    example_inputs=(MagicMock(name="example_input"),),
                    compile_specs=specs,
                    artifact_dir=Path("/tmp/never_written"),
                    file_name=_FILE_NAME,
                    soc_model=TEST_SOC_MODEL,
                    backend_type=TEST_BACKEND_TYPE,
                )

        self.assertIn("soc_model", str(cm.exception))
        lower.assert_not_called()

    def test_soc_model_contradicting_the_specs_raises_before_lowering(self):
        """A SoC other than the specs' fails rather than being ignored."""
        specs = QnnCompileSpecBuilder(soc_model=QcomChipset.SM8650).build()
        adapter = DefaultCompilerAdapter()

        with patch(_LOWER) as lower:
            with self.assertRaises(ValueError) as cm:
                adapter.compile_model(
                    model=MagicMock(name="model"),
                    example_inputs=(MagicMock(name="example_input"),),
                    compile_specs=specs,
                    artifact_dir=Path("/tmp/never_written"),
                    file_name=_FILE_NAME,
                    soc_model=QcomChipset.SM8750,
                    backend_type=TEST_BACKEND_TYPE,
                )

        self.assertIn("soc_model", str(cm.exception))
        lower.assert_not_called()

    def test_backend_type_contradicting_the_specs_raises_before_lowering(self):
        """A backend other than the specs' fails rather than being ignored."""
        specs = QnnCompileSpecBuilder(soc_model=TEST_SOC_CHIPSET).build()
        adapter = DefaultCompilerAdapter()

        with patch(_LOWER) as lower:
            with self.assertRaises(ValueError) as cm:
                adapter.compile_model(
                    model=MagicMock(name="model"),
                    example_inputs=(MagicMock(name="example_input"),),
                    compile_specs=specs,
                    artifact_dir=Path("/tmp/never_written"),
                    file_name=_FILE_NAME,
                    soc_model=TEST_SOC_CHIPSET,
                    backend_type=QnnExecuTorchBackendType.kGpuBackend,
                )

        self.assertIn("backend_type", str(cm.exception))
        lower.assert_not_called()

    def test_specs_without_a_qnn_entry_are_not_checked(self):
        """A stub spec list disables the check rather than failing the call."""
        with tempfile.TemporaryDirectory() as tmp:
            _, lower, _ = _compile(
                DefaultCompilerAdapter(),
                Path(tmp),
                compile_specs=[MagicMock(name="not_a_qnn_spec")],
            )

            lower.assert_called_once()

    def test_an_undecodable_qnn_spec_raises_rather_than_skipping_the_check(self):
        """A QNN spec that fails to decode surfaces instead of disabling the check.

        Swallowing it would leave a schema change looking like a graph with no
        QNN spec, silently compiling for whatever the specs really say.
        """
        spec = SimpleNamespace(key=QCOM_QNN_COMPILE_SPEC, value=b"not-a-buffer")
        adapter = DefaultCompilerAdapter()

        with patch(_DECODE, side_effect=RuntimeError("schema mismatch")):
            with patch(_LOWER) as lower:
                with self.assertRaises(RuntimeError):
                    adapter.compile_model(
                        model=MagicMock(name="model"),
                        example_inputs=(MagicMock(name="example_input"),),
                        compile_specs=[spec],
                        artifact_dir=Path("/tmp/never_written"),
                        file_name=_FILE_NAME,
                        soc_model=TEST_SOC_CHIPSET,
                        backend_type=TEST_BACKEND_TYPE,
                    )

        lower.assert_not_called()


if __name__ == "__main__":
    unittest.main()
