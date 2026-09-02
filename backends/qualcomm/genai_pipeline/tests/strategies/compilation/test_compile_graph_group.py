# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
    ARTIFACT_TEXT_DECODER,
)
from executorch.backends.qualcomm.genai_pipeline.compilation import (
    QnnCompileSpecBuilder,
)
from executorch.backends.qualcomm.genai_pipeline.graph_bundle import GraphBundle
from executorch.backends.qualcomm.genai_pipeline.graph_names import (
    GRAPH_KV_FORWARD,
    GRAPH_PREFILL_FORWARD,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.compilation.default_compiler_adapter import (
    DefaultCompilerAdapter,
)
from executorch.backends.qualcomm.genai_pipeline.tests.test_utils import (
    TEST_BACKEND_TYPE,
    TEST_SOC_CHIPSET,
    TEST_SOC_MODEL,
)

# The adapter imports both helpers lazily from their defining module, so
# patching there intercepts the calls.
_LOWER = "executorch.backends.qualcomm.utils.utils.to_edge_transform_and_lower_to_qnn"
_SPILL_FILL = "executorch.backends.qualcomm.utils.utils.update_spill_fill_size"

_FILE_NAME = "test_decoder"


def _make_edge_program_manager():
    """Create a mock EdgeProgramManager whose program writes bytes to file."""
    exec_prog_mgr = MagicMock(name="exec_prog_mgr")
    exec_prog_mgr.write_to_file.side_effect = lambda f: f.write(b"pte-bytes")

    edge_prog_mgr = MagicMock(name="edge_prog_mgr")
    edge_prog_mgr.to_executorch.return_value = exec_prog_mgr
    return edge_prog_mgr


def _make_bundle(**overrides) -> GraphBundle:
    """Create a GraphBundle with distinguishable placeholder members."""
    kwargs = {
        "module": MagicMock(name="module"),
        "inputs": (MagicMock(name="input"),),
        "meta": {},
    }
    kwargs.update(overrides)
    return GraphBundle(**kwargs)


def _make_specs(num_sharding: int = 1):
    """Build real compile specs for the decoder, sharded or not."""
    builder = QnnCompileSpecBuilder(soc_model=TEST_SOC_MODEL)
    return builder.build(use_multi_contexts=num_sharding > 1, use_mha2sha=True)


def _compile_group(adapter, artifact_dir, graphs, compile_specs, **overrides):
    """Invoke compile_graph_group with lowering and spill-fill mocked.

    Returns:
        A (result, lower_mock, spill_fill_mock) tuple.
    """
    kwargs = {
        "graphs": graphs,
        "compile_specs": compile_specs,
        "artifact_dir": artifact_dir,
        "file_name": _FILE_NAME,
        "soc_model": TEST_SOC_CHIPSET,
        "backend_type": TEST_BACKEND_TYPE,
        "executorch_config": MagicMock(name="executorch_config"),
    }
    kwargs.update(overrides)

    with patch(_LOWER, return_value=_make_edge_program_manager()) as lower:
        with patch(_SPILL_FILL) as spill_fill:
            result = adapter.compile_graph_group(**kwargs)

    return result, lower, spill_fill


class TestCompileGraphGroup(unittest.TestCase):

    def test_lowers_all_graphs_in_one_keyed_call(self):
        """Both graphs reach one lowering call as graph-name-keyed dicts."""
        decode, prefill = _make_bundle(), _make_bundle()
        graphs = {GRAPH_KV_FORWARD: decode, GRAPH_PREFILL_FORWARD: prefill}
        specs = _make_specs()
        constants = {"get_n_layers": 4}
        dep_table = {name: {"dep": ["table"]} for name in graphs}
        passes_job = {name: MagicMock(name=f"passes_job_{name}") for name in graphs}

        with tempfile.TemporaryDirectory() as tmp:
            result, lower, _ = _compile_group(
                DefaultCompilerAdapter(),
                Path(tmp),
                graphs,
                {name: specs for name in graphs},
                constant_methods=constants,
                dep_table=dep_table,
                passes_job=passes_job,
            )

        lower.assert_called_once()
        kwargs = lower.call_args.kwargs
        self.assertEqual(
            kwargs["module"],
            {GRAPH_KV_FORWARD: decode.module, GRAPH_PREFILL_FORWARD: prefill.module},
        )
        self.assertEqual(
            kwargs["inputs"],
            {GRAPH_KV_FORWARD: decode.inputs, GRAPH_PREFILL_FORWARD: prefill.inputs},
        )
        self.assertEqual(kwargs["dep_table"], dep_table)
        self.assertEqual(kwargs["passes_job"], passes_job)
        self.assertEqual(kwargs["constant_methods"], constants)
        self.assertEqual(set(result.artifact_paths), {ARTIFACT_TEXT_DECODER})

    def test_writes_one_pte_for_the_whole_group(self):
        """Several graphs produce a single artifact, so they share weights."""
        graphs = {
            GRAPH_KV_FORWARD: _make_bundle(),
            GRAPH_PREFILL_FORWARD: _make_bundle(),
        }
        specs = _make_specs()

        with tempfile.TemporaryDirectory() as tmp:
            result, _, _ = _compile_group(
                DefaultCompilerAdapter(),
                Path(tmp),
                graphs,
                {name: specs for name in graphs},
            )

            pte_path = result.artifact_paths[ARTIFACT_TEXT_DECODER]
            self.assertEqual(pte_path, Path(tmp) / f"{_FILE_NAME}.pte")
            self.assertTrue(pte_path.exists())
            self.assertEqual(list(Path(tmp).glob("*.pte")), [pte_path])

    def test_sizes_spill_fill_once_per_graph_when_sharded(self):
        """Sharded specs request multiple contexts, so each graph is sized."""
        graphs = {
            GRAPH_KV_FORWARD: _make_bundle(),
            GRAPH_PREFILL_FORWARD: _make_bundle(),
        }
        specs = _make_specs(num_sharding=4)

        with tempfile.TemporaryDirectory() as tmp:
            _, _, spill_fill = _compile_group(
                DefaultCompilerAdapter(),
                Path(tmp),
                graphs,
                {name: specs for name in graphs},
            )

        self.assertEqual(spill_fill.call_count, len(graphs))

    def test_skips_spill_fill_when_not_sharded(self):
        """An unsharded graph never pays for backend init to size nothing."""
        graphs = {GRAPH_KV_FORWARD: _make_bundle()}
        specs = _make_specs(num_sharding=1)

        with tempfile.TemporaryDirectory() as tmp:
            _, _, spill_fill = _compile_group(
                DefaultCompilerAdapter(),
                Path(tmp),
                graphs,
                {name: specs for name in graphs},
            )

        spill_fill.assert_not_called()

    def test_rejects_specs_that_do_not_cover_every_graph(self):
        """A missing per-graph spec fails before any graph work starts."""
        graphs = {
            GRAPH_KV_FORWARD: _make_bundle(),
            GRAPH_PREFILL_FORWARD: _make_bundle(),
        }

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as ctx:
                _compile_group(
                    DefaultCompilerAdapter(),
                    Path(tmp),
                    graphs,
                    {GRAPH_KV_FORWARD: _make_specs()},
                )

        self.assertIn(GRAPH_PREFILL_FORWARD, str(ctx.exception))

    def test_rejects_a_pass_config_that_does_not_cover_every_graph(self):
        """A graph lowered with another graph's passes is silently wrong."""
        graphs = {
            GRAPH_KV_FORWARD: _make_bundle(),
            GRAPH_PREFILL_FORWARD: _make_bundle(),
        }
        specs = _make_specs()

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as ctx:
                _compile_group(
                    DefaultCompilerAdapter(),
                    Path(tmp),
                    graphs,
                    {name: specs for name in graphs},
                    passes_job={GRAPH_KV_FORWARD: MagicMock(name="passes_job")},
                )

        self.assertIn("passes_job", str(ctx.exception))

    def test_rejects_a_dependency_table_that_does_not_cover_every_graph(self):
        """A graph lowered with another graph's pass order is silently wrong."""
        graphs = {
            GRAPH_KV_FORWARD: _make_bundle(),
            GRAPH_PREFILL_FORWARD: _make_bundle(),
        }
        specs = _make_specs()

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as ctx:
                _compile_group(
                    DefaultCompilerAdapter(),
                    Path(tmp),
                    graphs,
                    {name: specs for name in graphs},
                    dep_table={GRAPH_KV_FORWARD: {"dep": ["table"]}},
                )

        self.assertIn("dep_table", str(ctx.exception))

    def test_rejects_a_bundle_without_inputs(self):
        """torch.export cannot trace a graph with no positional signature."""
        graphs = {GRAPH_KV_FORWARD: _make_bundle(inputs=None)}

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as ctx:
                _compile_group(
                    DefaultCompilerAdapter(),
                    Path(tmp),
                    graphs,
                    {GRAPH_KV_FORWARD: _make_specs()},
                )

        self.assertIn("inputs", str(ctx.exception))

    def test_rejects_a_soc_model_contradicting_the_specs(self):
        """Compiling for a SoC other than the specs' would only fail on device."""
        from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset

        graphs = {GRAPH_KV_FORWARD: _make_bundle()}

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                _compile_group(
                    DefaultCompilerAdapter(),
                    Path(tmp),
                    graphs,
                    {GRAPH_KV_FORWARD: _make_specs()},
                    soc_model=QcomChipset.SM8650,
                )

    def test_applies_the_given_executorch_config_to_the_group(self):
        """The strategy's per-component config reaches to_executorch as is."""
        config = MagicMock(name="executorch_config")
        graphs = {GRAPH_KV_FORWARD: _make_bundle()}
        specs = _make_specs()

        edge_prog_mgr = _make_edge_program_manager()
        with tempfile.TemporaryDirectory() as tmp:
            with patch(_LOWER, return_value=edge_prog_mgr):
                with patch(_SPILL_FILL):
                    DefaultCompilerAdapter().compile_graph_group(
                        graphs=graphs,
                        compile_specs={GRAPH_KV_FORWARD: specs},
                        artifact_dir=Path(tmp),
                        file_name=_FILE_NAME,
                        soc_model=TEST_SOC_CHIPSET,
                        backend_type=TEST_BACKEND_TYPE,
                        executorch_config=config,
                    )

        edge_prog_mgr.to_executorch.assert_called_once_with(config)


if __name__ == "__main__":
    unittest.main()
