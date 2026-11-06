# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from executorch.backends.qualcomm.genai_pipeline.compilation import (
    QnnCompileSpecBuilder,
)
from executorch.backends.qualcomm.genai_pipeline.graph_bundle import GraphBundle
from executorch.backends.qualcomm.genai_pipeline.graph_names import (
    GRAPH_KV_FORWARD,
    GRAPH_PREFILL_FORWARD,
    GRAPH_TOK_EMBEDDING_KV_FORWARD,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.compilation.llm_compiler_adapter import (
    LLMCompilerAdapter,
)
from executorch.backends.qualcomm.genai_pipeline.tests.test_utils import (
    TEST_BACKEND_TYPE,
    TEST_SOC_CHIPSET,
    TEST_SOC_MODEL,
)

_LOWER = "executorch.backends.qualcomm.utils.utils.to_edge_transform_and_lower_to_qnn"
_SPILL_FILL = "executorch.backends.qualcomm.utils.utils.update_spill_fill_size"
_PRINT_DELEGATION = "executorch.devtools.backend_debug.print_delegation_info"
_FILE_NAME = "test_decoder"

# The custom op the sharding pass inserts to mark split points.
_FALLBACK_OP = "llama.fallback.default"


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


def _make_specs():
    """Build real decoder compile specs for the test SoC."""
    return QnnCompileSpecBuilder(soc_model=TEST_SOC_MODEL).build(use_mha2sha=True)


def _compile_group(adapter, artifact_dir, graphs, **overrides):
    """Invoke compile_graph_group with the external calls mocked.

    Returns:
        A (result, lower_mock, edge_prog_mgr) tuple.
    """
    specs = _make_specs()
    kwargs = {
        "graphs": graphs,
        "compile_specs": {name: specs for name in graphs},
        "artifact_dir": artifact_dir,
        "file_name": _FILE_NAME,
        "soc_model": TEST_SOC_CHIPSET,
        "backend_type": TEST_BACKEND_TYPE,
        "executorch_config": MagicMock(name="executorch_config"),
    }
    kwargs.update(overrides)

    edge_prog_mgr = _make_edge_program_manager()
    with patch(_LOWER, return_value=edge_prog_mgr) as lower:
        with patch(_SPILL_FILL):
            result = adapter.compile_graph_group(**kwargs)

    return result, lower, edge_prog_mgr


class TestLLMCompilerAdapter(unittest.TestCase):

    def test_never_delegates_the_fallback_op(self):
        """The sharding marker must stay outside the delegated partitions."""
        graphs = {GRAPH_KV_FORWARD: _make_bundle()}

        with tempfile.TemporaryDirectory() as tmp:
            _, lower, _ = _compile_group(LLMCompilerAdapter(), Path(tmp), graphs)

        self.assertEqual(lower.call_args.kwargs["skip_node_op_set"], {_FALLBACK_OP})

    def test_unions_rather_than_replaces_a_caller_skip_set(self):
        """A caller's own skipped op must not re-enable the marker's delegation."""
        graphs = {GRAPH_KV_FORWARD: _make_bundle()}

        with tempfile.TemporaryDirectory() as tmp:
            _, lower, _ = _compile_group(
                LLMCompilerAdapter(),
                Path(tmp),
                graphs,
                extra_options={"skip_node_op_set": {"my.op.default"}},
            )

        self.assertEqual(
            lower.call_args.kwargs["skip_node_op_set"],
            {_FALLBACK_OP, "my.op.default"},
        )

    def test_takes_constant_methods_from_the_decode_graph(self):
        """Decode's meta is authoritative: the runtime uses its KV scales."""
        decode_meta = {"get_n_layers": 4, "get_kv_output_0_quant_attr": "decode"}
        graphs = {
            GRAPH_PREFILL_FORWARD: _make_bundle(meta={"get_n_layers": 999}),
            GRAPH_KV_FORWARD: _make_bundle(meta=decode_meta),
        }

        with tempfile.TemporaryDirectory() as tmp:
            _, lower, _ = _compile_group(LLMCompilerAdapter(), Path(tmp), graphs)

        self.assertEqual(lower.call_args.kwargs["constant_methods"], decode_meta)

    def test_lowers_without_constant_methods_when_there_is_no_decode_graph(self):
        """The token embedding carries no constants of its own."""
        graphs = {GRAPH_TOK_EMBEDDING_KV_FORWARD: _make_bundle(meta={"unused": 1})}

        with tempfile.TemporaryDirectory() as tmp:
            _, lower, _ = _compile_group(LLMCompilerAdapter(), Path(tmp), graphs)

        self.assertIsNone(lower.call_args.kwargs["constant_methods"])

    def test_prefers_caller_supplied_constant_methods(self):
        """An explicit set overrides the decode-graph default."""
        constants = {"get_n_layers": 1}
        graphs = {GRAPH_KV_FORWARD: _make_bundle(meta={"get_n_layers": 999})}

        with tempfile.TemporaryDirectory() as tmp:
            _, lower, _ = _compile_group(
                LLMCompilerAdapter(),
                Path(tmp),
                graphs,
                constant_methods=constants,
            )

        self.assertEqual(lower.call_args.kwargs["constant_methods"], constants)

    def test_prints_delegation_info_per_graph_when_verbose(self):
        """Verbose reporting matches the legacy flow's per-graph output."""
        graphs = {
            GRAPH_KV_FORWARD: _make_bundle(),
            GRAPH_PREFILL_FORWARD: _make_bundle(),
        }

        with tempfile.TemporaryDirectory() as tmp:
            with patch(_PRINT_DELEGATION) as print_delegation:
                _compile_group(LLMCompilerAdapter(verbose=True), Path(tmp), graphs)

        self.assertEqual(print_delegation.call_count, len(graphs))

    def test_stays_quiet_when_not_verbose(self):
        """Delegation info is opt-in."""
        graphs = {GRAPH_KV_FORWARD: _make_bundle()}

        with tempfile.TemporaryDirectory() as tmp:
            with patch(_PRINT_DELEGATION) as print_delegation:
                _compile_group(LLMCompilerAdapter(), Path(tmp), graphs)

        print_delegation.assert_not_called()


if __name__ == "__main__":
    unittest.main()
