# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from executorch.devtools.backend_debug import get_delegation_info
from executorch.examples.models.llama.export_llama_lib import (
    _export_llama,
    build_args_parser,
)
from executorch.extension.llm.export.config.llm_config import LlmConfig

UNWANTED_OPS = [
    "aten_permute_copy_default",
    "aten_transpose_copy_default",
]


class ExportLlamaLibTest(unittest.TestCase):
    def test_has_expected_ops_and_op_counts(self):
        """
        Checks the presence of unwanted expensive ops.

        Serves as a proxy for a performance regression test, as performance
        is directly tied to which and how many of each ops are in the graph.

        If this test breaks, please ensure that the difference in ops
        is intentional before updating the expected ops.
        """
        # Since we aren't loading a checkpoint, it doesn't
        # matter what model we specify. Note that
        # we cannot test quantization args in this way
        # since quantization requires promoting meta tensors
        # to device=cpu, which requires real weights.
        parser = build_args_parser()
        args = parser.parse_args([])
        args.use_sdpa_with_kv_cache = True
        args.use_kv_cache = True
        args.verbose = True

        llm_config = LlmConfig.from_args(args)
        builder = _export_llama(llm_config)
        graph_module = builder.edge_manager.exported_program().graph_module
        delegation_info = get_delegation_info(graph_module)

        for op, _op_info in delegation_info.delegation_by_operator.items():
            self.assertTrue(op not in UNWANTED_OPS)
