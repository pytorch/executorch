# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from executorch.devtools.backend_debug import get_delegation_info
from executorch.examples.models.llama.export_llama_lib import _export_llama, build_args_parser


# Ops expected to be found in the default exported llama_transformer. Obtained through
# print_delegation_info from the backend_debug module, which is displayed with
# export_llama under --verbose.
BASE_EXPECTED_OPS = {
    "sym_size": 1,
    "alloc": 288,
    "aten_embedding_default": 1,
    "aten_select_copy_int": 12,
    "_local_scalar_dense": 11,
    "add": 1,
    "aten_slice_copy_tensor": 23,
    "aten_mul_tensor": 83,
    "aten_mean_dim": 11,
    "aten_add_tensor": 31,
    "aten_rsqrt_default": 11,
    "aten_linear_default": 36,
    "aten_view_copy_default": 40,
    "aten_squeeze_copy_dims": 20,
    "aten_sub_tensor": 10,
    "aten_unsqueeze_copy_default": 20,
    "aten_cat_default": 10,
    "update_cache": 10,
    "llama_custom_sdpa_default": 5,
    "aten_sigmoid_default": 5,
}

UNWANTED_OPS = [
    "aten_permute_copy_default",
]

class ExportLlamaLibTest(unittest.TestCase):
    def test_has_expected_ops_and_op_counts(self):
        """
        Tests that the presence of expected ops and counts for each op are
        do not change.

        Serves as a proxy for a performance regression test, as performance
        is directly tied to which and how many of each ops are in the graph.

        If this test breaks, please ensure that the difference in ops
        is intentional before updating the expected ops.
        """
        # Since we aren't loading a checkpoint, it doesn't
        # matter what model we specify. Note that
        # we cannot test quantization args in this way
        # since quantization requires promoting meta tensors
        # to the cpu device, which requires real weights.
        export_args_str = """
            --use_sdpa_with_kv_cache
            -kv
            --verbose
        """
        args_list = export_args_str.strip().split()
        parser = build_args_parser()
        args = parser.parse_args(args_list)

        builder = _export_llama(args)
        graph_module = builder.edge_manager.exported_program().graph_module
        delegation_info = get_delegation_info(graph_module)

        for op, op_info in delegation_info.delegation_by_operator.items():
            self.assertTrue(op in BASE_EXPECTED_OPS)
            self.assertTrue(op not in UNWANTED_OPS)
            self.assertEqual(op_info.non_delegated, BASE_EXPECTED_OPS[op])

        self.assertEqual(len(delegation_info.delegation_by_operator), len(BASE_EXPECTED_OPS))
        
