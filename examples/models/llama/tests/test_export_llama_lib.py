# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import tempfile
import unittest
from pathlib import Path

from executorch.devtools.backend_debug import get_delegation_info

try:
    from executorch.backends.arm.quantizer.arm_quantizer import (
        EthosUQuantizer,
        TOSAQuantizer,
        VgfQuantizer,
    )

    HAS_ARM_BACKEND = True
except ImportError:
    HAS_ARM_BACKEND = False
    EthosUQuantizer = None
    TOSAQuantizer = None
    VgfQuantizer = None

from executorch.examples.models.llama.export_llama_lib import (
    _export_llama,
    _prepare_for_llama_export,
    build_args_parser,
    get_quantizer_and_quant_params,
)
from executorch.extension.llm.export.config.llm_config import LlmConfig, Pt2eQuantize

UNWANTED_OPS = [
    "aten_permute_copy_default",
    "aten_transpose_copy_default",
]


class ExportLlamaLibTest(unittest.TestCase):
    def _make_tiny_qwen35_params(self) -> dict:
        return {
            "dim": 64,
            "hidden_dim": 128,
            "n_heads": 4,
            "head_dim": 16,
            "n_kv_heads": 2,
            "n_layers": 4,
            "norm_eps": 1e-6,
            "rope_theta": 10000000.0,
            "use_scaled_rope": False,
            "vocab_size": 256,
            "use_hf_rope": True,
            "partial_rotary_factor": 0.25,
            "attention_qkv_bias": False,
            "use_qk_norm": True,
            "qk_norm_before_rope": True,
            "attention_type": "mha",
            "use_q_gate": True,
            "rms_norm_add_unit_offset": True,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_num_key_heads": 4,
            "linear_num_value_heads": 4,
            "layer_types": [
                "linear_attention",
                "full_attention",
                "linear_attention",
                "full_attention",
            ],
        }

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

    def test_tiny_qwen35_export_uses_recurrent_gated_delta_rule(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            params_path = Path(temp_dir) / "tiny_qwen35.json"
            params_path.write_text(json.dumps(self._make_tiny_qwen35_params()))

            parser = build_args_parser()
            args = parser.parse_args(
                [
                    "--model",
                    "qwen3_5_0_8b",
                    "--params",
                    str(params_path),
                    "--use_kv_cache",
                    "--disable_dynamic_shape",
                    "--max_seq_length",
                    "8",
                    "--max_context_length",
                    "8",
                ]
            )

            llm_config = LlmConfig.from_args(args)
            builder = _prepare_for_llama_export(llm_config).export()
            assert builder.pre_autograd_graph_module is not None

            recurrent_nodes = [
                node
                for node in builder.pre_autograd_graph_module.graph.nodes
                if "auto_functionalized_v2" in str(node.target)
                and node.args
                and "llama.recurrent_gated_delta_rule" in str(node.args[0])
            ]

            self.assertEqual(len(recurrent_nodes), 2)

    @unittest.skipUnless(HAS_ARM_BACKEND, "ARM backend not available")
    def test_get_quantizer_and_quant_params_returns_tosa_quantizer(self):
        llm_config = LlmConfig()
        llm_config.backend.tosa.enabled = True
        llm_config.quantization.pt2e_quantize = Pt2eQuantize.tosa_8a8w

        pt2e_quant_params, quantizers, quant_dtype = get_quantizer_and_quant_params(
            llm_config
        )

        self.assertIsNone(pt2e_quant_params)
        self.assertIsNone(quant_dtype)
        self.assertEqual(len(quantizers), 1)
        self.assertIsInstance(quantizers[0], TOSAQuantizer)

    @unittest.skipUnless(HAS_ARM_BACKEND, "ARM backend not available")
    def test_get_quantizer_and_quant_params_returns_ethosu_quantizer(self):
        llm_config = LlmConfig()
        llm_config.backend.ethosu.enabled = True
        llm_config.quantization.pt2e_quantize = Pt2eQuantize.ethosu_8a8w

        pt2e_quant_params, quantizers, quant_dtype = get_quantizer_and_quant_params(
            llm_config
        )

        self.assertIsNone(pt2e_quant_params)
        self.assertIsNone(quant_dtype)
        self.assertEqual(len(quantizers), 1)
        self.assertIsInstance(quantizers[0], EthosUQuantizer)

    @unittest.skipUnless(HAS_ARM_BACKEND, "ARM backend not available")
    def test_get_quantizer_and_quant_params_returns_vgf_quantizer(self):
        llm_config = LlmConfig()
        llm_config.backend.vgf.enabled = True
        llm_config.backend.vgf.compile_spec = "TOSA-1.0+INT"
        llm_config.quantization.pt2e_quantize = Pt2eQuantize.vgf_8a8w

        pt2e_quant_params, quantizers, quant_dtype = get_quantizer_and_quant_params(
            llm_config
        )

        self.assertIsNone(pt2e_quant_params)
        self.assertIsNone(quant_dtype)
        self.assertEqual(len(quantizers), 1)
        self.assertIsInstance(quantizers[0], VgfQuantizer)
