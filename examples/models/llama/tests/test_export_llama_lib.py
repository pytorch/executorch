# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

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
    build_args_parser,
    get_quantizer_and_quant_params,
)
from executorch.extension.llm.export.config.llm_config import (
    LlmConfig,
    Pt2eQuantize,
    VgfQuantizeScope,
)

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
        self.assertIsNotNone(quantizers[0].global_config)
        self.assertEqual(quantizers[0].module_type_config, {})

    @unittest.skipUnless(HAS_ARM_BACKEND, "ARM backend not available")
    def test_get_quantizer_and_quant_params_returns_vgf_linear_quantizer(self):
        llm_config = LlmConfig()
        llm_config.backend.vgf.enabled = True
        llm_config.backend.vgf.compile_spec = "TOSA-1.0+INT"
        llm_config.backend.vgf.quantize_scope = VgfQuantizeScope.linear
        llm_config.quantization.pt2e_quantize = Pt2eQuantize.vgf_8a8w

        _pt2e_quant_params, quantizers, _quant_dtype = get_quantizer_and_quant_params(
            llm_config
        )

        self.assertEqual(len(quantizers), 1)
        self.assertIsInstance(quantizers[0], VgfQuantizer)
        self.assertIsNone(quantizers[0].global_config)
        self.assertIn(torch.nn.Linear, quantizers[0].module_type_config)

    @unittest.skipUnless(HAS_ARM_BACKEND, "ARM backend not available")
    def test_vgf_16a8w_requires_int16_compile_spec_extension(self):
        llm_config = LlmConfig()
        llm_config.backend.vgf.enabled = True
        llm_config.backend.vgf.compile_spec = "TOSA-1.0+INT"
        llm_config.backend.vgf.quantize_scope = VgfQuantizeScope.linear
        llm_config.quantization.pt2e_quantize = Pt2eQuantize.vgf_16a8w

        with self.assertRaisesRegex(ValueError, "INT16 support"):
            get_quantizer_and_quant_params(llm_config)

    @unittest.skipUnless(HAS_ARM_BACKEND, "ARM backend not available")
    def test_vgf_16a8w_accepts_int16_compile_spec_extension(self):
        llm_config = LlmConfig()
        llm_config.backend.vgf.enabled = True
        llm_config.backend.vgf.compile_spec = "TOSA-1.0+INT+int16"
        llm_config.backend.vgf.quantize_scope = VgfQuantizeScope.linear
        llm_config.quantization.pt2e_quantize = Pt2eQuantize.vgf_16a8w

        _pt2e_quant_params, quantizers, _quant_dtype = get_quantizer_and_quant_params(
            llm_config
        )

        self.assertEqual(len(quantizers), 1)
        self.assertIsInstance(quantizers[0], VgfQuantizer)
        self.assertIn(torch.nn.Linear, quantizers[0].module_type_config)
