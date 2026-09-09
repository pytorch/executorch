# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
import unittest
from unittest.mock import patch

import torch
from executorch.devtools.backend_debug import get_delegation_info
from executorch.devtools.etrecord import parse_etrecord
from executorch.devtools.etrecord._etrecord import ETRecord

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

from executorch.examples.models.llama import export_llama_lib
from executorch.examples.models.llama.export_llama_lib import (
    _export_llama,
    build_args_parser,
    get_quantizer_and_quant_params,
)
from executorch.extension.llm.export.builder import LLMEdgeManager
from executorch.extension.llm.export.config.llm_config import (
    LlmConfig,
    MethodConfig,
    Pt2eQuantize,
    VgfQuantizeScope,
)

UNWANTED_OPS = [
    "aten_permute_copy_default",
    "aten_transpose_copy_default",
]


def _tiny_llm_builder():
    """An LLMEdgeManager small enough to lower in a unit test.

    `_export_llama` exports it, so this returns it unexported.
    """

    class Tiny(torch.nn.Module):
        def forward(self, tokens):
            return tokens.to(torch.float32) * 2.0 + 1.0

    return LLMEdgeManager(
        model=Tiny(),
        modelname="tiny",
        max_seq_len=4,
        use_kv_cache=False,
        example_inputs=(torch.ones(1, 4, dtype=torch.long),),
    )


class ExportLlamaLibTest(unittest.TestCase):
    def _assert_routes_to(self, lowering, coreml=False, vulkan=False, qnn=False):
        """Assert which lowering an export routes to, without running one."""
        llm_config = LlmConfig()
        llm_config.backend.coreml.enabled = coreml
        llm_config.backend.vulkan.enabled = vulkan
        llm_config.backend.qnn.enabled = qnn
        # _validate_args rejects dynamic shapes when Core ML or QNN is enabled.
        llm_config.model.enable_dynamic_shape = False
        # With the KV cache on, the source transforms import the Qualcomm SDK, which routing
        # does not need and which is not present on most machines.
        llm_config.model.use_kv_cache = False

        class Reached(Exception):
            pass

        # Routing is decided before the model is touched, so the export is stubbed out: without
        # this each case traces and lowers the whole model to check one branch.
        with patch.object(
            export_llama_lib, lowering, side_effect=Reached
        ) as target, patch.object(export_llama_lib, "_prepare_for_llama_export"):
            with self.assertRaises(Reached):
                _export_llama(llm_config)
        target.assert_called_once()
        return target

    def test_core_ml_alone_routes_to_the_core_ml_lowering(self):
        """Core ML on its own must reach the Core ML lowering.

        The guard read a backend config field that no longer exists, so this raised AttributeError
        before reaching any lowering.
        """
        self._assert_routes_to("_to_edge_and_lower_llama_coreml", coreml=True)

    def test_core_ml_with_qnn_routes_to_the_combined_lowering(self):
        """Core ML with QNN must keep the QNN partitioner, so it takes the combined lowering."""
        target = self._assert_routes_to(
            "_to_edge_and_lower_llama", coreml=True, qnn=True
        )
        self.assertTrue(target.call_args.kwargs["coreml"])
        self.assertTrue(target.call_args.kwargs["qnn"])

    def test_core_ml_with_vulkan_routes_to_the_combined_lowering(self):
        """Core ML with Vulkan must keep the Vulkan partitioner the same way.

        This is what pins the Vulkan half of the exclusion clause: the Core ML lowering takes no
        Vulkan argument, so routing there would drop the partitioner silently.
        """
        target = self._assert_routes_to(
            "_to_edge_and_lower_llama", coreml=True, vulkan=True
        )
        self.assertTrue(target.call_args.kwargs["coreml"])
        self.assertTrue(target.call_args.kwargs["vulkan"])

    def _run_tiny_export(
        self,
        generate_etrecord,
        directory,
        output_name="tiny.pte",
        output_dir=None,
        xnnpack=True,
    ):
        """Export the tiny model, writing the .pte into `directory`.

        The builder is pointed at a directory rather than chdir'ing the process, because these
        tests run under pytest-xdist where the cwd is shared between tests in a worker.

        `output_dir` defaults to `directory`, and is set separately only by the test that
        checks a `.pte` output name does not separate the model from its record.

        `xnnpack=False` selects the combined lowering, which reaches the edge export by a
        different route than the backend-specific helpers.
        """
        llm_config = LlmConfig()
        llm_config.backend.xnnpack.enabled = xnnpack
        llm_config.debug.generate_etrecord = generate_etrecord
        llm_config.model.enable_dynamic_shape = False
        llm_config.export.output_dir = output_dir or directory
        llm_config.export.output_name = os.path.join(directory, output_name)
        builder = _tiny_llm_builder().set_output_dir(output_dir or directory)
        with patch(
            "executorch.examples.models.llama.export_llama_lib._prepare_for_llama_export",
            return_value=builder,
        ):
            _export_llama(llm_config)
        return sorted(os.listdir(directory))

    def test_combined_lowering_saves_the_etrecord_beside_the_model(self):
        """The combined lowering must write a record too.

        It reaches the edge export by a different route than the backend helpers, so the flag
        has to be set on the builder before that export rather than at the lowering.
        """
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(
                self._run_tiny_export(
                    generate_etrecord=True, directory=directory, xnnpack=False
                ),
                ["etrecord.bin", "tiny.pte"],
            )
            self.assertIsNotNone(
                parse_etrecord(
                    os.path.join(directory, "etrecord.bin")
                ).edge_dialect_program
            )

    def test_export_saves_the_etrecord_beside_the_model(self):
        """The record must be written, and land beside the model.

        A `.pte` output name is used as the path verbatim, so `output_dir` points somewhere
        else here: a record keyed off `output_dir` would separate it from the model.
        """
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as elsewhere:
            self.assertEqual(
                self._run_tiny_export(
                    generate_etrecord=True, directory=directory, output_dir=elsewhere
                ),
                ["etrecord.bin", "tiny.pte"],
            )
            # An empty file would satisfy the listing, so load it back.
            self.assertIsNotNone(
                parse_etrecord(
                    os.path.join(directory, "etrecord.bin")
                ).edge_dialect_program
            )

    def test_export_writes_no_etrecord_when_not_asked(self):
        """The common case must stay silent rather than raise or leave a stray file."""
        with tempfile.TemporaryDirectory() as directory:
            with self.assertNoLogs(level="WARNING"):
                landed = self._run_tiny_export(
                    generate_etrecord=False, directory=directory
                )
            self.assertEqual(landed, ["tiny.pte"])

    def test_export_keeps_the_model_when_the_etrecord_cannot_be_written(self):
        """A debug artifact must not cost the caller the export.

        The record is written after the model, so a failed record write costs the record and
        not the .pte. A full disk is the likely trigger, since the record is the larger.
        """
        with tempfile.TemporaryDirectory() as directory:
            # A directory of that name makes the rename fail without touching permissions.
            os.mkdir(os.path.join(directory, "etrecord.bin"))
            with self.assertLogs(level="WARNING") as logs:
                landed = self._run_tiny_export(
                    generate_etrecord=True, directory=directory
                )
            self.assertIn("tiny.pte", landed)
            self.assertTrue(any("Could not write" in line for line in logs.output))
            # The staged record must not be left behind next to the model.
            self.assertEqual(landed, ["etrecord.bin", "tiny.pte"])

    def test_export_keeps_the_previous_etrecord_when_a_rewrite_fails(self):
        """A failed rewrite must not destroy the record that was already there.

        The record format truncates its target on open, so writing in place would leave a
        short file under the real name and report only a warning.
        """
        with tempfile.TemporaryDirectory() as directory:
            self._run_tiny_export(generate_etrecord=True, directory=directory)
            record = os.path.join(directory, "etrecord.bin")
            good = os.path.getsize(record)

            with patch.object(
                ETRecord, "_save_graph_map", side_effect=RuntimeError("disk full")
            ):
                with self.assertLogs(level="WARNING"):
                    self._run_tiny_export(generate_etrecord=True, directory=directory)

            self.assertEqual(os.path.getsize(record), good)
            self.assertIsNotNone(parse_etrecord(record).edge_dialect_program)

    def test_multimethod_export_saves_the_etrecord_beside_the_model(self):
        """The multimethod path must write the record too.

        It builds a record the same way the single-method paths do, so leaving out the save
        would reproduce the silent-flag bug on that path alone.
        """
        with tempfile.TemporaryDirectory() as directory:
            llm_config = LlmConfig()
            llm_config.backend.xnnpack.enabled = True
            llm_config.debug.generate_etrecord = True
            llm_config.multimethod.methods = [MethodConfig(method_name="forward")]
            llm_config.export.output_dir = directory
            llm_config.export.output_name = os.path.join(directory, "tiny.pte")
            with patch.object(
                export_llama_lib,
                "_prepare_for_llama_export",
                side_effect=lambda _: _tiny_llm_builder().set_output_dir(directory),
            ):
                _export_llama(llm_config)
            self.assertEqual(
                sorted(os.listdir(directory)), ["etrecord.bin", "tiny.pte"]
            )

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

    def test_bf16_xnnpack_delegates_linears_when_enabled(self):
        parser = build_args_parser()
        args = parser.parse_args([])
        args.use_kv_cache = True
        args.xnnpack = True
        args.xnnpack_extended_ops = True
        args.xnnpack_enable_bf16 = True
        args.dtype_override = "bf16"

        llm_config = LlmConfig.from_args(args)
        builder = _export_llama(llm_config)
        graph_module = builder.edge_manager.exported_program().graph_module
        delegation_info = get_delegation_info(graph_module)

        linear = delegation_info.delegation_by_operator["aten_linear_default"]
        self.assertGreater(linear.delegated, 0)
        self.assertEqual(linear.non_delegated, 0)

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
