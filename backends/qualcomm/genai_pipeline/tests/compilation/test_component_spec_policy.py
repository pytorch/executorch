# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from executorch.backends.qualcomm._passes.build_quant_io import BuildQuantIo
from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
    ARTIFACT_TEXT_DECODER,
    ARTIFACT_TOK_EMBEDDING,
    ARTIFACT_VISION_ENCODER,
)
from executorch.backends.qualcomm.genai_pipeline.compilation.compile_spec_builder import (
    QnnCompileSpecBuilder,
)
from executorch.backends.qualcomm.genai_pipeline.compilation.component_spec_policy import (
    build_component_compile_specs,
    build_executorch_config,
)
from executorch.backends.qualcomm.genai_pipeline.control_args import ControlArgs
from executorch.backends.qualcomm.genai_pipeline.graph_names import (
    DECODER_GRAPH_NAMES,
    TOK_EMBEDDING_GRAPH_NAMES,
)
from executorch.backends.qualcomm.genai_pipeline.tests.test_utils import TEST_SOC_MODEL
from executorch.backends.qualcomm.serialization.qc_schema_serialize import (
    flatbuffer_to_option,
)
from executorch.backends.qualcomm.utils.constants import QCOM_QNN_COMPILE_SPEC


def _decode(specs):
    """Recover the QnnExecuTorchOptions carried by a list of compile specs."""
    for spec in specs:
        if spec.key == QCOM_QNN_COMPILE_SPEC:
            return flatbuffer_to_option(spec.value)

    raise AssertionError("no QNN compile spec found")


def _build(artifact_key, graph_names, enable_x86_64=False, quantized=True, **kwargs):
    """Build one component's compile specs."""
    return build_component_compile_specs(
        builder=QnnCompileSpecBuilder(
            soc_model=TEST_SOC_MODEL, enable_x86_64=enable_x86_64
        ),
        artifact_key=artifact_key,
        graph_names=graph_names,
        control_args=ControlArgs(soc_model=TEST_SOC_MODEL, **kwargs),
        quantized=quantized,
    )


class TestComponentSpecPolicy(unittest.TestCase):

    def test_builds_one_spec_entry_per_graph(self):
        """Lowering keys specs by graph name, so every graph needs an entry."""
        specs = _build(ARTIFACT_TEXT_DECODER, DECODER_GRAPH_NAMES)

        self.assertEqual(sorted(specs), sorted(DECODER_GRAPH_NAMES))

    def test_decoder_requests_mha2sha(self):
        """Only the decoder converts multi-head to single-head attention."""
        specs = _build(ARTIFACT_TEXT_DECODER, DECODER_GRAPH_NAMES)

        options = _decode(specs[DECODER_GRAPH_NAMES[0]])
        self.assertTrue(options.use_mha2sha)

    def test_token_embedding_does_not_request_mha2sha(self):
        """The embedding lookup has no attention to convert."""
        specs = _build(ARTIFACT_TOK_EMBEDDING, TOK_EMBEDDING_GRAPH_NAMES)

        options = _decode(specs[TOK_EMBEDDING_GRAPH_NAMES[0]])
        self.assertFalse(options.use_mha2sha)

    def test_sharding_requests_multiple_contexts_for_the_decoder(self):
        """Multiple contexts are what make one spill-fill buffer shareable."""
        specs = build_component_compile_specs(
            builder=QnnCompileSpecBuilder(soc_model=TEST_SOC_MODEL),
            artifact_key=ARTIFACT_TEXT_DECODER,
            graph_names=DECODER_GRAPH_NAMES,
            control_args=ControlArgs(soc_model=TEST_SOC_MODEL),
            num_sharding=4,
        )

        options = _decode(specs[DECODER_GRAPH_NAMES[0]])
        self.assertTrue(options.backend_options.htp_options.use_multi_contexts)

    def test_unsharded_decoder_does_not_request_multiple_contexts(self):
        """One shard means one context, so there is nothing to share."""
        specs = _build(ARTIFACT_TEXT_DECODER, DECODER_GRAPH_NAMES)

        options = _decode(specs[DECODER_GRAPH_NAMES[0]])
        self.assertFalse(options.backend_options.htp_options.use_multi_contexts)

    def test_a_sharded_encoder_still_gets_one_context(self):
        """Only the decoder is sharded, so sharding must not leak to encoders."""
        specs = build_component_compile_specs(
            builder=QnnCompileSpecBuilder(soc_model=TEST_SOC_MODEL),
            artifact_key=ARTIFACT_VISION_ENCODER,
            graph_names=["forward"],
            control_args=ControlArgs(soc_model=TEST_SOC_MODEL),
            num_sharding=4,
        )

        options = _decode(specs["forward"])
        self.assertFalse(options.backend_options.htp_options.use_multi_contexts)

    def test_encoders_do_not_request_weight_sharing(self):
        """An encoder is one graph per .pte, so there is nothing to share."""
        specs = _build(ARTIFACT_VISION_ENCODER, ["forward"])

        options = _decode(specs["forward"])
        self.assertFalse(options.backend_options.htp_options.use_weight_sharing)

    def test_x86_target_disables_weight_sharing_and_shared_buffer(self):
        """The emulator supports neither, and the x86 CI run is a parity gate."""
        specs = _build(ARTIFACT_TEXT_DECODER, DECODER_GRAPH_NAMES, enable_x86_64=True)

        options = _decode(specs[DECODER_GRAPH_NAMES[0]])
        self.assertFalse(options.backend_options.htp_options.use_weight_sharing)
        self.assertFalse(options.shared_buffer)

    def test_device_target_keeps_shared_buffer_on(self):
        """``ControlArgs.shared_buffer`` is a runner flag and must not reach here."""
        specs = _build(ARTIFACT_TEXT_DECODER, DECODER_GRAPH_NAMES, shared_buffer=False)

        options = _decode(specs[DECODER_GRAPH_NAMES[0]])
        self.assertTrue(options.shared_buffer)

    def test_an_unquantized_component_compiles_in_fp16(self):
        """A component whose quantization was skipped needs FP16 specs."""
        from executorch.backends.qualcomm.serialization.qc_schema import (
            QnnExecuTorchHtpPrecision,
        )

        specs = _build(ARTIFACT_VISION_ENCODER, ["forward"], quantized=False)

        options = _decode(specs["forward"])
        self.assertEqual(
            options.backend_options.htp_options.precision,
            QnnExecuTorchHtpPrecision.kHtpFp16,
        )

    def test_a_quantized_component_ignores_the_global_fp16_flag(self):
        """Precision follows the component, not the flow-wide flag."""
        from executorch.backends.qualcomm.serialization.qc_schema import (
            QnnExecuTorchHtpPrecision,
        )

        specs = _build(ARTIFACT_TEXT_DECODER, DECODER_GRAPH_NAMES, use_fp16=True)

        options = _decode(specs[DECODER_GRAPH_NAMES[0]])
        self.assertNotEqual(
            options.backend_options.htp_options.precision,
            QnnExecuTorchHtpPrecision.kHtpFp16,
        )

    def test_encoders_ignore_online_prepare(self):
        """Encoders are composed ahead of time regardless of the flag."""
        specs = _build(ARTIFACT_VISION_ENCODER, ["forward"], online_prepare=True)

        options = _decode(specs["forward"])
        self.assertFalse(options.online_prepare)


class TestBuildExecutorchConfig(unittest.TestCase):

    def test_shared_buffer_components_leave_graph_io_unallocated(self):
        """The runner passes RPC memory for decoder and embedding IO."""
        for artifact_key in (ARTIFACT_TEXT_DECODER, ARTIFACT_TOK_EMBEDDING):
            with self.subTest(artifact_key=artifact_key):
                config = build_executorch_config(artifact_key)

                self.assertFalse(config.memory_planning_pass.alloc_graph_input)
                self.assertFalse(config.memory_planning_pass.alloc_graph_output)

    def test_encoders_let_memory_planning_allocate_graph_io(self):
        """Encoders do not use RPC memory, as in the legacy Modality.compile."""
        config = build_executorch_config(ARTIFACT_VISION_ENCODER)

        self.assertTrue(config.memory_planning_pass.alloc_graph_input)
        self.assertTrue(config.memory_planning_pass.alloc_graph_output)

    def test_every_component_builds_quant_io(self):
        """BuildQuantIo runs, so quantized IO tensors get their types."""
        for artifact_key in (ARTIFACT_TEXT_DECODER, ARTIFACT_VISION_ENCODER):
            with self.subTest(artifact_key=artifact_key):
                config = build_executorch_config(artifact_key)

                self.assertTrue(any(isinstance(p, BuildQuantIo) for p in config.passes))


if __name__ == "__main__":
    unittest.main()
