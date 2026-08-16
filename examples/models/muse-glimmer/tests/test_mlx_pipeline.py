# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MLX integration tests for the Muse Glimmer pipeline.

Run from an OSS ExecuTorch checkout with the MLX backend installed, via
``run.py``:

    python dev/run.py executorch.examples.models.muse_glimmer.tests.test_mlx_pipeline

Backend-agnostic tests live in ``test_pipeline.py``; CUDA tests in
``test_cuda_pipeline.py``.
"""

import os
import tempfile
import unittest
from dataclasses import replace

import torch
from executorch.examples.models.muse_glimmer.model.model import MuseGlimmerModel
from executorch.examples.models.muse_glimmer.tests.test_pipeline import (
    build_random_tiny_model,
    HAS_QUANT_DEPS,
    TINY_CONFIG,
)

# MLX export fixes the prefill chunk at 512, which sets two minimums: the KV
# cache length, and the sliding-window size (the ring buffer requires
# max_write_len <= window). The shared tiny fixture uses a 64-token cache and a
# window of 16, so widen both here.
MLX_CONFIG = replace(TINY_CONFIG, max_seq_len=1024, global_attn_cfg="[512,512,512,0]")


def _require_mlx(testcase: unittest.TestCase) -> None:
    try:
        import executorch.backends.mlx.custom_ops  # noqa: F401
    except Exception as e:  # noqa: BLE001 — any import failure means MLX is absent
        testcase.skipTest(f"MLX backend required (OSS-only): {e}")


class MLXSourceTransformTest(unittest.TestCase):
    def setUp(self) -> None:
        _require_mlx(self)

    def test_swaps_attention_modules(self) -> None:
        """Each layer's attention is replaced with the MLX attention module."""
        from executorch.examples.models.muse_glimmer.source_transformations.mlx import (
            mlx_source_transformations,
            MLXMuseGlimmerAttention,
        )

        model = build_random_tiny_model()
        mlx_source_transformations(model, dtype=torch.bfloat16)

        for layer in model.layers:
            self.assertIsInstance(layer.self_attn, MLXMuseGlimmerAttention)

    def test_installs_prefill_entry_points(self) -> None:
        """The MLX transforms install the embed_text / prefill entry points used
        by the vision path (embeds-input prefill + host sampling)."""
        from executorch.examples.models.muse_glimmer.source_transformations.mlx import (
            mlx_source_transformations,
        )

        model = build_random_tiny_model()
        mlx_source_transformations(model, dtype=torch.bfloat16)

        for name in (
            "mlx_embed_text",
            "mlx_prefill_forward",
            "mlx_logits_from_embeds",
        ):
            self.assertTrue(callable(getattr(model, name, None)), name)


@unittest.skipUnless(HAS_QUANT_DEPS, "torchao quantization dependencies not available")
class MLXExportTest(unittest.TestCase):
    def setUp(self) -> None:
        _require_mlx(self)

    def test_export_produces_pte(self) -> None:
        """quantize -> pack (MLX) -> export lowers to a .pte via the MLX backend."""
        from executorch.examples.models.muse_glimmer.export.export_solo import (
            export_and_lower,
        )
        from executorch.examples.models.muse_glimmer.tests.test_pipeline import (
            DEFAULT_RECIPE,
        )
        from executorch.extension.llm.export.load import assign_state_dict
        from executorch.extension.llm.export.quant import quantize_model, to_default
        from executorch.runtime import Runtime, Verification

        model = build_random_tiny_model()
        state_dict = quantize_model(model, DEFAULT_RECIPE)

        with torch.device("meta"):
            model = MuseGlimmerModel(MLX_CONFIG)
        assign_state_dict(model, state_dict, convert=to_default)
        model.eval()

        with tempfile.TemporaryDirectory() as out_dir:
            export_and_lower(model, MLX_CONFIG, out_dir, backend="mlx")
            pte = os.path.join(out_dir, "model.pte")
            self.assertTrue(os.path.exists(pte))
            program = Runtime.get().load_program(pte, verification=Verification.Minimal)
            self.assertTrue(
                {
                    "embed_text",
                    "forward_from_embeddings",
                }.issubset(program.method_names)
            )
            self.assertNotIn("decode_from_embedding", program.method_names)

    def test_vision_export_produces_pte(self) -> None:
        """Vision is additive to the shared embeddings-input contract."""
        try:
            import gguf  # noqa: F401
        except (ImportError, ModuleNotFoundError):  # noqa: B014
            self.skipTest("gguf package required")

        from executorch.examples.models.muse_glimmer.export.export_solo import (
            export_and_lower,
        )
        from executorch.examples.models.muse_glimmer.loaders.checkpoint_loader import (
            load_mmproj_vision_model,
        )
        from executorch.examples.models.muse_glimmer.tests.test_pipeline import (
            _mmproj_vision_config,
            build_mmproj_gguf,
            DEFAULT_RECIPE,
        )
        from executorch.extension.llm.export.load import assign_state_dict
        from executorch.extension.llm.export.quant import quantize_model, to_default

        model = build_random_tiny_model()
        state_dict = quantize_model(model, DEFAULT_RECIPE)
        with torch.device("meta"):
            model = MuseGlimmerModel(MLX_CONFIG)
        assign_state_dict(model, state_dict, convert=to_default)
        model.eval()

        with tempfile.TemporaryDirectory() as tmp:
            gguf_path = os.path.join(tmp, "mmproj.gguf")
            build_mmproj_gguf(gguf_path)
            vision_model, pos_table, _ = load_mmproj_vision_model(
                gguf_path, config=_mmproj_vision_config(), backend="mlx"
            )
            out_dir = os.path.join(tmp, "out")
            export_and_lower(
                model,
                MLX_CONFIG,
                out_dir,
                backend="mlx",
                vision_model=vision_model,
                pos_embed_table=pos_table,
                max_vision_patches=1024,
            )
            self.assertTrue(os.path.exists(os.path.join(out_dir, "model.pte")))
            self.assertTrue(os.path.exists(os.path.join(out_dir, "pos_embed.bin")))


class MLXVisionLoaderTest(unittest.TestCase):
    """MLX packing keeps quantized vision weights as raw GGUF tensors (for the
    MLX quantized/gguf custom kernels) rather than the CUDA packed classes."""

    def setUp(self) -> None:
        try:
            import gguf  # noqa: F401
        except (ImportError, ModuleNotFoundError):  # noqa: B014
            self.skipTest("gguf package required")

    def test_mlx_backend_keeps_raw_gguf(self) -> None:
        from executorch.examples.models.muse_glimmer.loaders.checkpoint_loader import (
            load_mmproj_vision_model,
        )
        from executorch.examples.models.muse_glimmer.tests.test_pipeline import (
            _mmproj_vision_config,
            build_mmproj_gguf,
        )
        from executorch.extension.llm.export.gguf import ExportableGGUFTensor

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mmproj.gguf")
            build_mmproj_gguf(path)
            model, pos_table, _ = load_mmproj_vision_model(
                path, config=_mmproj_vision_config(), backend="mlx"
            )

        blk0 = model.blocks[0]
        # Q4_K and Q6_K weights alike stay raw ExportableGGUFTensor for MLX.
        self.assertIsInstance(blk0.attn.q_proj.weight.data, ExportableGGUFTensor)
        self.assertIsInstance(blk0.attn.v_proj.weight.data, ExportableGGUFTensor)
        self.assertIsInstance(model.vision_proj.weight.data, ExportableGGUFTensor)
        # Biases / norms / patch embed stay plain bf16 (both backends).
        self.assertEqual(blk0.attn.q_proj.bias.dtype, torch.bfloat16)
        self.assertEqual(model.conv1_linear.weight.dtype, torch.bfloat16)
        self.assertEqual(pos_table.dtype, torch.bfloat16)


if __name__ == "__main__":
    # run.py executes this module via runpy without installing it as
    # sys.modules["__main__"], so unittest.main()'s default __main__ discovery
    # finds no tests. Collect this module's TestCases explicitly instead.
    import sys

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for _obj in list(globals().values()):
        if isinstance(_obj, type) and issubclass(_obj, unittest.TestCase):
            suite.addTests(loader.loadTestsFromTestCase(_obj))
    _result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if _result.wasSuccessful() else 1)
