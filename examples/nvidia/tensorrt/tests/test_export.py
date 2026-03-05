# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export correctness tests for TensorRT backend.

Exports each supported model with TensorRT and compares inference
outputs against eager PyTorch.
"""

import io
import logging
import os
import shutil
import unittest

import torch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Mapping from env var to expected cache filename.
# When set (e.g., by CI), these env vars point to pre-downloaded weight files
# that get copied into the torch cache to avoid network downloads.
_WEIGHT_ENV_VARS = {
    "DOG_JPG": "dog.jpg",
    "EDSR_WEIGHTS": "edsr64_x2.pt",
    "EFFICIENT_SAM_WEIGHTS": "efficient_sam_vitt.pt",
    "EMFORMER_WEIGHTS": "emformer_rnnt_base_librispeech.pt",
    "IC4_WEIGHTS": "inceptionv4-8e4777a0.pth",
    "MV2_WEIGHTS": "mobilenet_v2-b0353104.pth",
    "MV3_WEIGHTS": "mobilenet_v3_small-047dcff4.pth",
    "RESNET18_WEIGHTS": "resnet18-f37072fd.pth",
    "RESNET50_WEIGHTS": "resnet50-0676ba61.pth",
}


def _populate_weight_cache() -> None:
    """Copy Manifold-cached weights to torch/HF cache so models skip downloads."""
    cache_dir = os.path.join(
        os.environ.get("TORCH_HOME", os.path.expanduser("~/.cache/torch")),
        "hub",
        "checkpoints",
    )
    os.makedirs(cache_dir, exist_ok=True)
    for env_var, filename in _WEIGHT_ENV_VARS.items():
        src = os.environ.get(env_var)
        if src and os.path.isfile(src):
            if env_var == "DOG_JPG":
                dst = os.path.join(os.getcwd(), filename)
            elif env_var == "EMFORMER_WEIGHTS":
                torchaudio_dir = os.path.join(
                    os.environ.get("TORCH_HOME", os.path.expanduser("~/.cache/torch")),
                    "hub", "torchaudio", "models",
                )
                os.makedirs(torchaudio_dir, exist_ok=True)
                dst = os.path.join(torchaudio_dir, filename)
            else:
                dst = os.path.join(cache_dir, filename)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                logger.info(f"Cached {filename} from {src}")


_populate_weight_cache()


def _export_and_verify(model_name: str) -> None:
    """Export a model and verify its outputs against eager PyTorch."""
    from executorch.examples.nvidia.tensorrt.export import (
        _verify_correctness,
        export_model,
    )

    model, example_inputs, exec_prog = export_model(
        model_name, ".", True, logger,
    )
    buf = io.BytesIO()
    exec_prog.write_to_file(buf)
    _verify_correctness(model_name, model, example_inputs, buf.getvalue(), logger)


class ExportCorrectnessTest(unittest.TestCase):
    """Export each model with TensorRT and verify outputs against eager PyTorch."""

    def test_add(self) -> None:
        _export_and_verify("add")

    def test_add_mul(self) -> None:
        _export_and_verify("add_mul")

    def test_mul(self) -> None:
        _export_and_verify("mul")

    def test_add_bf16(self) -> None:
        """Test add model with bf16 precision."""
        from executorch.backends.nvidia.tensorrt.compile_spec import TensorRTCompileSpec, TensorRTPrecision
        from executorch.backends.nvidia.tensorrt.partitioner import TensorRTPartitioner
        from executorch.examples.models import MODEL_NAME_TO_MODEL
        from executorch.examples.models.model_factory import EagerModelFactory
        from executorch.exir import to_edge_transform_and_lower
        import torch
        from torch.export import export
        model, example_inputs, _, _ = EagerModelFactory.create_model(*MODEL_NAME_TO_MODEL["add"])
        model = model.eval()
        exported = export(model, example_inputs, strict=True)
        spec = TensorRTCompileSpec(precision=TensorRTPrecision.BF16)
        edge = to_edge_transform_and_lower(exported, partitioner=[TensorRTPartitioner(spec.to_compile_specs())])
        exec_prog = edge.to_executorch()
        self.assertIsNotNone(exec_prog)
        logger.info("PASS: add model exported with BF16 precision")

    def test_softmax(self) -> None:
        _export_and_verify("softmax")

    def test_mv3(self) -> None:
        _export_and_verify("mv3")

    def test_linear(self) -> None:
        _export_and_verify("linear")
    def test_conv1d(self) -> None:
        _export_and_verify("conv1d")

    def test_dl3(self) -> None:
        _export_and_verify("dl3")


    def test_w2l(self) -> None:
        _export_and_verify("w2l")

    def test_edsr(self) -> None:
        _export_and_verify("edsr")

    def test_ic3(self) -> None:
        _export_and_verify("ic3")

    def test_sdpa(self) -> None:
        _export_and_verify("sdpa")

    def test_emformer_join(self) -> None:
        _export_and_verify("emformer_join")

    def test_emformer_transcribe(self) -> None:
        _export_and_verify("emformer_transcribe")

    def test_softmax(self) -> None:
        _export_and_verify("softmax")

    def test_mv3(self) -> None:
        _export_and_verify("mv3")

    def test_ic3(self) -> None:
        _export_and_verify("ic3")

    def test_ic4(self) -> None:
        _export_and_verify("ic4")

    def test_mv2(self) -> None:
        _export_and_verify("mv2")

    def test_resnet18(self) -> None:
        _export_and_verify("resnet18")

    def test_resnet50(self) -> None:
        _export_and_verify("resnet50")

    def test_efficient_sam(self) -> None:
        _export_and_verify("efficient_sam")
