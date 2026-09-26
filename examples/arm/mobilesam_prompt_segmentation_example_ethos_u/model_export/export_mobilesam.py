# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import sys
from pathlib import Path
from typing import Any, cast

import executorch.kernels.quantized  # noqa: F401

import numpy as np
import torch
import torch.nn.functional as F
from executorch.backends.arm.common.pipeline_config import (
    ArmPassPipelineConfig,
    SoftmaxDecompositionConfig,
)
from executorch.backends.arm.ethosu import EthosUCompileSpec, EthosUPartitioner
from executorch.backends.arm.quantizer import (
    EthosUQuantizer,
    get_symmetric_a16w8_quantization_config,
    get_symmetric_quantization_config,
)
from executorch.devtools.backend_debug import get_delegation_info
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.extension.export_util.utils import save_pte_program
from PIL import Image
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


ROOT = Path(__file__).resolve().parents[4]
WORK_DIR = ROOT / "arm_test" / "mobilesam"
SOURCE_DIR = (
    Path.home()
    / ".cache"
    / "executorch"
    / "mobilesam"
    / "f706ad9c4eb7f219c00d9050e46328518ffb65d2"
    / "source"
)
CHECKPOINT = SOURCE_DIR.parent / "mobile_sam.pt"
IMAGE = ROOT / "examples" / "models" / "dinov2" / "dog.jpg"
POINT = (219.0, 193.0)
INPUT_SIZE = 448
MINIMUM_IOU = 0.9


class MobileSAMFixedPrompt(torch.nn.Module):
    def __init__(self, sam: torch.nn.Module) -> None:
        super().__init__()
        sam = cast(Any, sam)
        self.image_encoder = sam.image_encoder
        self.mask_decoder = sam.mask_decoder

        with torch.no_grad():
            points = (
                torch.tensor([[POINT]], dtype=torch.float32),
                torch.ones((1, 1), dtype=torch.int64),
            )
            sparse, dense = sam.prompt_encoder(points=points, boxes=None, masks=None)
            image_pe = sam.prompt_encoder.get_dense_pe()
        self.register_buffer("sparse_prompt", sparse)
        self.register_buffer("dense_prompt", dense)
        self.register_buffer("image_pe", image_pe)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        masks, _ = self.mask_decoder(
            image_embeddings=self.image_encoder(image),
            image_pe=self.image_pe,
            sparse_prompt_embeddings=self.sparse_prompt,
            dense_prompt_embeddings=self.dense_prompt,
            multimask_output=False,
        )
        return masks


def load_model() -> torch.nn.Module:
    if not SOURCE_DIR.exists() or not CHECKPOINT.exists():
        raise RuntimeError("Run prepare_mobilesam.py first.")
    sys.path.insert(0, str(SOURCE_DIR))
    from mobile_sam import sam_model_registry  # type: ignore[import-not-found]

    return sam_model_registry["vit_t"](
        checkpoint=str(CHECKPOINT), image_size=INPUT_SIZE
    ).eval()


def prepare_image(sam: torch.nn.Module) -> tuple[Image.Image, torch.Tensor]:
    image = Image.open(IMAGE).convert("RGB")
    scale = INPUT_SIZE / max(image.size)
    resized_size = (round(image.width * scale), round(image.height * scale))
    resized = image.resize(resized_size, Image.Resampling.BILINEAR)
    padded = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE))
    padded.paste(resized)

    sam = cast(Any, sam)
    mean = sam.pixel_mean.detach().cpu().reshape(3).numpy()
    std = sam.pixel_std.detach().cpu().reshape(3).numpy()
    tensor = torch.from_numpy((np.asarray(resized, dtype=np.float32) - mean) / std)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    tensor = F.pad(
        tensor, (0, INPUT_SIZE - resized.width, 0, INPUT_SIZE - resized.height)
    )
    return padded, tensor.contiguous()


def mask(logits: torch.Tensor) -> np.ndarray:
    return (logits.detach().cpu().squeeze().numpy() > 0).astype(np.uint8)


def iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = np.logical_and(first, second).sum()
    union = np.logical_or(first, second).sum()
    return 1.0 if union == 0 else float(intersection / union)


def quantize(
    model: torch.nn.Module, image: torch.Tensor, quantizer: EthosUQuantizer
) -> torch.export.ExportedProgram:
    exported = torch.export.export(model, (image,))
    prepared = prepare_pt2e(exported.module(), quantizer)
    prepared(image)
    return torch.export.export(convert_pt2e(prepared), (image,))


def main() -> None:
    export_dir = WORK_DIR / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    sam = load_model()
    input_image, example_input = prepare_image(sam)
    model = MobileSAMFixedPrompt(sam).eval()

    compile_spec = EthosUCompileSpec(
        "ethos-u85-256", memory_mode="Dedicated_Sram_384KB"
    )
    compile_spec.set_pass_pipeline_config(
        ArmPassPipelineConfig(softmax=SoftmaxDecompositionConfig.STABLE)
    )
    compile_spec.dump_intermediate_artifacts_to(str(export_dir / "artifacts"))

    quantizer = EthosUQuantizer(compile_spec)
    quantizer.set_global(get_symmetric_quantization_config())
    attention_type = next(
        type(module)
        for module in model.image_encoder.modules()
        if type(module).__name__ == "Attention"
    )
    # Int16 attention activations preserve the segmentation mask quality.
    quantizer.set_module_type(attention_type, get_symmetric_a16w8_quantization_config())

    with torch.no_grad():
        fp32_mask = mask(model(example_input))
        quantized = quantize(model, example_input, quantizer)
        quantized_mask = mask(quantized.module()(example_input))

    host_iou = iou(fp32_mask, quantized_mask)
    if host_iou < MINIMUM_IOU:
        raise RuntimeError(f"FP32/quantized mask IoU is too low: {host_iou:.4f}")

    edge = to_edge_transform_and_lower(
        quantized,
        partitioner=[EthosUPartitioner(compile_spec)],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    delegation = get_delegation_info(edge.exported_program().graph_module)
    if delegation.num_delegated_subgraphs != 1:
        raise RuntimeError("Expected one Ethos-U delegate.")

    program = edge.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=False)
    )
    save_pte_program(program, "mobilesam", output_dir=str(export_dir))

    input_image.save(export_dir / "input.png")
    Image.fromarray(fp32_mask * 255).save(export_dir / "fp32_mask.png")
    Image.fromarray(quantized_mask * 255).save(export_dir / "quantized_mask.png")
    example_input.numpy().astype(np.float32).tofile(export_dir / "input.bin")
    (export_dir / "delegation.txt").write_text(delegation.get_summary() + "\n")
    (export_dir / "metrics.json").write_text(
        json.dumps({"fp32_quantized_iou": host_iou}, indent=2) + "\n"
    )

    print(f"FP32/quantized mask IoU: {host_iou:.4f}")
    print(f"Saved {export_dir / 'mobilesam.pte'}")


if __name__ == "__main__":
    main()
