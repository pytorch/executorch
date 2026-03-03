# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Export LFM2.5-VL-1.6B as a single multi-method PTE for ExecuTorch's
generic MultimodalRunner (C++ llava_main).

Methods:
  vision_encoder  : [1, 3, 512, 512] f32 NCHW pixels [0,255] -> [1, 256, 2048] f32
  token_embedding : [1, seq_len] i64                          -> [1, seq_len, 2048] f32
  text_decoder    : ([1, seq_len, 2048] f32, [seq_len] i64)   -> [1, 65536] f32

Usage:
    python examples/models/lfm2_5_vl/export_lfm2_5_vl.py \
        --model_dir /path/to/LFM2-VL-1.6B \
        [--dtype fp32|fp16] [--quantize] [--output lfm2_5_vl_xnnpack.pte]
"""

import logging
from argparse import ArgumentParser, BooleanOptionalAction

import torch
from executorch.backends.xnnpack.partition.config.xnnpack_config import (
    ConfigPrecisionType,
)
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.examples.models.llama.export_llama_lib import (
    get_quantizer_and_quant_params,
)
from executorch.examples.models.llama.source_transformation.custom_kv_cache import (
    replace_kv_cache_with_custom_kv_cache,
)
from executorch.examples.models.llama.source_transformation.quantize import (
    EmbeddingQuantHandler,
    get_quant_weight_transform,
)
from executorch.examples.models.llama.source_transformation.sdpa import (
    replace_sdpa_with_custom_op,
)
from executorch.examples.models.lfm2_5_vl.model import (
    Lfm2p5VlModel,
    MAX_SEQ_LEN,
    IMAGE_SIZE,
)
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower
from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.passes.quant_fusion_pass import QuantFusionPass
from executorch.exir.passes.sym_shape_eval_pass import (
    ConstraintBasedSymShapeEvalPass,
    HintBasedSymShapeEvalPass,
)
from executorch.extension.llm.export.builder import DType, LLMEdgeManager
from executorch.extension.llm.export.config.llm_config import LlmConfig
from torch.export import Dim
from torch.nn.attention import SDPBackend

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


class Lfm2p5VlEdgeManager(LLMEdgeManager):
    """LLMEdgeManager subclass for LFM2.5-VL.

    Overrides export() to use SDPBackend.MATH (avoids dynamo tracing errors)
    and strict=False (required for hybrid conv layer buffer mutations).
    Mirrors LlavaEdgeManager in examples/models/llava/export_llava.py.
    """

    def export(self) -> "Lfm2p5VlEdgeManager":
        dynamic_shape = self._get_dynamic_shape()
        with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
            self.export_program = torch.export.export(
                self.model,
                self.example_inputs,
                dynamic_shapes=dynamic_shape,
                strict=False,
            )
            self.pre_autograd_graph_module = self.export_program.module()
        return self


def export_image_encoder(lfm2) -> torch.export.ExportedProgram:
    """Export vision encoder as 'vision_encoder' method.

    Input:  [1, 3, 512, 512] float32 NCHW pixels in [0, 255]
    Output: [1, 256, 2048]   float32 image embeddings

    Normalize + patch extraction are baked in so the C++ runner only
    needs to resize to 512x512 and pass the raw pixel buffer.
    """

    class ImageEncoder(torch.nn.Module):
        def __init__(self, lfm2):
            super().__init__()
            self.lfm2 = lfm2

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return self.lfm2.image_embedding(images)

    encoder = ImageEncoder(lfm2)
    example_pixels = torch.randint(0, 256, (1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32)

    logging.info("Exporting vision encoder...")
    with torch.no_grad():
        ep = torch.export.export(encoder, (example_pixels,), strict=False)
    return ep


def export_text_decoder(
    lfm2,
    quantize: bool,
    dtype: DType = DType.fp32,
) -> torch.export.ExportedProgram:
    """Export hybrid LFM2.5 decoder as 'text_decoder' method.

    Uses Lfm2p5VlEdgeManager (strict=False, SDPBackend.MATH).
    enable_dynamic_shape=False in ModelArgs avoids .item() in rope.get_freqs.
    """

    class TextDecoder(torch.nn.Module):
        def __init__(self, text_model):
            super().__init__()
            self.text_model = text_model

        def forward(self, embeddings: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
            return self.text_model(None, {"input_pos": input_pos}, embeddings)

    decoder = TextDecoder(lfm2.text_model)
    dim = lfm2.text_model_args.dim
    dummy_seq = 8
    dummy_embeddings = torch.randn(1, dummy_seq, dim, dtype=dtype.to_torch_dtype())
    dummy_input_pos = torch.arange(dummy_seq, dtype=torch.int64)
    token_dim = Dim("token_dim", min=1, max=MAX_SEQ_LEN)
    dynamic_shapes = ({1: token_dim}, {0: token_dim})

    manager = Lfm2p5VlEdgeManager(
        model=decoder,
        modelname="lfm2_5_vl_text_decoder",
        max_seq_len=MAX_SEQ_LEN,
        dtype=dtype,
        use_kv_cache=True,
        example_inputs=(dummy_embeddings, dummy_input_pos),
        dynamic_shapes=dynamic_shapes,
    )

    source_transforms = [
        replace_kv_cache_with_custom_kv_cache,
        replace_sdpa_with_custom_op,
    ]

    if quantize:
        llm_config = LlmConfig()
        llm_config.quantization.qmode = "8da4w"
        llm_config.quantization.group_size = 128
        quant_transform = get_quant_weight_transform(
            quantization_mode=llm_config.quantization.qmode,
            group_size=llm_config.quantization.group_size,
            computation_dtype=dtype,
            checkpoint_path=None,
            tokenizer_path=None,
            calibration_tasks=None,
            calibration_limit=None,
            calibration_seq_length=None,
        )
        _, quantizers, _ = get_quantizer_and_quant_params(llm_config)
        source_transforms.append(quant_transform)
        logging.info("Exporting text decoder (8da4w quantized)...")
        manager = (
            manager.set_output_dir("./")
            .to_dtype(dtype)
            .source_transform(source_transforms)
            .export()
            .pt2e_quantize(quantizers)
        )
    else:
        logging.info("Exporting text decoder (fp32)...")
        manager = (
            manager.set_output_dir("./")
            .to_dtype(dtype)
            .source_transform(source_transforms)
            .export()
        )

    with torch.no_grad():
        decoder_ep = torch.export.export(
            manager.pre_autograd_graph_module,
            manager.example_inputs,
            dynamic_shapes=manager._get_dynamic_shape(),
            strict=True,
        )
    return decoder_ep


def export_token_embedding(
    lfm2,
    quantize: bool,
) -> torch.export.ExportedProgram:
    """Export token embedding table as 'token_embedding' method.

    Uses fixed MAX_SEQ_LEN=2048 input buffer. The C++ runner pads shorter
    sequences to this length before calling, then slices the output back.

    IMPORTANT: call this AFTER export_text_decoder. EmbeddingQuantHandler
    mutates model.embed_tokens in-place (replaces fp32 weight with int8).
    """
    language_model = lfm2.model_.model.language_model
    token_dim = Dim("token_dim_1", min=1, max=MAX_SEQ_LEN)
    dynamic_shapes = [{1: token_dim}]

    if quantize:
        logging.info("Exporting token embedding (int8 quantized)...")
        quantized_lm = EmbeddingQuantHandler(
            language_model, bitwidth=8, group_size=32, packed=False
        ).quantized_model()
        embed_module = quantized_lm.embed_tokens
    else:
        logging.info("Exporting token embedding (fp32)...")
        embed_module = language_model.get_input_embeddings()

    example_ids = torch.zeros(1, MAX_SEQ_LEN, dtype=torch.int64)
    with torch.no_grad():
        ep = torch.export.export(
            embed_module, (example_ids,), dynamic_shapes=dynamic_shapes, strict=True
        )
    return ep


def export_all(
    model_dir: str,
    output: str,
    dtype: DType = DType.fp32,
    quantize: bool = False,
) -> None:
    logging.info(f"Loading {model_dir}...")
    lfm2_model = Lfm2p5VlModel(model_dir=model_dir)
    lfm2 = lfm2_model.get_eager_model()
    if dtype != DType.fp32:
        lfm2 = lfm2.to(dtype.to_torch_dtype())

    logging.info("[1/3] Exporting vision encoder...")
    vision_ep = export_image_encoder(lfm2)

    # Text decoder MUST come before token embedding (see export_token_embedding docstring)
    logging.info("[2/3] Exporting text decoder...")
    decoder_ep = export_text_decoder(lfm2, quantize, dtype)

    logging.info("[3/3] Exporting token embedding...")
    token_ep = export_token_embedding(lfm2, quantize)

    logging.info("Lowering to Edge IR + XNNPACK...")
    lowered = to_edge_transform_and_lower(
        {
            "vision_encoder": vision_ep,
            "token_embedding": token_ep,
            "text_decoder": decoder_ep,
        },
        partitioner={
            "vision_encoder": [XnnpackPartitioner()],
            "token_embedding": [XnnpackPartitioner()],
            "text_decoder": [
                XnnpackPartitioner(
                    config_precisions=ConfigPrecisionType.DYNAMIC_QUANT,
                    per_op_mode=True,
                ),
                XnnpackPartitioner(),
            ] if quantize else [XnnpackPartitioner()],
        },
        constant_methods={
            "get_max_seq_len": MAX_SEQ_LEN,
            # EOS = <|im_end|> (token 7). Runner reads this from PTE rather than
            # relying on tokenizer default.
            "get_eos_ids": [7],
        },
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )

    logging.info("Finalizing ExecuTorch program...")
    et_program = lowered.to_executorch(
        ExecutorchBackendConfig(
            extract_delegate_segments=True,
            passes=[QuantFusionPass()] if quantize else [],
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            sym_shape_eval_pass={
                "vision_encoder": ConstraintBasedSymShapeEvalPass(),
                "token_embedding": HintBasedSymShapeEvalPass(),
                "text_decoder": ConstraintBasedSymShapeEvalPass(),
            },
        )
    )

    logging.info(f"Saving {output}...")
    with open(output, "wb") as f:
        et_program.write_to_file(f)
    logging.info(f"Saved {output}. Methods: {et_program.methods}")


def main():
    parser = ArgumentParser(description="Export LFM2.5-VL-1.6B to ExecuTorch")
    parser.add_argument(
        "--model_dir",
        default="LiquidAI/LFM2-VL-1.6B",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--dtype",
        default="fp32",
        choices=["fp32", "fp16"],
        help="Model dtype (default: fp32)",
    )
    parser.add_argument(
        "--quantize",
        default=False,
        action=BooleanOptionalAction,
        help="Quantize decoder (8da4w) and embedding (int8)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PTE path (default: lfm2_5_vl[_fp16][_quantized]_xnnpack.pte)",
    )
    args = parser.parse_args()

    dtype = DType.fp16 if args.dtype == "fp16" else DType.fp32
    suffix = ("_fp16" if dtype == DType.fp16 else "") + ("_quantized" if args.quantize else "")
    output = args.output or f"lfm2_5_vl{suffix}_xnnpack.pte"

    export_all(args.model_dir, output, dtype, args.quantize)


if __name__ == "__main__":
    main()
