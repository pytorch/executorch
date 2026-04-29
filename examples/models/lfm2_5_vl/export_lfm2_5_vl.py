# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Export LFM2.5-VL as a single multi-method PTE for ExecuTorch's generic
MultimodalRunner (C++ llava_main). Supports both LFM2.5-VL-1.6B (text dim
2048) and LFM2.5-VL-450M (text dim 1024); the architecture config is picked
from the bundled config/ directory based on --model_dir, or you can pass
--params to point at a custom JSON.

Methods (D = text hidden dim: 2048 for 1.6B, 1024 for 450M):
  vision_encoder  : [1, 3, 512, 512] f32 NCHW pixels [0,255] -> [1, 256, D] f32
  token_embedding : [1, seq_len] i64                          -> [1, seq_len, D] f32
  text_decoder    : ([1, seq_len, D] f32, [seq_len] i64)      -> [1, 65536] f32

Usage:
    python examples/models/lfm2_5_vl/export_lfm2_5_vl.py \
        --model_dir LiquidAI/LFM2.5-VL-450M \
        [--dtype fp32|fp16] [--quantize] [--output lfm2_5_vl_xnnpack.pte]
"""

import logging
import os
from argparse import ArgumentParser, BooleanOptionalAction
from typing import Optional

import torch
from executorch.backends.xnnpack.partition.config.xnnpack_config import (
    ConfigPrecisionType,
)
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
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
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.passes.quant_fusion_pass import QuantFusionPass
from executorch.exir.passes.sym_shape_eval_pass import ConstraintBasedSymShapeEvalPass
from executorch.extension.llm.export.builder import DType, LLMEdgeManager
from executorch.extension.llm.export.config.llm_config import LlmConfig
from torch.export import Dim
from torch.nn.attention import SDPBackend

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")


def _resolve_params_path(model_dir: str, params: Optional[str]) -> Optional[str]:
    """Pick a bundled config based on model_dir if --params was not provided.

    Returns None to fall back to model.py's default (1.6B).
    """
    if params is not None:
        return params
    name = model_dir.lower()
    if "450m" in name:
        return os.path.join(_CONFIG_DIR, "lfm2_5_vl_450m_config.json")
    if "1.6b" in name or "1_6b" in name:
        return os.path.join(_CONFIG_DIR, "lfm2_5_vl_1_6b_config.json")
    return None


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


def export_image_encoder(lfm2, quantize: bool = False) -> torch.export.ExportedProgram:
    """Export vision encoder as 'vision_encoder' method.

    Input:  [1, 3, 512, 512] float32 NCHW pixels in [0, 255]
    Output: [1, 256, 2048]   float32 image embeddings

    Normalize + patch extraction are baked in so the C++ runner only
    needs to resize to 512x512 and pass the raw pixel buffer.

    When quantize=True, mirrors LLaVA's export_image_encoder: uses
    LLMEdgeManager.export().pt2e_quantize() so quantization happens on
    the pre-autograd graph (aten.linear still intact), then re-exports
    the quantized graph for to_edge_transform_and_lower.
    """

    class ImageEncoder(torch.nn.Module):
        def __init__(self, lfm2):
            super().__init__()
            self.lfm2 = lfm2

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return self.lfm2.image_embedding(images)

    encoder = ImageEncoder(lfm2)
    example_pixels = torch.randint(
        0, 256, (1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32
    )

    if quantize:
        logging.info("Exporting vision encoder (int8 dynamic quantized)...")
        quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
        manager = (
            Lfm2p5VlEdgeManager(
                model=encoder,
                modelname="lfm2_5_vl_image_encoder",
                max_seq_len=MAX_SEQ_LEN,
                dtype=DType.fp32,
                use_kv_cache=False,
                example_inputs=(example_pixels,),
            )
            .export()
            .pt2e_quantize([quantizer])
        )
        with torch.no_grad():
            ep = torch.export.export(
                manager.pre_autograd_graph_module,
                manager.example_inputs,
                strict=False,
            )
    else:
        logging.info("Exporting vision encoder (fp32)...")
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

        def forward(
            self, embeddings: torch.Tensor, input_pos: torch.Tensor
        ) -> torch.Tensor:
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

    return manager.export_program


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
            embed_module, (example_ids,), dynamic_shapes=dynamic_shapes, strict=False
        )
    return ep


def export_all(
    model_dir: str,
    output: Optional[str],
    dtype: DType = DType.fp32,
    quantize: bool = False,
    max_seq_len: int = MAX_SEQ_LEN,
    max_context_len: int = MAX_SEQ_LEN,
    params_path: Optional[str] = None,
    _return_program: bool = False,
):
    logging.info(f"Loading {model_dir}...")
    lfm2_model = Lfm2p5VlModel(
        model_dir=model_dir,
        max_seq_len=max_seq_len,
        max_context_len=max_context_len,
        params_path=params_path,
    )
    lfm2 = lfm2_model.get_eager_model()
    if dtype != DType.fp32:
        lfm2 = lfm2.to(dtype.to_torch_dtype())

    logging.info("[1/3] Exporting vision encoder...")
    vision_ep = export_image_encoder(lfm2, quantize=False)

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
            ]
            if quantize
            else [XnnpackPartitioner()],
        },
        constant_methods={
            "get_max_seq_len": lfm2.text_model_args.max_seq_len,
            "get_max_context_len": lfm2.text_model_args.max_context_len,
            "get_n_layers": lfm2.text_model_args.n_layers,
            "get_vocab_size": lfm2.text_model_args.vocab_size,
            "use_kv_cache": lfm2.text_model_args.use_kv_cache,
            "use_sdpa_with_kv_cache": lfm2.text_model_args.use_sdpa_with_kv_cache_op,
            "enable_dynamic_shape": lfm2.text_model_args.enable_dynamic_shape,
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
                "token_embedding": ConstraintBasedSymShapeEvalPass(),
                "text_decoder": ConstraintBasedSymShapeEvalPass(),
            },
        )
    )

    for execution_plan in et_program._emitter_output.program.execution_plan:
        logging.info(
            f"Required memory for activation in bytes: {execution_plan.non_const_buffer_sizes}"
        )

    if _return_program:
        return et_program

    logging.info(f"Saving {output}...")
    with open(output, "wb") as f:  # type: ignore[arg-type]
        et_program.write_to_file(f)
    logging.info(f"Saved {output}. Methods: {et_program.methods}")


def main():
    parser = ArgumentParser(description="Export LFM2.5-VL to ExecuTorch")
    parser.add_argument(
        "--model_dir",
        default="LiquidAI/LFM2-VL-1.6B",
        help=(
            "HuggingFace model ID or local path. Supported: "
            "LiquidAI/LFM2-VL-1.6B, LiquidAI/LFM2.5-VL-450M."
        ),
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
        "--max_seq_len",
        type=int,
        default=MAX_SEQ_LEN,
        help=f"Maximum sequence length (default: {MAX_SEQ_LEN})",
    )
    parser.add_argument(
        "--max_context_len",
        type=int,
        default=MAX_SEQ_LEN,
        help=f"Maximum context length (default: {MAX_SEQ_LEN})",
    )
    parser.add_argument(
        "--params",
        default=None,
        help=(
            "Path to model params JSON (architecture config). When omitted, "
            "the bundled 1.6B or 450M config is selected from --model_dir."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PTE path (default: lfm2_5_vl[_fp16][_quantized]_xnnpack.pte)",
    )
    args = parser.parse_args()

    dtype = DType.fp16 if args.dtype == "fp16" else DType.fp32
    params_path = _resolve_params_path(args.model_dir, args.params)
    size_tag = "_450m" if (params_path or "").endswith("450m_config.json") else ""
    suffix = (
        size_tag
        + ("_fp16" if dtype == DType.fp16 else "")
        + ("_quantized" if args.quantize else "")
    )
    output = args.output or f"lfm2_5_vl{suffix}_xnnpack.pte"

    export_all(
        args.model_dir,
        output,
        dtype,
        args.quantize,
        args.max_seq_len,
        args.max_context_len,
        params_path,
    )


if __name__ == "__main__":
    main()
