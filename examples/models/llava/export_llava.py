# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from argparse import ArgumentParser, BooleanOptionalAction

import torch
from executorch.backends.xnnpack.partition.config.xnnpack_config import (
    ConfigPrecisionType,
)
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.examples.models.llama2.export_llama_lib import (
    build_args_parser,
    get_quantizer_and_quant_params,
)
from executorch.examples.models.llama2.source_transformation.quantize import (
    EmbeddingQuantHandler,
    get_quant_weight_transform,
)
from executorch.examples.models.llama2.source_transformation.sdpa import (
    replace_sdpa_with_custom_op,
)
from executorch.examples.models.llava.image_util import serialize_image
from executorch.examples.models.llava.model import LlavaModel
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)

from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.passes.quant_fusion_pass import QuantFusionPass
from executorch.exir.passes.sym_shape_eval_pass import (
    ConstraintBasedSymShapeEvalPass,
    HintBasedSymShapeEvalPass,
)

from executorch.extension.llm.export.builder import DType, LLMEdgeManager
from executorch.extension.llm.tokenizer.tokenizer import Tokenizer
from executorch.util.activation_memory_profiler import generate_memory_trace
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
from torch.export import Dim
from torch.nn.attention import SDPBackend

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


class LlavaEdgeManager(LLMEdgeManager):
    def capture_pre_autograd_graph(self) -> "LlavaEdgeManager":
        dynamic_shape = self._get_dynamic_shape()
        # 1. torch.nn.attention.sdpa_kernel([SDPBackend.MATH]) is for bypassing the dynamo error when tracing
        # 2. torch.no_grad() is for getting rid of the dropout (not sure why training ops will show up)
        with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
            self.export_program = torch.export.export(
                self.model,
                self.example_inputs,
                dynamic_shapes=dynamic_shape,
                strict=False,
            )
            self.pre_autograd_graph_module = self.export_program.module()
        return self


def export_text_model(llava, embeddings, dynamic_shapes):
    class LlavaTextModel(torch.nn.Module):
        """Takes images and prompts and encode them into embeddings. Result will be sent to the text model LlavaTextModel."""

        def __init__(self, llava):
            super().__init__()
            self.text_model = llava.text_model

        def forward(self, input_pos, embeddings):
            return self.text_model(None, input_pos, embeddings)

    llava_text_model = LlavaTextModel(llava)

    text_model_em = LLMEdgeManager(
        model=llava_text_model,
        modelname="llava_text_model",
        max_seq_len=llava.text_model_args.max_seq_len,
        dtype=DType.fp32,
        use_kv_cache=True,
        example_inputs=(torch.tensor([0], dtype=torch.int64), embeddings),
        dynamic_shapes=dynamic_shapes,
        args=llava.text_model_args,
    )

    dtype_override = DType.fp32
    parser = build_args_parser()
    args = parser.parse_args(
        ["-X", "-qmode", "8da4w", "--group_size", "128", "--embedding-quantize", "4,32"]
    )
    quant_transform = get_quant_weight_transform(args, dtype_override, False)
    _, quantizers, _ = get_quantizer_and_quant_params(args)
    source_transforms = []
    if llava.use_sdpa_with_kv_cache_op:
        source_transforms.append(replace_sdpa_with_custom_op)
    source_transforms.append(quant_transform)
    manager = (
        text_model_em.set_output_dir("./")
        .to_dtype(dtype_override)
        .source_transform(source_transforms)
        .capture_pre_autograd_graph()
        .pt2e_quantize(quantizers)
    )

    with torch.no_grad():
        text_model_ep = torch.export.export(
            manager.pre_autograd_graph_module,
            manager.example_inputs,
            dynamic_shapes=manager._get_dynamic_shape(),
        )
    return text_model_ep


def export_image_encoder(llava, resized, dynamic_shapes):
    class LlavaImageEncoder(torch.nn.Module):
        """Takes images and prompts and encode them into embeddings. Result will be sent to the text model LlavaTextModel."""

        def __init__(self, llava):
            super().__init__()
            self.llava = llava

        def forward(self, images):
            return self.llava.image_embedding(images)

    llava_image_encode = LlavaImageEncoder(llava)

    # quantizer
    quantizer = XNNPACKQuantizer()
    quantizer.set_global(get_symmetric_quantization_config())

    manager = (
        LlavaEdgeManager(
            model=llava_image_encode,
            modelname="llava_image_encoder",
            max_seq_len=llava.text_model_args.max_seq_len,  # This may not be right
            dtype=DType.fp32,
            use_kv_cache=True,
            example_inputs=(resized,),
            dynamic_shapes=dynamic_shapes,
            args=None,
        )
        .capture_pre_autograd_graph()
        .pt2e_quantize([quantizer])
    )

    # lower to executorch
    with torch.no_grad():
        image_encoder_ep = torch.export.export(
            manager.pre_autograd_graph_module,
            manager.example_inputs,
            dynamic_shapes=manager.dynamic_shapes,
        )
    return image_encoder_ep


def export_token_embedding(llava, prompt):
    def quant_embedding(model):
        return EmbeddingQuantHandler(
            model,
            bitwidth=8,
            group_size=32,
            packed=False,
        ).quantized_model()

    quantized_token_embed = quant_embedding(llava.model_.language_model.model)
    token_dim_1 = Dim("token_dim_1", min=2, max=llava.text_model_args.max_seq_len)
    dynamic_shapes = [{1: token_dim_1}]
    with torch.no_grad():
        token_embedding_ep = torch.export.export(
            quantized_token_embed.embed_tokens, (prompt,), dynamic_shapes=dynamic_shapes
        )
    return token_embedding_ep


def export_all(llava_model: LlavaModel):
    llava = llava_model.get_eager_model()

    (
        prompt_before_image,
        resized,
        prompt_after_image,
    ) = llava_model.get_inputs_for_prefill()

    image_encoder_ep = export_image_encoder(
        llava, resized, llava_model._get_image_dynamic_shapes()
    )

    embeddings = llava.prefill_embedding(
        prompt_before_image, resized, prompt_after_image
    )

    text_model_ep = export_text_model(
        llava, embeddings, llava_model._get_prompt_dynamic_shapes()
    )

    token_embedding_ep = export_token_embedding(llava, prompt_before_image)

    lowered_and_edge = to_edge_transform_and_lower(
        {
            "image_encoder": image_encoder_ep,
            "token_embedding": token_embedding_ep,
            "text_model": text_model_ep,
        },
        partitioner={
            "image_encoder": [XnnpackPartitioner()],
            "text_model": [
                # First partition the DQLinear nodes, then partition the rest of the nodes,
                # to avoid multiple DQLinear nodes in the same partition,
                # to avoid holding multiple unpacked and packed weight buffers in memory,
                # to reduce peak memory footprint.
                XnnpackPartitioner(
                    config_precisions=ConfigPrecisionType.DYNAMIC_QUANT,
                    per_op_mode=True,
                ),
                XnnpackPartitioner(),
            ],
        },
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )

    executorch_program = lowered_and_edge.to_executorch(
        ExecutorchBackendConfig(
            extract_delegate_segments=True,
            passes=[
                QuantFusionPass(),
            ],
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            sym_shape_eval_pass={
                "image_encoder": ConstraintBasedSymShapeEvalPass(),
                "text_model": ConstraintBasedSymShapeEvalPass(),
                "token_embedding": HintBasedSymShapeEvalPass(),
            },
        )
    )
    for execution_plan in executorch_program._emitter_output.program.execution_plan:
        logging.info(
            f"Required memory for activation in bytes: {execution_plan.non_const_buffer_sizes}"
        )
    return executorch_program


def get_image_tensor_for_llava_runner(llava_model):
    # llava runner doesn't have image reader so an image tensor is needed.
    (resized,) = llava_model.get_example_inputs()

    serialize_image(resized, "image.pt")


def get_tokenizer_for_llava_runner(llava_model):
    # serialize tokenizer into tokenizer.bin
    llava_model.tokenizer.save_vocabulary("./")
    t = Tokenizer("tokenizer.model")
    t.export("tokenizer.bin")


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--use-sdpa-with-kv-cache",
        default=True,
        action=BooleanOptionalAction,
        help="Use sdpa_with_kv_cache custom op in LLava text model.",
    )
    parser.add_argument(
        "--max-seq-len",
        default=768,
        type=int,
        help="Maximum sequence length for the text model.",
    )
    parser.add_argument(
        "--pte-name",
        default="llava_combined_xnnpack.pte",
        help="Name of the exported ExecuTorch program.",
    )
    parser.add_argument(
        "--with-artifacts",
        default=False,
        action=BooleanOptionalAction,
        help="Generate artifacts for llava runner.",
    )
    parser.add_argument(
        "--profile_memory",
        required=False,
        action="store_true",
        help="Generate chrome trace of activation memory for intermediate tensors.",
    )
    args = parser.parse_args()
    logging.info(
        f"Exporting Llava model to ExecuTorch with sdpa_with_kv_cache: {args.use_sdpa_with_kv_cache}, max_seq_len: {args.max_seq_len}"
    )
    llava_model = LlavaModel(
        use_sdpa_with_kv_cache_op=args.use_sdpa_with_kv_cache,
        max_seq_len=args.max_seq_len,
    )

    executorch_program = export_all(llava_model)

    # memory profiling
    if args.profile_memory:
        for method_name in executorch_program.methods:
            generate_memory_trace(
                executorch_program,
                f"{args.pte_name}_{method_name}.json",
                method_name=method_name,
            )

    with open(args.pte_name, "wb") as f:
        executorch_program.write_to_file(f)
    logging.info(f"Exported ExecuTorch program to {args.pte_name}")

    # artifacts
    if args.with_artifacts:
        get_image_tensor_for_llava_runner(llava_model)
        get_tokenizer_for_llava_runner(llava_model)


if __name__ == "__main__":
    main()
