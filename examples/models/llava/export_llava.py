# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackDynamicallyQuantizedPartitioner,
    # XnnpackFloatingPointPartitioner,
)
from executorch.examples.models.llama2.export_llama_lib import (
    build_args_parser,
    get_quantizer_and_quant_params,
)
from executorch.examples.models.llama2.source_transformation.quantize import (
    get_quant_weight_transform,
)
from executorch.examples.models.llama2.source_transformation.sdpa import (
    replace_sdpa_with_custom_op,
)
from executorch.exir import EdgeCompileConfig, to_edge

from executorch.extension.llm.export.builder import DType, LLMEdgeManager
from model import LlavaModel
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
from torch.export import Dim
from torch.nn.attention import SDPBackend


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
    )

    dtype_override = DType.fp32
    parser = build_args_parser()
    args = parser.parse_args(
        ["-X", "-qmode", "8da4w", "--group_size", "128", "--embedding-quantize", "4,32"]
    )
    quant_transform = get_quant_weight_transform(args, dtype_override, False)
    pt2e_quant_params, quantizers, quant_dtype = get_quantizer_and_quant_params(args)

    manager = (
        text_model_em.set_output_dir("./")
        .to_dtype(dtype_override)
        .source_transform([replace_sdpa_with_custom_op, quant_transform])
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
    linear_quantizer = XNNPACKQuantizer()
    operator_config_dynamic = get_symmetric_quantization_config(
        is_per_channel=True, is_dynamic=True
    )
    linear_quantizer.set_global(operator_config_dynamic)

    manager = LlavaEdgeManager(
        model=llava_image_encode,
        modelname="llava_image_encoder",
        max_seq_len=llava.text_model_args.max_seq_len,  # This may not be right
        dtype=DType.fp32,
        use_kv_cache=True,
        example_inputs=(resized,),
        dynamic_shapes=dynamic_shapes,
    ).capture_pre_autograd_graph()

    # lower to executorch
    with torch.no_grad():
        image_encoder_ep = torch.export.export(
            manager.pre_autograd_graph_module,
            manager.example_inputs,
            dynamic_shapes=manager.dynamic_shapes,
        )
    return image_encoder_ep


def export_token_embedding(llava, prompt):
    embed = torch.nn.Embedding(
        llava.model_.config.vocab_size,
        llava.model_.config.hidden_size,
        llava.model_.config.pad_token_id,
    )
    embed.load_state_dict(
        llava.model_.get_model().embed_tokens.state_dict(), strict=True, assign=True
    )
    embed = embed.to(torch.float32)
    token_dim_1 = Dim("token_dim_1", min=2, max=3518)
    dynamic_shapes = [{1: token_dim_1}]
    with torch.no_grad():
        token_embedding_ep = torch.export.export(
            embed, (prompt,), dynamic_shapes=dynamic_shapes
        )
    return token_embedding_ep


def main():
    llava_model = LlavaModel()
    llava = llava_model.get_eager_model()

    prompt_before_image, resized, prompt_after_image = (
        llava_model.get_inputs_for_prefill()
    )

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

    edge_ep = to_edge(
        {
            "image_encoder": image_encoder_ep,
            "token_embedding": token_embedding_ep,
            "text_model": text_model_ep,
        },
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )

    executorch_program = edge_ep.to_backend(
        {
            # TODO: Fix Xnnpack partitioner issue on image encoder.
            # "image_encoder": XnnpackFloatingPointPartitioner(),
            "text_model": XnnpackDynamicallyQuantizedPartitioner(),
        }
    ).to_executorch()

    with open("llava_combined_xnnpack.pte", "wb") as f:
        executorch_program.write_to_file(f)


if __name__ == "__main__":
    main()
