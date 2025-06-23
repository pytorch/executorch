# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import copy
import json

import logging
import sys

from typing import List, Tuple

import torch
import torch.nn as nn
from executorch.backends.qualcomm.quantizer.custom_annotation import (
    annotate_linear_16a8w_in_affine_layer,
    annotate_matmul_16a8w,
)

from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.backends.qualcomm.utils.utils import convert_linear_to_conv2d

from executorch.examples.models.llama.eval_llama_lib import (
    build_args_parser,
    GraphModuleEvalWrapper,
)

from executorch.examples.models.llama.source_transformation.quantize import (
    get_quant_embedding_transform,
)

from executorch.examples.qualcomm.oss_scripts.llama.llama import calibrate

from executorch.examples.qualcomm.oss_scripts.llama.model.static_llama import (
    LlamaModel,
    ModelArgs,
)

from executorch.examples.qualcomm.utils import make_quantizer

from lm_eval.evaluator import simple_evaluate

from pytorch_tokenizers import get_tokenizer

from torchao.quantization.pt2e import MinMaxObserver
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

sys.setrecursionlimit(4096)
FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logging.getLogger().setLevel(logging.INFO)


class WrappedLlamaModel(nn.Module):
    def __init__(
        self, model, atten_mask, use_kv_cache=False, max_seq_len=512, device="cuda"
    ):
        super(WrappedLlamaModel, self).__init__()
        self.model = model
        self.max_seq_len = max_seq_len
        self.use_kv_cache = use_kv_cache
        self.device = device
        self.atten_mask = atten_mask

    def forward(
        self,
        tokens: torch.Tensor,
        *args,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        # Pad input if necessary, since LlamaModel requires static shape
        if tokens.shape[1] != self.max_seq_len:
            tokens = torch.nn.functional.pad(
                tokens, (0, self.max_seq_len - tokens.shape[1])
            )
        return self.model.forward(tokens, self.atten_mask)


def gen_eval_wrapper(model_name, args):
    tokenizer = get_tokenizer(args.tokenizer_path)
    with open(args.params) as f:
        kv_config = ModelArgs(**json.load(f))
        # TODO: support batch inputs if necessary
        kv_config.max_batch_size = 1
        kv_config.max_seq_len = args.max_seq_length
        kv_config.use_kv_cache = True

        prefill_config = copy.copy(kv_config)
        prefill_config.max_seq_len = args.max_seq_length
        prefill_config.use_kv_cache = (
            False if args.max_seq_length == args.prefill_ar_len else True
        )
    config = prefill_config
    use_i64_token = args.embedding_quantize is not None
    model = LlamaModel(
        config,
        ar_len=args.prefill_ar_len,
        output_new_cache_only=True,
        output_cache=False,
        use_i64_token=use_i64_token,
    )
    state_dict = torch.load(
        args.checkpoint, weights_only=True, map_location=args.device, mmap=True
    )

    # Change to HuggingFace weight to improve the performance of RoPE in HTP backend.
    def permute(w, heads):
        dim_0 = w.size(0)
        dim_1 = w.size(1)
        return (
            w.view(heads, dim_0 // heads // 2, 2, dim_1)
            .transpose(1, 2)
            .reshape(dim_0, dim_1)
        )

    n_heads = model.n_heads
    n_kv_heads = model.n_kv_heads
    n_layers = model.n_layers

    for layer_i in range(n_layers):
        state_dict[f"layers.{layer_i}.attention.wq.weight"] = permute(
            state_dict[f"layers.{layer_i}.attention.wq.weight"], n_heads
        )
        state_dict[f"layers.{layer_i}.attention.wk.weight"] = permute(
            state_dict[f"layers.{layer_i}.attention.wk.weight"], n_kv_heads
        )

    model.load_state_dict(
        state_dict,
        strict=True,
        assign=True,
    )

    if "model" in state_dict:
        state_dict = state_dict["model"]

    for layer in model.layers:
        if getattr(layer.attention, "prepare_sha", None):
            layer.attention.prepare_sha()
        if getattr(layer.feed_forward, "prepare_feedfoward_conv", None):
            layer.feed_forward.prepare_feedfoward_conv()

    model.to(dtype=torch.bfloat16)
    model.to(device=args.device)

    tokens, atten_mask = model.get_example_inputs(use_kv_cache=False)
    tokens = tokens.to(device=args.device)
    atten_mask = atten_mask.to(device=args.device)
    atten_mask = atten_mask.to(dtype=torch.bfloat16)
    inputs = (tokens, atten_mask)

    if args.embedding_quantize:
        model = get_quant_embedding_transform(
            embedding_quantize=args.embedding_quantize
        )(model)

    model = convert_linear_to_conv2d(model)

    if args.ptq:
        quant_dtype = getattr(QuantDtype, f"use_{args.ptq}")

        custom_annotations = (annotate_matmul_16a8w,)
        if args.llama_model == "stories110m":
            custom_annotations = custom_annotations + (
                annotate_linear_16a8w_in_affine_layer,
            )
        quantizer = make_quantizer(
            quant_dtype=quant_dtype,
            per_channel_conv=True,
            per_channel_linear=True,
            act_observer=MinMaxObserver,
        )
        quantizer.add_custom_quant_annotations(custom_annotations)

        model.has_quant_io = True

        with torch.no_grad():
            model = torch.export.export(model, inputs, strict=True).module()
            if quant_dtype == QuantDtype.use_16a4w_block:
                conv_nodes = [n for n in model.graph.nodes if "conv" in n.name]
                block_size_map = {n.name: (1, 64, 1, 1) for n in conv_nodes}
                quantizer.set_block_size_map(block_size_map)

            model = prepare_pt2e(model, quantizer)

        logging.info("Quantizing the model...")

        calibrate(
            inputs,
            "Once upon a time",
            model,
            tokenizer=tokenizer,
            ar_len=args.prefill_ar_len,
            max_seq_len=args.max_seq_len,
            kv_updater=None,
            use_i64_token=use_i64_token,
        )

        model = convert_pt2e(model)

    model = WrappedLlamaModel(
        model, atten_mask, args.use_kv_cache, args.max_seq_length, args.device
    )

    return GraphModuleEvalWrapper(
        model=model,
        tokenizer=tokenizer,
        max_seq_length=args.calibration_seq_length,
        use_kv_cache=args.use_kv_cache,
        generate_full_logits=args.generate_full_logits,
        enable_dynamic_shape=args.enable_dynamic_shape,
    )


def eval_llama(
    model_name: str,
    args: argparse.Namespace,
) -> None:
    # Generate the eval wrapper
    eval_wrapper = gen_eval_wrapper(model_name, args)

    # Needed for loading mmlu dataset.
    # See https://github.com/EleutherAI/lm-evaluation-harness/pull/1998/files
    if args.tasks and "mmlu" in args.tasks:
        import datasets

        datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True
    # Evaluate the model
    with torch.no_grad():
        eval_results = simple_evaluate(
            model=eval_wrapper,
            tasks=args.tasks,
            num_fewshot=args.num_fewshot,
            limit=args.limit,
        )

    for task, res in eval_results["results"].items():
        print(f"{task}: {res}")


def main() -> None:
    seed = 42
    torch.manual_seed(seed)
    modelname = "llama2"
    parser = build_args_parser()
    args = parser.parse_args()
    args.llama_model = "llama3_2"
    # Overrides this arg, because evaluation requires full logits.
    args.generate_full_logits = True

    args.max_seq_len = args.max_seq_length
    args.calibration_seq_length = args.max_seq_length

    # Prefill mode
    args.use_kv_cache = False
    args.prefill_ar_len = args.max_seq_length

    # To do fewer samples for faster evaluation
    args.limit = 0.1
    # args.samples = {'wikitext': list(range(1))}

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(args.device)

    args.ptq = "8a8w"

    eval_llama(modelname, args)


if __name__ == "__main__":
    main()  # pragma: no cover
