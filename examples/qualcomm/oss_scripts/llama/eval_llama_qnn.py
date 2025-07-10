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

from executorch.backends.qualcomm.quantizer.observers.per_channel_param_observer import (
    PerChannelParamObserver,
)
from executorch.backends.qualcomm.quantizer.qconfig import (
    _derived_bias_quant_spec,
    QuantizationConfig,
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
from torchao.quantization.pt2e.quantizer import QuantizationSpec


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


def add_mse_weight_observer(quant_dtype, quantizer):
    weight_dtype = (
        torch.int4
        if quant_dtype in (QuantDtype.use_16a4w, QuantDtype.use_16a4w_block)
        else torch.int8
    )
    per_channel_q_config = quantizer.default_quant_config.quant_config
    weight_qspec = QuantizationSpec(
        dtype=torch.int8 if weight_dtype == torch.int4 else weight_dtype,
        quant_min=(
            -7 if weight_dtype == torch.int4 else torch.iinfo(weight_dtype).min + 1
        ),
        quant_max=(7 if weight_dtype == torch.int4 else torch.iinfo(weight_dtype).max),
        qscheme=torch.per_channel_symmetric,
        ch_axis=0,
        observer_or_fake_quant_ctr=PerChannelParamObserver.with_args(
            **{"steps": 200, "use_mse": True}
        ),
    )
    quantizer.default_quant_config.per_channel_quant_config = QuantizationConfig(
        input_activation=per_channel_q_config.input_activation,
        output_activation=per_channel_q_config.output_activation,
        weight=weight_qspec,
        bias=_derived_bias_quant_spec,
    )


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

    model.to(dtype=torch.float)
    model.to(device=args.device)

    tokens, atten_mask = model.get_example_inputs(use_kv_cache=False)
    tokens = tokens.to(device=args.device)
    atten_mask = atten_mask.to(device=args.device)
    atten_mask = atten_mask.to(dtype=torch.float)
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

        if args.range_setting == "mse_weight":
            add_mse_weight_observer(quant_dtype, quantizer)

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
    parser.add_argument(
        "-P",
        "--ptq",
        help="If specified, will do PTQ quantization. default is 16bits activation and 4bits weight. Support 8a8w, 16a4w and 16a4w_block.",
        type=str,
    )
    parser.add_argument(
        "--range_setting",
        help="Choose which range setting method (e.g. mse_weight). If not specified, will do minmax for weights and activations",
        type=str,
    )
    parser.add_argument(
        "--limit",
        help="the number of examples per task (only use this for testing), If <1, limit is a percentage of the total number of examples",
        type=str,
    )

    args = parser.parse_args()
    args.llama_model = "llama3_2"
    # Overrides this arg, because evaluation requires full logits.
    args.generate_full_logits = True

    args.max_seq_len = args.max_seq_length
    args.calibration_seq_length = args.max_seq_length

    # Prefill mode
    args.use_kv_cache = False
    args.prefill_ar_len = args.max_seq_length

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(args.device)

    eval_llama(modelname, args)


if __name__ == "__main__":
    main()  # pragma: no cover
