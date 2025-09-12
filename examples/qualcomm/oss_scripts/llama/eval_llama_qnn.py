# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json

import logging
import sys
import types
from functools import partial

import torch
from executorch.backends.qualcomm.quantizer.custom_annotation import (
    annotate_kv_8bit,
    annotate_output_16a8w,
    annotate_qkv_proj_sha,
    StaticLLMQuantConfig,
)

from executorch.backends.qualcomm.quantizer.observers.per_channel_param_observer import (
    PerChannelParamObserver,
)
from executorch.backends.qualcomm.quantizer.qconfig import (
    _derived_bias_quant_spec,
    get_ptq_per_channel_quant_config,
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

from executorch.examples.qualcomm.oss_scripts.llama.decoder_utils import calibrate

from executorch.examples.qualcomm.oss_scripts.llama.model.static_llama import (
    LlamaModel,
    ModelArgs,
)
from executorch.examples.qualcomm.oss_scripts.llama.range_setting_pt2e import (
    compute_scales,
    make_custom_quantizer,
    reverse_quantize_module_swap,
    set_scales,
    WrappedLlamaModel,
)
from lm_eval.evaluator import simple_evaluate

from pytorch_tokenizers import get_tokenizer
from torchao.prototype.spinquant import apply_spinquant
from torchao.quantization.pt2e import MinMaxObserver

from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from torchao.quantization.pt2e.quantizer import QuantizationSpec


sys.setrecursionlimit(4096)
FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logging.getLogger().setLevel(logging.INFO)


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


def prepare_model(model_name, args):
    with open(args.params) as f:
        prefill_config = ModelArgs(**json.load(f))
        # TODO: support batch inputs if necessary
        prefill_config.max_batch_size = 1
        prefill_config.max_seq_len = args.max_seq_length
        prefill_config.use_kv_cache = False
    use_i64_token = args.embedding_quantize is not None
    model = LlamaModel(
        prefill_config,
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

    # TODO: use dtype of model checkpoint
    model = model.to(device=args.device, dtype=torch.float)
    inputs = model.get_example_inputs(use_kv_cache=False)
    tokens, atten_mask = inputs

    scales_state_dict = {}
    if args.spinquant:
        config = types.SimpleNamespace(
            dim=prefill_config.dim,
            head_dim=prefill_config.dim // prefill_config.n_heads,
            n_local_heads=prefill_config.n_heads,
            intermediate_size=4 * prefill_config.dim,
        )
        model.config = config
        apply_spinquant(
            model,
            use_r1=True,
            use_r2=True,
            use_r4=False,
            pretrained_rotation_path=None,
            qkv_split=True,
        )
        logging.info("Applied SpinQuant to the model")

    if args.range_setting == "mse_with_act_loss":
        wrapped_model = WrappedLlamaModel(
            model, atten_mask, args.use_kv_cache, args.max_seq_length, args.device
        )
        act_bits, weight_bits = {
            "8a8w": (8, 8),
            "16a4w": (16, 4),
            "16a4w_block": (16, 4),
        }[args.ptq]
        scales_state_dict = compute_scales(
            wrapped_model, tokens, weight_bits, act_bits, 1600
        )
        torch.save(scales_state_dict, "scales_state_dict.pth")
        logging.info("Saved scales to scales_state_dict.pth!")
        reverse_quantize_module_swap(wrapped_model)

    for layer in model.layers:
        if getattr(layer.attention, "prepare_sha", None):
            layer.attention.prepare_sha()
        if getattr(layer.feed_forward, "prepare_feedfoward_conv", None):
            layer.feed_forward.prepare_feedfoward_conv()
    if args.embedding_quantize:
        model = get_quant_embedding_transform(
            embedding_quantize=args.embedding_quantize
        )(model)

    model = convert_linear_to_conv2d(model)
    return model, prefill_config, inputs, scales_state_dict


def gen_eval_wrapper(model_name, args):
    tokenizer = get_tokenizer(args.tokenizer_path)
    model, config, inputs, scales_state_dict = prepare_model(model_name, args)
    tokens, atten_mask = inputs
    use_i64_token = args.embedding_quantize is not None

    if args.ptq is not None:
        quant_dtype = getattr(QuantDtype, f"use_{args.ptq}")

        quantization_config_wv_sha_8a4w = get_ptq_per_channel_quant_config(
            act_dtype=torch.uint8,
            weight_dtype=torch.int4,
            act_observer=MinMaxObserver,
            act_symmetric=True,
        )
        custom_annotations = (
            annotate_kv_8bit,
            partial(
                annotate_qkv_proj_sha,
                qkv_tags={StaticLLMQuantConfig.wv_sha},
                quantization_config=quantization_config_wv_sha_8a4w,
            ),
        )
        if args.llama_model == "stories110m":
            custom_annotations = custom_annotations + (annotate_output_16a8w,)

        quantizer = make_custom_quantizer(
            quant_dtype, args.range_setting, custom_annotations, args.quant_linear_only
        )

        with torch.no_grad():
            logging.info("Starting export...")
            model = torch.export.export(model, inputs, strict=True).module()
            if quant_dtype == QuantDtype.use_16a4w_block:
                conv_nodes = [n for n in model.graph.nodes if "conv" in n.name]
                block_size_map = {n.name: (1, 64, 1, 1) for n in conv_nodes}
                quantizer.set_block_size_map(block_size_map)
            logging.info("Finished export, adding observers (prepare_pt2e)...")
            model = prepare_pt2e(model, quantizer)

        logging.info("Observers added, starting calibration...")

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

        if args.range_setting == "mse_with_act_loss":
            # scales_state_dict = torch.load("scales_state_dict.pth")
            set_scales(model, scales_state_dict, config.head_dim)

        logging.info("Quantizing the model...")
        model = convert_pt2e(model)
        logging.info("Quantization complete! Here is some sample generated text:")

        calibrate(
            inputs,
            "Could you tell me about Facebook?",
            model,
            tokenizer=tokenizer,
            ar_len=args.prefill_ar_len,
            max_seq_len=args.max_seq_len,
            kv_updater=None,
            use_i64_token=use_i64_token,
        )

    model = WrappedLlamaModel(
        model, atten_mask, args.use_kv_cache, args.max_seq_length, args.device
    )

    return GraphModuleEvalWrapper(
        model=model,
        tokenizer=tokenizer,
        max_seq_length=args.calibration_seq_length,
        use_kv_cache=args.use_kv_cache,
        generate_full_logits=args.generate_full_logits,
        enable_dynamic_shape=False,
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
            limit=args.fraction,
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
        help="Choose which range setting method for weight quantization (e.g. mse_weight_only or mse_with_act_loss). If not specified, defaults to minmax",
        type=str,
    )
    parser.add_argument(
        "--spinquant",
        help="Apply SpinQuant (R1+R2) to the model. Uses random Hadamard matrices for rotations",
        action="store_true",
    )
    parser.add_argument(
        "--fraction",
        help="the fraction of examples per task (only use this for testing)",
        type=float,
    )
    parser.add_argument(
        "--quant_linear_only",
        help="if you select this option we quantize linear layers only",
        action="store_true",
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
