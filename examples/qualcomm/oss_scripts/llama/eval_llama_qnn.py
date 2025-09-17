# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

""" Utilities for running fast evals (using prefill mode version of model) on eager-quantized model and QDQ model, for experimentation purposes. """

import json

import logging
import sys
import types

import torch

from executorch.backends.qualcomm.quantizer.observers.per_channel_param_observer import (
    PerChannelParamObserver,
)
from executorch.backends.qualcomm.quantizer.qconfig import (
    _derived_bias_quant_spec,
    QuantizationConfig,
)

from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.backends.qualcomm.utils.utils import convert_linear_to_conv2d

from executorch.examples.models.llama.eval_llama_lib import build_args_parser
from executorch.examples.models.llama.hf_download import (
    download_and_convert_hf_checkpoint,
)

from executorch.examples.models.llama.source_transformation.quantize import (
    get_quant_embedding_transform,
)
from executorch.examples.qualcomm.oss_scripts.llama import SUPPORTED_LLM_MODELS

from executorch.examples.qualcomm.oss_scripts.llama.decoder_utils import (
    graph_module_inference,
)

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
from pytorch_tokenizers import get_tokenizer, TiktokenTokenizer
from pytorch_tokenizers.llama2c import Llama2cTokenizer as SentencePieceTokenizer
from torchao.prototype.quantization.module_swap.module_swap import (
    QuantizationRecipe,
    quantize_module_swap,
)
from torchao.prototype.spinquant import apply_spinquant

from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from torchao.quantization.pt2e.quantizer import QuantizationSpec
from transformers import AutoTokenizer


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


def prepare_tokenizer(args):
    runtime_tokenizer_path = ""
    if args.decoder_model in {"stories110m", "stories260k"}:
        tokenizer = get_tokenizer(args.tokenizer_model)
        assert isinstance(
            tokenizer, SentencePieceTokenizer
        ), "Wrong tokenizer provided for stories."
        assert (
            args.tokenizer_bin is not None
        ), "Please provide tokenizer_bin for stories."
        runtime_tokenizer_path = args.tokenizer_bin
    elif args.decoder_model == "llama3_2":
        tokenizer = get_tokenizer(args.tokenizer_model)
        assert isinstance(
            tokenizer, TiktokenTokenizer
        ), "Wrong tokenizer provided for llama3_2."
        runtime_tokenizer_path = args.tokenizer_model
    elif args.decoder_model == "phi_4_mini":
        model_id = SUPPORTED_LLM_MODELS[args.decoder_model].repo_id
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        runtime_tokenizer_path = tokenizer.save_pretrained(args.artifact)[-1]
        tokenizer = get_tokenizer(runtime_tokenizer_path)
        with open(runtime_tokenizer_path, "r+") as file:
            data = json.load(file)
            # TODO: Encountered the following error during runtime, so switched behavior for now.
            # Error: libc++abi: terminating due to uncaught exception of type std::runtime_error: invert=true is not supported for Split PreTokenizer. Only invert=false is supported.
            data["pre_tokenizer"]["pretokenizers"][-2]["invert"] = False
            file.seek(0)
            json.dump(data, file, indent=4)
            file.truncate()
    elif args.decoder_model in SUPPORTED_LLM_MODELS:
        model_id = SUPPORTED_LLM_MODELS[args.decoder_model].repo_id
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        runtime_tokenizer_path = tokenizer.save_pretrained(args.artifact)[-1]
        tokenizer = get_tokenizer(runtime_tokenizer_path)
    else:
        raise RuntimeError(f"Unknown decoder_model: {args.decoder_model}.")
    return tokenizer


def prepare_model(args):
    if args.params:
        params_path = args.params
    else:
        params_path = SUPPORTED_LLM_MODELS[args.decoder_model].params_path
    with open(params_path) as f:
        prefill_config = ModelArgs(**json.load(f))
    # TODO: support batch inputs if necessary
    prefill_config.max_batch_size = 1
    prefill_config.max_seq_len = args.max_seq_length
    prefill_config.use_kv_cache = False
    prefill_config.enable_r3 = args.r3
    use_i64_token = args.embedding_quantize is not None
    model = LlamaModel(
        prefill_config,
        ar_len=args.prefill_ar_len,
        output_new_cache_only=True,
        output_cache=False,
        use_i64_token=use_i64_token,
    )
    if args.checkpoint is None:  # HF models
        checkpoint = download_and_convert_hf_checkpoint(
            SUPPORTED_LLM_MODELS[args.decoder_model].repo_id,
            SUPPORTED_LLM_MODELS[args.decoder_model].convert_weights.__func__,
        )
        state_dict = torch.load(
            checkpoint, weights_only=True, map_location=args.device, mmap=True
        )
        transform_weight = SUPPORTED_LLM_MODELS[args.decoder_model].transform_weight
    else:
        state_dict = torch.load(
            args.checkpoint, weights_only=True, map_location=args.device, mmap=True
        )

        if "model" in state_dict:
            state_dict = state_dict["model"]

        if args.decoder_model == "stories260k":
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        transform_weight = True

    if transform_weight:
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
    return model, prefill_config


def prequant_algorithm(model, prefill_config, args):
    # TODO: use dtype of model checkpoint
    model = model.to(device=args.device, dtype=torch.float)
    inputs = model.get_example_inputs(use_kv_cache=False)
    tokens, atten_mask = inputs
    tokens.to(args.device)
    for mask in atten_mask.masks:
        mask.mask.to(args.device)

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


def eager_eval_quanty(
    model,
    weight_bits,
    act_bits,
    embedding_quantization,
    dynamic_activations=False,
    dynamic_weights=False,
):
    """
    Run evaluations where we quantize only linear layers with Quanty (eager-mode module swap quantization flow)
    Although when lowering to Qualcomm backend using the PT2E flow we quantize all (not just linear) layers,
    Quanty flow is fast and can be used for rapid experimentation.
    """

    recipe = QuantizationRecipe(
        weight_bits=weight_bits,
        weight_quantization=True,
        dynamic_weights=dynamic_weights,
        weight_group_size="per_channel",
        activation_bits=act_bits,
        activation_quantization=True,
        activation_group_size="per_tensor",
        input_quantization=True,
        output_quantization=True,
        dynamic_activations=dynamic_activations,
        embedding_quantization=embedding_quantization,
    )

    quantized_model = quantize_module_swap(model, recipe)
    simple_evaluate(
        model=model,
        tasks=["wikitext"],
    )

    reverse_quantize_module_swap(quantized_model)


def eval_llm(args):
    tokenizer = prepare_tokenizer(args)
    model, prefill_config = prepare_model(args)
    model, config, inputs, scales_state_dict = prequant_algorithm(
        model, prefill_config, args
    )
    use_i64_token = args.embedding_quantize is not None

    if args.ptq is not None:
        quant_dtype = getattr(QuantDtype, f"use_{args.ptq}")
        decoder_model_config = SUPPORTED_LLM_MODELS[args.decoder_model]
        custom_annotations = decoder_model_config.custom_annotation

        quantizer = make_custom_quantizer(
            quant_dtype, args.range_setting, custom_annotations, args.quant_linear_only
        )

        with torch.no_grad():
            logging.info("Starting export...")
            model = torch.export.export(
                model, (inputs[0], *inputs[1]), strict=True
            ).module()
            if quant_dtype == QuantDtype.use_16a4w_block:
                conv_nodes = [n for n in model.graph.nodes if "conv" in n.name]
                block_size_map = {n.name: (1, 64, 1, 1) for n in conv_nodes}
                quantizer.set_block_size_map(block_size_map)
            logging.info("Finished export, adding observers (prepare_pt2e)...")
            model = prepare_pt2e(model, quantizer)

        logging.info("Observers added, starting calibration...")
        graph_module_inference(
            use_kv_cache=False,
            get_example_inputs=lambda use_kv_cache=False: inputs,
            module=model,
            tokenizer=tokenizer,
            ar_len=args.max_seq_len,
            max_seq_len=args.max_seq_len,
            kv_updater=args.kv_updater,
            tasks=["wikitext"],
            tasks_limit=1,
            use_i64_token=use_i64_token,
            event_name="prepare_pt2e_prompt",
        )

        if args.range_setting == "mse_with_act_loss":
            # scales_state_dict = torch.load("scales_state_dict.pth")
            set_scales(model, scales_state_dict, config.head_dim)

        logging.info("Quantizing the model...")
        model = convert_pt2e(model)
        logging.info("Quantization complete! Here is some sample generated text:")

        graph_module_inference(
            use_kv_cache=False,
            get_example_inputs=lambda use_kv_cache=False: inputs,
            module=model,
            tokenizer=tokenizer,
            ar_len=args.max_seq_len,
            max_seq_len=args.max_seq_len,
            kv_updater=args.kv_updater,
            prompt="Can you tell me about Facebook?",
            use_i64_token=use_i64_token,
            event_name="convert_pt2e_prompt",
        )

    logging.info("Evaluation of QDQ model:")
    graph_module_inference(
        use_kv_cache=False,
        get_example_inputs=lambda use_kv_cache=False: inputs,
        module=model,
        tokenizer=tokenizer,
        ar_len=args.max_seq_len,
        max_seq_len=args.max_seq_len,
        kv_updater=args.kv_updater,
        tasks=["wikitext"],
        use_i64_token=use_i64_token,
        event_name="convert_pt2e_prompt",
    )


def main() -> None:
    seed = 42
    torch.manual_seed(seed)
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
    parser.add_argument(
        "--kv_updater",
        help="Choose how to update kv cache during runtime",
        choices=["smart_mask", "shift_pointer"],
        default="smart_mask",
        type=str,
    )
    parser.add_argument(
        "--decoder_model",
        choices=["stories260k", "stories110m", "llama3_2"]
        + list(SUPPORTED_LLM_MODELS.keys()),
        help=f"The Llama model to export. Current available options are: [stories260k, stories110m, llama3_2] + {SUPPORTED_LLM_MODELS.keys()}",
        required=True,
    )
    parser.add_argument(
        "-a",
        "--artifact",
        help="path for storing generated artifacts and output by this example. Default ./llama_qnn",
        default="./eval_llama_qnn",
        type=str,
    )
    parser.add_argument(
        "--r3",
        help="Enable SpinQuant R3 quantization optimization. Please notice enable R3 could possibly cause performance drop.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--tokenizer_model",
        help="Pass llama tokenizer model.",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    # Overrides this arg, because evaluation requires full logits.
    args.generate_full_logits = True

    args.max_seq_len = args.max_seq_length
    args.calibration_seq_length = args.max_seq_length

    # Prefill mode
    args.use_kv_cache = False
    args.prefill_ar_len = args.max_seq_length

    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(args.device)

    eval_llm(args)


if __name__ == "__main__":
    main()  # pragma: no cover
